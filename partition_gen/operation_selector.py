from __future__ import annotations

from typing import Dict, List, Sequence, Set

from partition_gen.operation_types import RESIDUAL, OperationCandidate, OperationExplainerConfig, OperationSelectionResult


def _candidate_allowed(candidate: OperationCandidate, config: OperationExplainerConfig) -> bool:
    if not candidate.valid:
        return False
    if candidate.operation_type == RESIDUAL:
        return True
    return bool(candidate.compression_gain > config.min_compression_gain)


def _greedy_select(
    candidates: Sequence[OperationCandidate],
    all_face_ids: Sequence[int],
    *,
    reason: str,
    config: OperationExplainerConfig,
) -> OperationSelectionResult:
    selected: List[str] = []
    covered: Set[int] = set()
    by_face_residual: Dict[int, OperationCandidate] = {}
    for candidate in candidates:
        if candidate.operation_type == RESIDUAL and len(candidate.covered_face_ids) == 1:
            by_face_residual[int(candidate.covered_face_ids[0])] = candidate

    high_level = [
        candidate
        for candidate in candidates
        if candidate.operation_type != RESIDUAL and _candidate_allowed(candidate, config)
    ]
    high_level.sort(key=lambda item: (-float(item.compression_gain), len(item.covered_face_ids), item.id))
    for candidate in high_level:
        face_set = set(candidate.covered_face_ids)
        if face_set & covered:
            continue
        selected.append(candidate.id)
        covered.update(face_set)

    residual_face_ids = []
    for face_id in sorted(int(value) for value in all_face_ids):
        if face_id in covered:
            continue
        residual = by_face_residual.get(face_id)
        if residual is not None and residual.valid:
            selected.append(residual.id)
            covered.add(face_id)
            residual_face_ids.append(face_id)

    objective = sum(float(candidate.compression_gain) for candidate in candidates if candidate.id in set(selected))
    return OperationSelectionResult(
        selected_candidate_ids=selected,
        residual_face_ids=residual_face_ids,
        objective_value=float(objective),
        solver_status=reason,
        selection_method="greedy_fallback",
        global_optimal=False,
        diagnostics={
            "cost_profile": config.cost_profile,
            "objective_scale": int(1 if config.cost_profile == "token_length_v1" else config.objective_scale),
            "covered_face_count": int(len(covered)),
            "all_face_count": int(len(set(int(value) for value in all_face_ids))),
            "fallback_reason": reason,
        },
    )


def select_operations_with_ortools(
    candidates: Sequence[OperationCandidate],
    all_face_ids: Sequence[int],
    config: OperationExplainerConfig,
) -> OperationSelectionResult:
    if not config.use_ortools:
        return _greedy_select(candidates, all_face_ids, reason="ortools_disabled", config=config)

    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        if not config.allow_greedy_fallback:
            raise
        return _greedy_select(candidates, all_face_ids, reason=f"ortools_import_failed:{exc}", config=config)

    model = cp_model.CpModel()
    variables = {candidate.id: model.NewBoolVar(candidate.id) for candidate in candidates}
    candidate_by_id = {candidate.id: candidate for candidate in candidates}

    for candidate in candidates:
        if not _candidate_allowed(candidate, config):
            model.Add(variables[candidate.id] == 0)

    for face_id in sorted(set(int(value) for value in all_face_ids)):
        covering = [variables[candidate.id] for candidate in candidates if face_id in set(candidate.covered_face_ids)]
        if not covering:
            if not config.allow_greedy_fallback:
                raise ValueError(f"No candidate covers face {face_id}")
            return _greedy_select(candidates, all_face_ids, reason=f"missing_candidate_for_face:{face_id}", config=config)
        model.Add(sum(covering) == 1)

    objective_terms = []
    objective_scale = 1 if config.cost_profile == "token_length_v1" else config.objective_scale
    for candidate in candidates:
        gain = int(round(float(candidate.compression_gain) * objective_scale))
        if gain != 0:
            objective_terms.append(gain * variables[candidate.id])
    model.Maximize(sum(objective_terms) if objective_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.ortools_time_limit_seconds)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not feasible:
        if not config.allow_greedy_fallback:
            raise RuntimeError(f"OR-Tools failed with status {status_name}")
        return _greedy_select(candidates, all_face_ids, reason=f"ortools_status:{status_name}", config=config)

    selected = [candidate.id for candidate in candidates if solver.BooleanValue(variables[candidate.id])]
    residual_face_ids = sorted(
        {
            int(face_id)
            for candidate_id in selected
            for face_id in candidate_by_id[candidate_id].covered_face_ids
            if candidate_by_id[candidate_id].operation_type == RESIDUAL
        }
    )
    objective = sum(float(candidate_by_id[candidate_id].compression_gain) for candidate_id in selected)
    return OperationSelectionResult(
        selected_candidate_ids=selected,
        residual_face_ids=residual_face_ids,
        objective_value=float(objective),
        solver_status=status_name,
        selection_method="ortools_cp_sat",
        global_optimal=bool(status == cp_model.OPTIMAL),
        diagnostics={
            "cost_profile": config.cost_profile,
            "objective_scale": int(objective_scale),
            "num_conflicts": int(solver.NumConflicts()),
            "num_branches": int(solver.NumBranches()),
            "wall_time": float(solver.WallTime()),
            "selected_count": int(len(selected)),
        },
    )
