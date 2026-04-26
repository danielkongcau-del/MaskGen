from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.operation_explainer import build_operation_explanation_payload  # noqa: E402
from partition_gen.operation_types import OperationExplainerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark operation-level explainer over evidence JSON files.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ortools-time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--max-patch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-patch", type=int, default=16)
    parser.add_argument("--cost-profile", type=str, default="token_length_v1", choices=["heuristic_v1", "token_length_v1"])
    parser.add_argument("--token-encode-evidence-refs", action="store_true")
    parser.add_argument(
        "--independent-baseline-profile",
        type=str,
        default="both",
        choices=["both", "atoms_only", "polygon_only", "all"],
        help="Which independent baseline encoding to use. 'all' writes one row per baseline profile.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_evidence_paths(root: Path, split: str | None = None) -> Iterable[Path]:
    search_root = root / split if split and (root / split).exists() else root
    for path in sorted(search_root.rglob("*.json")):
        yield path


def _baseline_flags(profile: str) -> tuple[bool, bool]:
    if profile == "both":
        return True, True
    if profile == "atoms_only":
        return False, True
    if profile == "polygon_only":
        return True, False
    raise ValueError(f"Unsupported independent baseline profile: {profile}")


def _baseline_profiles(profile: str) -> List[str]:
    if profile == "all":
        return ["both", "atoms_only", "polygon_only"]
    return [profile]


def _config_for_baseline_profile(config: OperationExplainerConfig, profile: str) -> OperationExplainerConfig:
    include_polygon, include_atoms = _baseline_flags(profile)
    return replace(
        config,
        independent_include_face_polygon=include_polygon,
        independent_include_convex_atoms=include_atoms,
    )


def benchmark_one(path: Path, config: OperationExplainerConfig, *, independent_baseline_profile: str = "both") -> dict | None:
    payload = load_json(path)
    if payload.get("format") != "maskgen_explanation_evidence_v1":
        return None
    started = time.perf_counter()
    try:
        explanation = build_operation_explanation_payload(payload, config=config, source_tag=str(path.as_posix()))
        runtime_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = explanation["diagnostics"]
        validation = explanation["validation"]
        candidate_summary = explanation.get("candidate_summary", {})
        return {
            "source": str(path.as_posix()),
            "cost_profile": diagnostics.get("cost_profile"),
            "independent_baseline_profile": independent_baseline_profile,
            "independent_include_face_polygon": bool(config.independent_include_face_polygon),
            "independent_include_convex_atoms": bool(config.independent_include_convex_atoms),
            "token_encode_evidence_refs": bool(config.token_encode_evidence_refs),
            "token_relation_endpoint": int(config.token_relation_endpoint),
            "token_relation_type": int(config.token_relation_type),
            "token_label": int(config.token_label),
            "token_geometry_model": int(config.token_geometry_model),
            "face_count": diagnostics.get("face_count"),
            "patch_count": diagnostics.get("patch_count"),
            "candidate_count": diagnostics.get("candidate_count"),
            "raw_candidate_count": diagnostics.get("raw_candidate_count"),
            "deduplicated_candidate_count": diagnostics.get("deduplicated_candidate_count"),
            "dropped_duplicate_count": diagnostics.get("dropped_duplicate_count"),
            "valid_candidate_count": candidate_summary.get("valid_candidate_count"),
            "invalid_candidate_count": candidate_summary.get("invalid_candidate_count"),
            "selected_operation_count": diagnostics.get("selected_operation_count"),
            "operation_histogram": diagnostics.get("operation_histogram", {}),
            "role_histogram": diagnostics.get("role_histogram", {}),
            "residual_face_count": diagnostics.get("residual_face_count"),
            "residual_area_ratio": diagnostics.get("residual_area_ratio"),
            "total_independent_cost": diagnostics.get("total_independent_cost"),
            "total_operation_cost": diagnostics.get("total_operation_cost"),
            "total_compression_gain": diagnostics.get("total_compression_gain"),
            "selection_method": diagnostics.get("selection_method"),
            "solver_status": diagnostics.get("solver_status"),
            "global_optimal": diagnostics.get("global_optimal"),
            "validation_is_valid": validation.get("is_valid"),
            "false_cover_area_total": diagnostics.get("false_cover_area_total"),
            "false_cover_ratio_max": diagnostics.get("false_cover_ratio_max"),
            "hard_false_cover_candidate_count": candidate_summary.get("hard_false_cover_candidate_count"),
            "failure_reason_histogram": candidate_summary.get("failure_reason_histogram", {}),
            "failure_reasons": diagnostics.get("failure_reasons", []),
            "runtime_ms": runtime_ms,
        }
    except Exception as exc:
        return {
            "source": str(path.as_posix()),
            "cost_profile": config.cost_profile,
            "independent_baseline_profile": independent_baseline_profile,
            "independent_include_face_polygon": bool(config.independent_include_face_polygon),
            "independent_include_convex_atoms": bool(config.independent_include_convex_atoms),
            "token_encode_evidence_refs": bool(config.token_encode_evidence_refs),
            "token_relation_endpoint": int(config.token_relation_endpoint),
            "token_relation_type": int(config.token_relation_type),
            "token_label": int(config.token_label),
            "token_geometry_model": int(config.token_geometry_model),
            "error": str(exc),
            "validation_is_valid": False,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        }


def run_benchmark(
    evidence_root: Path,
    *,
    split: str | None = None,
    max_samples: int | None = None,
    config: OperationExplainerConfig,
    independent_baseline_profile: str = "both",
) -> List[dict]:
    rows: List[dict] = []
    sample_count = 0
    profiles = _baseline_profiles(independent_baseline_profile)
    for path in iter_evidence_paths(evidence_root, split=split):
        payload = load_json(path)
        if payload.get("format") != "maskgen_explanation_evidence_v1":
            continue
        for profile in profiles:
            profiled_config = _config_for_baseline_profile(config, profile)
            row = benchmark_one(path, profiled_config, independent_baseline_profile=profile)
            if row is not None:
                rows.append(row)
        sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break
    return rows


def main() -> None:
    args = parse_args()
    config = OperationExplainerConfig(
        max_patch_size=args.max_patch_size,
        max_candidates_per_patch=args.max_candidates_per_patch,
        ortools_time_limit_seconds=args.ortools_time_limit_seconds,
        cost_profile=args.cost_profile,
        token_encode_evidence_refs=bool(args.token_encode_evidence_refs),
    )
    rows = run_benchmark(
        args.evidence_root,
        split=args.split,
        max_samples=args.max_samples,
        config=config,
        independent_baseline_profile=args.independent_baseline_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
