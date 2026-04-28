from __future__ import annotations

from collections import Counter
import copy
from statistics import mean
from typing import Callable, List, Sequence

from partition_gen.manual_topology_placeholder_geometry import decode_topology_tokens_to_target
from partition_gen.manual_topology_sample_validation import validate_topology_tokens


GeometrySampler = Callable[[dict, int], tuple[dict | None, dict]]
ConditionedGeometrySampler = Callable[[dict, dict, int], tuple[dict | None, dict]]


def _percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * float(q)))
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _numeric_summary(values: Sequence[int]) -> dict:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": float(mean(int(value) for value in values)),
        "p95": _percentile(values, 0.95),
        "max": int(max(values)),
    }


def attach_generated_geometry(
    topology_target: dict,
    geometry_sampler: GeometrySampler,
) -> tuple[dict, dict]:
    return _attach_generated_geometry_impl(
        topology_target,
        lambda topology, node, node_index: geometry_sampler(node, node_index),
    )


def attach_conditioned_generated_geometry(
    topology_target: dict,
    geometry_sampler: ConditionedGeometrySampler,
) -> tuple[dict, dict]:
    return _attach_generated_geometry_impl(topology_target, geometry_sampler)


def _attach_generated_geometry_impl(
    topology_target: dict,
    geometry_sampler: ConditionedGeometrySampler,
) -> tuple[dict, dict]:
    graph = topology_target.get("parse_graph", {}) or {}
    nodes: List[dict] = []
    attach_modes: Counter[str] = Counter()
    missing_nodes: List[str] = []
    geometry_rows: List[dict] = []

    for node_index, node in enumerate(graph.get("nodes", []) or []):
        output_node = copy.deepcopy(node)
        geometry_ref = output_node.pop("geometry_ref", None)
        if geometry_ref:
            geometry_target, geometry_diag = geometry_sampler(topology_target, output_node, int(node_index))
            geometry_rows.append(
                {
                    "node_index": int(node_index),
                    "node_id": str(output_node.get("id", "")),
                    "role": str(output_node.get("role", "")),
                    "label": int(output_node.get("label", 0)),
                    "geometry_model": str(output_node.get("geometry_model", "none")),
                    **dict(geometry_diag),
                }
            )
            if geometry_target is None:
                attach_modes["missing"] += 1
                missing_nodes.append(str(output_node.get("id", "")))
            else:
                attach_modes["generated"] += 1
                output_node["geometry_model"] = copy.deepcopy(
                    geometry_target.get("geometry_model", output_node.get("geometry_model"))
                )
                if "frame" in geometry_target:
                    output_node["frame"] = copy.deepcopy(geometry_target["frame"])
                if "geometry" in geometry_target:
                    output_node["geometry"] = copy.deepcopy(geometry_target["geometry"])
                if "atoms" in geometry_target:
                    output_node["atoms"] = copy.deepcopy(geometry_target["atoms"])
                output_node["generated_geometry_source_node_id"] = str(geometry_target.get("source_node_id", ""))
                output_node["generated_geometry_attach_mode"] = "generated"
        nodes.append(output_node)

    geometry_lengths = [int(row["length"]) for row in geometry_rows if row.get("length") is not None]
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": copy.deepcopy(topology_target.get("size", [0, 0])),
        "parse_graph": {
            "nodes": nodes,
            "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
            "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
        },
        "metadata": {
            "generated_geometry": True,
            "source_topology_metadata": copy.deepcopy(topology_target.get("metadata", {})),
            "geometry_request_count": int(len(geometry_rows)),
            "geometry_valid_count": int(sum(1 for row in geometry_rows if bool(row.get("valid", False)))),
            "attached_geometry_count": int(attach_modes.get("generated", 0)),
            "missing_geometry_count": int(len(missing_nodes)),
            "attach_modes": dict(attach_modes),
            "geometry_lengths": _numeric_summary(geometry_lengths),
        },
    }
    diagnostics = {
        "node_count": int(len(nodes)),
        "relation_count": int(len(target["parse_graph"]["relations"])),
        "geometry_request_count": int(len(geometry_rows)),
        "geometry_valid_count": int(target["metadata"]["geometry_valid_count"]),
        "attached_geometry_count": int(target["metadata"]["attached_geometry_count"]),
        "missing_geometry_count": int(len(missing_nodes)),
        "attach_modes": dict(attach_modes),
        "geometry_lengths": _numeric_summary(geometry_lengths),
        "geometry_rows": geometry_rows,
    }
    return target, diagnostics


def build_generated_geometry_targets_from_sample_rows(
    rows: Sequence[dict],
    geometry_sampler: GeometrySampler,
    *,
    include_invalid: bool = False,
) -> tuple[List[dict], dict]:
    targets: List[dict] = []
    diagnostics_rows: List[dict] = []
    skipped_invalid = 0
    for fallback_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not include_invalid:
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target["metadata"].update(
            {
                "sample_index": sample_index,
                "semantic_valid": bool(validation["semantic_valid"]),
                "checkpoint": row.get("checkpoint"),
            }
        )
        target, diagnostics = attach_generated_geometry(topology_target, geometry_sampler)
        target["metadata"]["sample_index"] = sample_index
        targets.append(target)
        diagnostics_rows.append({"sample_index": sample_index, **diagnostics})

    node_counts = [int(row["node_count"]) for row in diagnostics_rows]
    relation_counts = [int(row["relation_count"]) for row in diagnostics_rows]
    geometry_lengths: List[int] = []
    attach_modes = Counter()
    for row in diagnostics_rows:
        attach_modes.update(row.get("attach_modes", {}))
        for geometry_row in row.get("geometry_rows", []) or []:
            if geometry_row.get("length") is not None:
                geometry_lengths.append(int(geometry_row["length"]))

    summary = {
        "format": "maskgen_generated_geometry_summary_v1",
        "input_count": int(len(rows)),
        "output_count": int(len(targets)),
        "skipped_invalid_count": int(skipped_invalid),
        "geometry_request_count": int(sum(int(row["geometry_request_count"]) for row in diagnostics_rows)),
        "geometry_valid_count": int(sum(int(row["geometry_valid_count"]) for row in diagnostics_rows)),
        "attached_geometry_count": int(sum(int(row["attached_geometry_count"]) for row in diagnostics_rows)),
        "missing_geometry_count": int(sum(int(row["missing_geometry_count"]) for row in diagnostics_rows)),
        "node_count_mean": float(mean(node_counts)) if node_counts else None,
        "relation_count_mean": float(mean(relation_counts)) if relation_counts else None,
        "geometry_lengths": _numeric_summary(geometry_lengths),
        "attach_modes": dict(attach_modes),
        "rows": diagnostics_rows,
    }
    return targets, summary


def build_conditioned_generated_geometry_targets_from_sample_rows(
    rows: Sequence[dict],
    geometry_sampler: ConditionedGeometrySampler,
    *,
    include_invalid: bool = False,
) -> tuple[List[dict], dict]:
    targets: List[dict] = []
    diagnostics_rows: List[dict] = []
    skipped_invalid = 0
    for fallback_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not include_invalid:
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target["metadata"].update(
            {
                "sample_index": sample_index,
                "semantic_valid": bool(validation["semantic_valid"]),
                "checkpoint": row.get("checkpoint"),
            }
        )
        target, diagnostics = attach_conditioned_generated_geometry(topology_target, geometry_sampler)
        target["metadata"]["sample_index"] = sample_index
        targets.append(target)
        diagnostics_rows.append({"sample_index": sample_index, **diagnostics})

    node_counts = [int(row["node_count"]) for row in diagnostics_rows]
    relation_counts = [int(row["relation_count"]) for row in diagnostics_rows]
    geometry_lengths: List[int] = []
    attach_modes = Counter()
    for row in diagnostics_rows:
        attach_modes.update(row.get("attach_modes", {}))
        for geometry_row in row.get("geometry_rows", []) or []:
            if geometry_row.get("length") is not None:
                geometry_lengths.append(int(geometry_row["length"]))

    summary = {
        "format": "maskgen_conditioned_generated_geometry_summary_v1",
        "input_count": int(len(rows)),
        "output_count": int(len(targets)),
        "skipped_invalid_count": int(skipped_invalid),
        "geometry_request_count": int(sum(int(row["geometry_request_count"]) for row in diagnostics_rows)),
        "geometry_valid_count": int(sum(int(row["geometry_valid_count"]) for row in diagnostics_rows)),
        "attached_geometry_count": int(sum(int(row["attached_geometry_count"]) for row in diagnostics_rows)),
        "missing_geometry_count": int(sum(int(row["missing_geometry_count"]) for row in diagnostics_rows)),
        "node_count_mean": float(mean(node_counts)) if node_counts else None,
        "relation_count_mean": float(mean(relation_counts)) if relation_counts else None,
        "geometry_lengths": _numeric_summary(geometry_lengths),
        "attach_modes": dict(attach_modes),
        "rows": diagnostics_rows,
    }
    return targets, summary
