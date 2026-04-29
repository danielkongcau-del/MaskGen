from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_parse_graph_spatial_audit import audit_manual_parse_graph_target_spatial  # noqa: E402
from partition_gen.manual_parse_graph_target_audit import iter_manual_parse_graph_target_paths, load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect invisible/tiny manual parse-graph polygon nodes and summarize failure patterns."
    )
    parser.add_argument("--target-root", type=Path, required=True, help="A target JSON, graphs dir, or output root with manifest.jsonl.")
    parser.add_argument("--spatial-audit-json", type=Path, default=None, help="Optional existing spatial_audit.json.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Per-failure node rows.")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--edge-margin", type=float, default=None)
    parser.add_argument("--min-bbox-area", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--include-edge-warnings",
        action="store_true",
        help="Also include visible nodes whose origin or bbox center is near an edge/corner.",
    )
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _numeric_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "p95": None, "max": None}
    floats = [float(value) for value in values]
    sorted_values = sorted(floats)
    p95_index = min(len(sorted_values) - 1, max(0, int(round(0.95 * (len(sorted_values) - 1)))))
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "p95": float(sorted_values[p95_index]),
        "max": float(max(floats)),
    }


def _top_items(counter: Counter, limit: int) -> list[dict]:
    return [{"key": str(key), "count": int(value)} for key, value in counter.most_common(int(limit))]


def _frame_clamp_mode(diagnostics: dict | None) -> str:
    diagnostics = diagnostics or {}
    if diagnostics.get("geometry_frame_clamp_strong"):
        return "strong_geometry_clamp"
    if diagnostics.get("geometry_scale_clamped"):
        return "geometry_clamp"
    if diagnostics.get("tokenizer_scale_clamped"):
        return "tokenizer_clamp"
    if diagnostics.get("scale_clamped"):
        return "scale_clamp"
    return "none"


def _node_reasons(spatial_row: dict, *, include_edge_warnings: bool) -> list[str]:
    reasons: list[str] = []
    if not bool(spatial_row.get("has_world_bbox")):
        reasons.append("missing_world_bbox")
    if not bool(spatial_row.get("bbox_intersects_canvas")):
        reasons.append("off_canvas_bbox")
    if bool(spatial_row.get("bbox_tiny")):
        reasons.append("tiny_bbox")
    if include_edge_warnings:
        if not bool(spatial_row.get("origin_inside_canvas")):
            reasons.append("origin_outside_canvas")
        if bool(spatial_row.get("origin_near_edge")):
            reasons.append("origin_near_edge")
        if spatial_row.get("origin_corner") is not None:
            reasons.append("origin_corner")
        if bool(spatial_row.get("bbox_center_near_edge")):
            reasons.append("bbox_center_near_edge")
        if spatial_row.get("bbox_center_corner") is not None:
            reasons.append("bbox_center_corner")
    return reasons


def _nodes_by_id(target: dict) -> dict[str, tuple[int, dict]]:
    graph = target.get("parse_graph", {}) or {}
    return {
        str(node.get("id", "")): (int(index), node)
        for index, node in enumerate(graph.get("nodes", []) or [])
    }


def _mapping_mode_for_node(target: dict, node_index: int) -> str:
    metadata = target.get("metadata", {}) or {}
    mapping_modes = ((metadata.get("mapping_diagnostics", {}) or {}).get("node_mapping_modes", {}) or {})
    return str(mapping_modes.get(str(node_index), mapping_modes.get(int(node_index), "unknown")))


def _metadata_value(target: dict, key: str, default=None):
    return (target.get("metadata", {}) or {}).get(key, default)


def _local_bbox_width(local_bbox: dict | None) -> float | None:
    if not local_bbox:
        return None
    return abs(float(local_bbox.get("width", 0.0)))


def _local_bbox_height(local_bbox: dict | None) -> float | None:
    if not local_bbox:
        return None
    return abs(float(local_bbox.get("height", 0.0)))


def _inspect_target(
    path: Path,
    *,
    spatial_row: dict | None,
    edge_margin: float,
    min_bbox_area: float,
    include_edge_warnings: bool,
) -> tuple[list[dict], dict]:
    target = load_json(path)
    spatial = spatial_row or audit_manual_parse_graph_target_spatial(
        target,
        source=str(path.as_posix()),
        edge_margin=float(edge_margin),
        min_bbox_area=float(min_bbox_area),
    )
    node_lookup = _nodes_by_id(target)
    failures: list[dict] = []
    sample_index = _metadata_value(target, "sample_index")
    source = str(path.as_posix())

    for spatial_node in spatial.get("nodes", []) or []:
        reasons = _node_reasons(spatial_node, include_edge_warnings=include_edge_warnings)
        if not reasons:
            continue
        node_index, node = node_lookup.get(str(spatial_node.get("node_id", "")), (-1, {}))
        frame_clamp = node.get("frame_clamp", {}) or {}
        generated_local_bbox = node.get("generated_local_bbox", {}) or None
        row = {
            "source": source,
            "sample_index": sample_index,
            "node_index": int(node_index),
            "node_id": str(spatial_node.get("node_id", "")),
            "role": str(spatial_node.get("role", node.get("role", ""))),
            "label": int(node.get("label", spatial_node.get("label", 0))),
            "geometry_model": str(node.get("geometry_model", "none")),
            "mapping_mode": _mapping_mode_for_node(target, int(node_index)) if node_index >= 0 else "unknown",
            "retrieval_score": _metadata_value(target, "retrieval_score"),
            "retrieved_stem": _metadata_value(target, "retrieved_stem"),
            "reasons": reasons,
            "primary_reason": reasons[0],
            "polygon_count": int(spatial_node.get("polygon_count", 0)),
            "origin": spatial_node.get("origin"),
            "scale": float(spatial_node.get("scale", 0.0)),
            "bbox": spatial_node.get("bbox"),
            "bbox_width": float(spatial_node.get("bbox_width", 0.0)),
            "bbox_height": float(spatial_node.get("bbox_height", 0.0)),
            "bbox_area": float(spatial_node.get("bbox_area", 0.0)),
            "bbox_center": spatial_node.get("bbox_center"),
            "bbox_intersects_canvas": bool(spatial_node.get("bbox_intersects_canvas")),
            "bbox_inside_canvas": bool(spatial_node.get("bbox_inside_canvas")),
            "bbox_tiny": bool(spatial_node.get("bbox_tiny")),
            "origin_inside_canvas": bool(spatial_node.get("origin_inside_canvas")),
            "origin_near_edge": bool(spatial_node.get("origin_near_edge")),
            "origin_corner": spatial_node.get("origin_corner"),
            "bbox_center_near_edge": bool(spatial_node.get("bbox_center_near_edge")),
            "bbox_center_corner": spatial_node.get("bbox_center_corner"),
            "layout_frame_source": node.get("layout_frame_source"),
            "layout_shape_attach_mode": node.get("layout_shape_attach_mode"),
            "frame_clamp_mode": _frame_clamp_mode(frame_clamp),
            "frame_clamp": frame_clamp,
            "generated_local_bbox": generated_local_bbox,
            "generated_local_bbox_width": _local_bbox_width(generated_local_bbox),
            "generated_local_bbox_height": _local_bbox_height(generated_local_bbox),
            "retrieved_frame": node.get("retrieved_frame"),
            "refined_frame": node.get("refined_frame"),
            "final_frame": node.get("frame"),
        }
        failures.append(row)
    return failures, spatial


def _summarize(rows: Sequence[dict], spatial_rows: Sequence[dict], *, target_root: Path, edge_margin: float, min_bbox_area: float) -> dict:
    reason_histogram = Counter()
    role_histogram = Counter()
    role_label_histogram = Counter()
    geometry_model_histogram = Counter()
    mapping_mode_histogram = Counter()
    frame_clamp_mode_histogram = Counter()
    source_histogram = Counter()
    retrieved_stem_histogram = Counter()
    for row in rows:
        reason_histogram.update(row.get("reasons", []) or [])
        role_histogram[str(row.get("role", ""))] += 1
        role_label_histogram[f"{row.get('role', '')}:{row.get('label', 0)}"] += 1
        geometry_model_histogram[str(row.get("geometry_model", "none"))] += 1
        mapping_mode_histogram[str(row.get("mapping_mode", "unknown"))] += 1
        frame_clamp_mode_histogram[str(row.get("frame_clamp_mode", "none"))] += 1
        source_histogram[str(row.get("source", ""))] += 1
        if row.get("retrieved_stem") is not None:
            retrieved_stem_histogram[str(row.get("retrieved_stem"))] += 1

    renderable_count = int(sum(int(row.get("renderable_polygon_node_count", 0)) for row in spatial_rows))
    visible_count = int(sum(int(row.get("visible_polygon_node_count", 0)) for row in spatial_rows))
    return {
        "format": "maskgen_manual_parse_graph_spatial_failure_inspection_v1",
        "target_root": str(target_root.as_posix()),
        "edge_margin": float(edge_margin),
        "min_bbox_area": float(min_bbox_area),
        "loaded_count": int(len(spatial_rows)),
        "renderable_polygon_node_count": renderable_count,
        "visible_polygon_node_count": visible_count,
        "failure_count": int(len(rows)),
        "failure_ratio": float(len(rows) / renderable_count) if renderable_count else 0.0,
        "reason_histogram": dict(reason_histogram.most_common()),
        "role_histogram": dict(role_histogram.most_common()),
        "role_label_histogram": dict(role_label_histogram.most_common()),
        "geometry_model_histogram": dict(geometry_model_histogram.most_common()),
        "mapping_mode_histogram": dict(mapping_mode_histogram.most_common()),
        "frame_clamp_mode_histogram": dict(frame_clamp_mode_histogram.most_common()),
        "top_sources": _top_items(source_histogram, 20),
        "top_retrieved_stems": _top_items(retrieved_stem_histogram, 20),
        "scale_stats": _numeric_stats([float(row["scale"]) for row in rows]),
        "bbox_width_stats": _numeric_stats([float(row["bbox_width"]) for row in rows]),
        "bbox_height_stats": _numeric_stats([float(row["bbox_height"]) for row in rows]),
        "bbox_area_stats": _numeric_stats([float(row["bbox_area"]) for row in rows]),
        "generated_local_bbox_width_stats": _numeric_stats(
            [float(row["generated_local_bbox_width"]) for row in rows if row.get("generated_local_bbox_width") is not None]
        ),
        "generated_local_bbox_height_stats": _numeric_stats(
            [float(row["generated_local_bbox_height"]) for row in rows if row.get("generated_local_bbox_height") is not None]
        ),
        "retrieval_score_stats": _numeric_stats(
            [float(row["retrieval_score"]) for row in rows if row.get("retrieval_score") is not None]
        ),
    }


def _markdown_table_from_hist(title: str, histogram: dict, *, top_k: int) -> list[str]:
    lines = [f"## {title}", "", "| key | count |", "| --- | ---: |"]
    for key, value in list(histogram.items())[: int(top_k)]:
        lines.append(f"| {key} | {value} |")
    return lines + [""]


def write_summary_md(path: Path, summary: dict, rows: Sequence[dict], *, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Parse Graph Spatial Failure Inspection",
        "",
        f"- target root: `{summary['target_root']}`",
        f"- loaded: {summary['loaded_count']}",
        f"- renderable polygon nodes: {summary['renderable_polygon_node_count']}",
        f"- visible polygon nodes: {summary['visible_polygon_node_count']}",
        f"- inspected failure nodes: {summary['failure_count']} ({summary['failure_ratio']:.4f})",
        f"- edge margin: {summary['edge_margin']}",
        f"- min bbox area: {summary['min_bbox_area']}",
        "",
    ]
    for title, key in (
        ("Reasons", "reason_histogram"),
        ("Roles", "role_histogram"),
        ("Role Labels", "role_label_histogram"),
        ("Mapping Modes", "mapping_mode_histogram"),
        ("Frame Clamp Modes", "frame_clamp_mode_histogram"),
    ):
        lines.extend(_markdown_table_from_hist(title, summary[key], top_k=top_k))

    lines.extend(
        [
            "## Numeric Stats",
            "",
            "| metric | count | mean | min | median | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "scale_stats",
        "bbox_width_stats",
        "bbox_height_stats",
        "bbox_area_stats",
        "generated_local_bbox_width_stats",
        "generated_local_bbox_height_stats",
        "retrieval_score_stats",
    ):
        stats = summary[key]
        lines.append(
            f"| {key} | {stats['count']} | {stats['mean']} | {stats['min']} | "
            f"{stats['median']} | {stats['p95']} | {stats['max']} |"
        )

    lines.extend(
        [
            "",
            "## Example Failures",
            "",
            "| sample | node | role | label | reasons | scale | bbox_area | local_bbox | clamp |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows[: int(top_k)]:
        local_bbox = row.get("generated_local_bbox") or {}
        local_bbox_text = (
            f"{local_bbox.get('width')}x{local_bbox.get('height')}" if local_bbox else ""
        )
        lines.append(
            f"| {row.get('sample_index')} | {row.get('node_id')} | {row.get('role')} | "
            f"{row.get('label')} | {','.join(row.get('reasons', []))} | "
            f"{row.get('scale')} | {row.get('bbox_area')} | {local_bbox_text} | "
            f"{row.get('frame_clamp_mode')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_spatial_audit(path: Path | None) -> dict | None:
    if path is None:
        return None
    return load_json(path)


def main() -> None:
    args = parse_args()
    spatial_audit = _load_spatial_audit(args.spatial_audit_json)
    edge_margin = float(
        args.edge_margin
        if args.edge_margin is not None
        else ((spatial_audit or {}).get("edge_margin", 16.0))
    )
    min_bbox_area = float(
        args.min_bbox_area
        if args.min_bbox_area is not None
        else ((spatial_audit or {}).get("min_bbox_area", 1.0))
    )
    spatial_by_source = {
        str(row.get("source")): row
        for row in ((spatial_audit or {}).get("rows", []) or [])
        if row.get("source") is not None
    }

    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]

    failure_rows: list[dict] = []
    spatial_rows: list[dict] = []
    load_errors: list[dict] = []
    for path in paths:
        try:
            source = str(path.as_posix())
            rows, spatial_row = _inspect_target(
                path,
                spatial_row=spatial_by_source.get(source),
                edge_margin=edge_margin,
                min_bbox_area=min_bbox_area,
                include_edge_warnings=bool(args.include_edge_warnings),
            )
            failure_rows.extend(rows)
            spatial_rows.append(spatial_row)
        except Exception as exc:
            load_errors.append({"source": str(path.as_posix()), "error": f"{type(exc).__name__}:{exc}"})

    failure_rows.sort(
        key=lambda row: (
            str(row.get("primary_reason", "")),
            float(row.get("bbox_area", 0.0)),
            str(row.get("role", "")),
            int(row.get("label", 0)),
        )
    )
    summary = _summarize(
        failure_rows,
        spatial_rows,
        target_root=args.target_root,
        edge_margin=edge_margin,
        min_bbox_area=min_bbox_area,
    )
    summary["input_path_count"] = int(len(paths))
    summary["load_error_count"] = int(len(load_errors))
    summary["load_errors"] = load_errors
    summary["spatial_audit_json"] = None if args.spatial_audit_json is None else str(args.spatial_audit_json.as_posix())

    write_jsonl(args.output_jsonl, failure_rows)
    if args.summary_json is not None:
        dump_json(args.summary_json, summary)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, summary, failure_rows, top_k=int(args.top_k))
    print(
        f"inspected spatial failures targets={summary['loaded_count']} "
        f"failures={summary['failure_count']} output={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
