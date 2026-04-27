from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_parse_graph_spatial_audit import audit_manual_parse_graph_targets_spatial  # noqa: E402
from partition_gen.manual_parse_graph_target_audit import iter_manual_parse_graph_target_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit spatial placement of renderable polygon nodes in full manual parse-graph targets."
    )
    parser.add_argument("--target-root", type=Path, required=True, help="A target JSON, graphs directory, or output root with manifest.jsonl.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--edge-margin", type=float, default=16.0)
    parser.add_argument("--min-bbox-area", type=float, default=1.0)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_summary_md(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(payload["renderable_polygon_node_count"])
    visible = int(payload["visible_polygon_node_count"])
    corner = int(payload["origin_corner_count"])
    edge = int(payload["origin_near_edge_count"])
    bbox_corner = int(payload["bbox_center_corner_count"])
    lines = [
        "# Manual Parse Graph Spatial Audit",
        "",
        f"- loaded: {payload['loaded_count']} / {payload['input_path_count']}",
        f"- renderable polygon nodes: {total}",
        f"- visible polygon nodes: {visible} ({_ratio(visible, total):.4f})",
        f"- invisible/tiny polygon nodes: {payload['invisible_polygon_node_count']}",
        f"- origin near edge: {edge} ({_ratio(edge, total):.4f})",
        f"- origin in corner: {corner} ({_ratio(corner, total):.4f})",
        f"- bbox center in corner: {bbox_corner} ({_ratio(bbox_corner, total):.4f})",
        "",
        "| metric | count | mean | min | median | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("origin_x_stats", "origin_y_stats", "scale_stats", "bbox_width_stats", "bbox_height_stats", "bbox_area_stats"):
        stats = payload[key]
        lines.append(
            f"| {key} | {stats['count']} | {stats['mean']} | {stats['min']} | "
            f"{stats['median']} | {stats['p95']} | {stats['max']} |"
        )
    lines.extend(
        [
            "",
            f"- origin quadrants: `{json.dumps(payload['origin_quadrant_histogram'], ensure_ascii=False)}`",
            f"- origin corners: `{json.dumps(payload['origin_corner_histogram'], ensure_ascii=False)}`",
            f"- bbox center corners: `{json.dumps(payload['bbox_center_corner_histogram'], ensure_ascii=False)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]
    payload = audit_manual_parse_graph_targets_spatial(
        paths,
        edge_margin=float(args.edge_margin),
        min_bbox_area=float(args.min_bbox_area),
    )
    payload["target_root"] = str(args.target_root.as_posix())
    dump_json(args.output_json, payload)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"spatial-audited targets={payload['loaded_count']} "
        f"renderable_polygons={payload['renderable_polygon_node_count']} "
        f"visible={payload['visible_polygon_node_count']} "
        f"origin_corners={payload['origin_corner_count']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
