from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize convex splitter benchmark JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "json"])
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_success(row: dict[str, Any]) -> bool:
    return not row.get("failure_reason") and bool(row.get("valid_partition"))


def is_fallback(row: dict[str, Any]) -> bool:
    backend = str(row.get("backend") or "")
    return backend.startswith("fallback")


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def approx_vertex_bucket(value: Any) -> str:
    count = int(value or 0)
    if count < 16:
        return "<16"
    if count < 32:
        return "16-31"
    if count < 64:
        return "32-63"
    if count < 128:
        return "64-127"
    return ">=128"


def group_summary(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        reductions = numeric([row.get("piece_reduction") for row in items])
        output.append(
            {
                "group": key,
                "count": len(items),
                "success_rate": sum(1 for row in items if is_success(row)) / len(items) if items else 0.0,
                "fallback_rate": sum(1 for row in items if is_fallback(row)) / len(items) if items else 0.0,
                "mean_piece_reduction": mean_or_none(reductions),
                "median_piece_reduction": median_or_none(reductions),
            }
        )
    return output


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_file",
        "split",
        "stem",
        "face_id",
        "label",
        "hole_count",
        "approx_vertex_count",
        "baseline_piece_count",
        "bridged_piece_count",
        "piece_reduction",
        "backend",
        "validation_iou",
        "failure_reason",
        "cut_slit_scale",
    ]
    return {key: row.get(key) for key in keys}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reductions = numeric([row.get("piece_reduction") for row in rows])
    sorted_by_iou = sorted(rows, key=lambda row: row.get("validation_iou") if row.get("validation_iou") is not None else -1.0)
    sorted_by_reduction = sorted(rows, key=lambda row: row.get("piece_reduction") if row.get("piece_reduction") is not None else -10**9)
    unique_faces = {
        (row.get("source_file"), row.get("face_id"))
        for row in rows
        if row.get("source_file") is not None and row.get("face_id") is not None
    }
    regressions = [
        row
        for row in rows
        if row.get("baseline_piece_count") is not None
        and row.get("bridged_piece_count") is not None
        and int(row["bridged_piece_count"]) > int(row["baseline_piece_count"])
    ]
    non_improvements = [
        row
        for row in rows
        if row.get("baseline_piece_count") is not None
        and row.get("bridged_piece_count") is not None
        and int(row["bridged_piece_count"]) >= int(row["baseline_piece_count"])
    ]
    return {
        "total_results": len(rows),
        "total_faces": len(unique_faces),
        "success_rate": sum(1 for row in rows if is_success(row)) / len(rows) if rows else 0.0,
        "fallback_rate": sum(1 for row in rows if is_fallback(row)) / len(rows) if rows else 0.0,
        "mean_piece_reduction": mean_or_none(reductions),
        "median_piece_reduction": median_or_none(reductions),
        "by_hole_count": group_summary(rows, lambda row: row.get("hole_count")),
        "by_label": group_summary(rows, lambda row: row.get("label")),
        "by_approx_vertex_count_bucket": group_summary(rows, lambda row: approx_vertex_bucket(row.get("approx_vertex_count"))),
        "worst_20_by_validation_iou": [compact_row(row) for row in sorted_by_iou[:20]],
        "worst_20_by_piece_reduction": [compact_row(row) for row in sorted_by_reduction[:20]],
        "samples_where_bridged_piece_count_gt_baseline": [compact_row(row) for row in regressions],
        "samples_where_bridged_piece_count_ge_baseline": [compact_row(row) for row in non_improvements],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Convex Splitter Benchmark Summary",
        "",
        f"- total results: {summary['total_results']}",
        f"- total faces: {summary['total_faces']}",
        f"- success rate: {summary['success_rate']:.3f}",
        f"- fallback rate: {summary['fallback_rate']:.3f}",
        f"- mean piece reduction: {summary['mean_piece_reduction']}",
        f"- median piece reduction: {summary['median_piece_reduction']}",
        "",
        "## By Hole Count",
        "",
        markdown_table(summary["by_hole_count"], ["group", "count", "success_rate", "fallback_rate", "mean_piece_reduction", "median_piece_reduction"]),
        "",
        "## By Label",
        "",
        markdown_table(summary["by_label"], ["group", "count", "success_rate", "fallback_rate", "mean_piece_reduction", "median_piece_reduction"]),
        "",
        "## By Approx Vertex Count Bucket",
        "",
        markdown_table(summary["by_approx_vertex_count_bucket"], ["group", "count", "success_rate", "fallback_rate", "mean_piece_reduction", "median_piece_reduction"]),
        "",
        "## Worst 20 By Validation IoU",
        "",
        "```json",
        json.dumps(summary["worst_20_by_validation_iou"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Worst 20 By Piece Reduction",
        "",
        "```json",
        json.dumps(summary["worst_20_by_piece_reduction"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Bridged Piece Count > Baseline",
        "",
        "```json",
        json.dumps(summary["samples_where_bridged_piece_count_gt_baseline"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Bridged Piece Count >= Baseline",
        "",
        "```json",
        json.dumps(summary["samples_where_bridged_piece_count_ge_baseline"], indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    summary = build_summary(rows)
    text = json.dumps(summary, indent=2, ensure_ascii=False) if args.format == "json" else render_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
