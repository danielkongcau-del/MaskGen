from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full-image shared-arc approximation benchmark JSONL.")
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


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def is_success(row: dict[str, Any]) -> bool:
    return not row.get("failure_reason") and bool(row.get("valid_partition"))


def face_count_bucket(value: Any) -> str:
    count = int(value or 0)
    if count < 16:
        return "<16"
    if count < 32:
        return "16-31"
    if count < 64:
        return "32-63"
    return ">=64"


def group_summary(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        compression = numeric([row.get("compression_ratio") for row in items])
        acceptance = numeric([row.get("owner_acceptance_rate") for row in items])
        output.append(
            {
                "group": key,
                "count": len(items),
                "success_rate": sum(1 for row in items if is_success(row)) / len(items) if items else 0.0,
                "mean_compression_ratio": mean_or_none(compression),
                "median_compression_ratio": median_or_none(compression),
                "mean_owner_acceptance_rate": mean_or_none(acceptance),
                "median_owner_acceptance_rate": median_or_none(acceptance),
            }
        )
    return output


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_file",
        "split",
        "stem",
        "valid_partition",
        "failure_reason",
        "face_count",
        "arc_count",
        "shared_arc_count",
        "original_arc_vertex_count",
        "final_arc_vertex_count",
        "arc_vertex_reduction",
        "compression_ratio",
        "union_iou",
        "overlap_area",
        "missing_adjacency_count",
        "extra_adjacency_count",
        "candidate_count",
        "accepted_owner_arc_count",
        "rejected_owner_arc_count",
        "owner_acceptance_rate",
        "runtime_ms",
    ]
    return {key: row.get(key) for key in keys}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if is_success(row)]
    compression = numeric([row.get("compression_ratio") for row in successful])
    reductions = numeric([row.get("arc_vertex_reduction") for row in successful])
    acceptance = numeric([row.get("owner_acceptance_rate") for row in successful])
    runtimes = numeric([row.get("runtime_ms") for row in rows])
    smoke_rows = [row for row in rows if isinstance(row.get("convex_smoke"), dict)]
    smoke_success = [
        row for row in smoke_rows if row.get("convex_smoke", {}).get("success")
    ]

    failures = [row for row in rows if not is_success(row)]
    worst_iou = sorted(rows, key=lambda row: row.get("union_iou") if row.get("union_iou") is not None else -1.0)
    worst_compression = sorted(
        rows,
        key=lambda row: row.get("compression_ratio") if row.get("compression_ratio") is not None else 10**9,
        reverse=True,
    )
    rejected = [row for row in rows if int(row.get("rejected_owner_arc_count") or 0) > 0]

    return {
        "total_images": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "mean_compression_ratio": mean_or_none(compression),
        "median_compression_ratio": median_or_none(compression),
        "mean_arc_vertex_reduction": mean_or_none(reductions),
        "median_arc_vertex_reduction": median_or_none(reductions),
        "mean_owner_acceptance_rate": mean_or_none(acceptance),
        "median_owner_acceptance_rate": median_or_none(acceptance),
        "mean_runtime_ms": mean_or_none(runtimes),
        "median_runtime_ms": median_or_none(runtimes),
        "convex_smoke_rows": len(smoke_rows),
        "convex_smoke_success_rate": len(smoke_success) / len(smoke_rows) if smoke_rows else None,
        "by_face_count_bucket": group_summary(rows, lambda row: face_count_bucket(row.get("face_count"))),
        "failures": [compact_row(row) for row in failures[:50]],
        "rows_with_rejected_owner_arcs": [compact_row(row) for row in rejected[:50]],
        "worst_20_by_union_iou": [compact_row(row) for row in worst_iou[:20]],
        "worst_20_by_compression_ratio": [compact_row(row) for row in worst_compression[:20]],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Global Approx Partition Benchmark Summary",
        "",
        f"- total images: {summary['total_images']}",
        f"- success count: {summary['success_count']}",
        f"- success rate: {summary['success_rate']:.3f}",
        f"- mean compression ratio: {summary['mean_compression_ratio']}",
        f"- median compression ratio: {summary['median_compression_ratio']}",
        f"- mean arc vertex reduction: {summary['mean_arc_vertex_reduction']}",
        f"- median arc vertex reduction: {summary['median_arc_vertex_reduction']}",
        f"- mean owner acceptance rate: {summary['mean_owner_acceptance_rate']}",
        f"- median owner acceptance rate: {summary['median_owner_acceptance_rate']}",
        f"- mean runtime ms: {summary['mean_runtime_ms']}",
        f"- median runtime ms: {summary['median_runtime_ms']}",
        f"- convex smoke rows: {summary['convex_smoke_rows']}",
        f"- convex smoke success rate: {summary['convex_smoke_success_rate']}",
        "",
        "## By Face Count Bucket",
        "",
        markdown_table(
            summary["by_face_count_bucket"],
            [
                "group",
                "count",
                "success_rate",
                "mean_compression_ratio",
                "median_compression_ratio",
                "mean_owner_acceptance_rate",
                "median_owner_acceptance_rate",
            ],
        ),
        "",
        "## Failures",
        "",
        "```json",
        json.dumps(summary["failures"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Rows With Rejected Owner Arcs",
        "",
        "```json",
        json.dumps(summary["rows_with_rejected_owner_arcs"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Worst 20 By Union IoU",
        "",
        "```json",
        json.dumps(summary["worst_20_by_union_iou"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Worst 20 By Compression Ratio",
        "",
        "```json",
        json.dumps(summary["worst_20_by_compression_ratio"], indent=2, ensure_ascii=False),
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
