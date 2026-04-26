from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a weak explainer benchmark JSONL file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", type=str, default="md", choices=["md", "json"])
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def worst_rows(rows: list[dict], key: str, *, reverse: bool, top_k: int, predicate: Callable[[dict], bool] | None = None) -> list[dict]:
    predicate = predicate or (lambda _row: True)
    filtered = [row for row in rows if row.get(key) is not None and predicate(row)]
    return sorted(filtered, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[:top_k]


def compact_row(row: dict) -> dict:
    keys = [
        "stem",
        "source_file",
        "success",
        "render_valid",
        "face_count",
        "atom_count",
        "atom_per_face",
        "full_iou",
        "mask_pixel_accuracy",
        "overlap_area",
        "gap_area",
        "low_iou_face_count",
        "runtime_ms",
        "failure_reason",
    ]
    return {key: row.get(key) for key in keys if key in row}


def build_summary(rows: list[dict], *, top_k: int) -> dict:
    success_rows = [row for row in rows if row.get("success")]
    render_valid_rows = [row for row in success_rows if row.get("render_valid")]
    failures = [row for row in rows if not row.get("success")]
    invalid_renders = [row for row in success_rows if not row.get("render_valid")]
    low_iou = [row for row in success_rows if row.get("full_iou") is not None and float(row["full_iou"]) < 0.999]
    overlap = [row for row in success_rows if float(row.get("overlap_area") or 0.0) > 1e-6]
    gap = [row for row in success_rows if row.get("gap_area") is not None and float(row["gap_area"]) > 1e-6]
    return {
        "total_rows": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(failures),
        "render_valid_count": len(render_valid_rows),
        "invalid_render_count": len(invalid_renders),
        "success_rate": float(len(success_rows) / len(rows)) if rows else 0.0,
        "render_valid_rate": float(len(render_valid_rows) / len(success_rows)) if success_rows else 0.0,
        "metric_stats": {
            "runtime_ms": stats(finite_values(rows, "runtime_ms")),
            "face_count": stats(finite_values(success_rows, "face_count")),
            "atom_count": stats(finite_values(success_rows, "atom_count")),
            "atom_per_face": stats(finite_values(success_rows, "atom_per_face")),
            "relation_count": stats(finite_values(success_rows, "relation_count")),
            "code_length": stats(finite_values(success_rows, "code_length")),
            "full_iou": stats(finite_values(success_rows, "full_iou")),
            "mask_pixel_accuracy": stats(finite_values(success_rows, "mask_pixel_accuracy")),
            "overlap_area": stats(finite_values(success_rows, "overlap_area")),
            "gap_area": stats(finite_values(success_rows, "gap_area")),
            "low_iou_face_count": stats(finite_values(success_rows, "low_iou_face_count")),
        },
        "problem_counts": {
            "low_full_iou_lt_0_999": len(low_iou),
            "overlap_gt_1e_minus_6": len(overlap),
            "gap_gt_1e_minus_6": len(gap),
            "invalid_render": len(invalid_renders),
        },
        "failures": [compact_row(row) for row in failures[:top_k]],
        "worst_full_iou": [compact_row(row) for row in worst_rows(success_rows, "full_iou", reverse=False, top_k=top_k)],
        "worst_mask_pixel_accuracy": [compact_row(row) for row in worst_rows(success_rows, "mask_pixel_accuracy", reverse=False, top_k=top_k)],
        "largest_overlap": [compact_row(row) for row in worst_rows(success_rows, "overlap_area", reverse=True, top_k=top_k)],
        "largest_gap": [compact_row(row) for row in worst_rows(success_rows, "gap_area", reverse=True, top_k=top_k)],
        "most_low_iou_faces": [compact_row(row) for row in worst_rows(success_rows, "low_iou_face_count", reverse=True, top_k=top_k)],
        "highest_atom_per_face": [compact_row(row) for row in worst_rows(success_rows, "atom_per_face", reverse=True, top_k=top_k)],
    }


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines) + "\n"


def to_markdown(summary: dict) -> str:
    metric_lines = []
    for key, value in summary["metric_stats"].items():
        metric_lines.append(
            f"|{key}|{value['count']}|{value['mean']}|{value['median']}|{value['min']}|{value['max']}|"
        )
    columns = ["stem", "render_valid", "face_count", "atom_count", "atom_per_face", "full_iou", "overlap_area", "gap_area", "low_iou_face_count", "runtime_ms", "failure_reason"]
    return "\n".join(
        [
            "# Weak Explainer Benchmark Summary",
            "",
            f"- total rows: {summary['total_rows']}",
            f"- success: {summary['success_count']} / {summary['total_rows']} ({summary['success_rate']:.3f})",
            f"- render valid: {summary['render_valid_count']} / {summary['success_count']} ({summary['render_valid_rate']:.3f})",
            f"- failures: {summary['failure_count']}",
            f"- problem counts: `{json.dumps(summary['problem_counts'], ensure_ascii=False)}`",
            "",
            "## Metrics",
            "",
            "|metric|count|mean|median|min|max|",
            "|---|---:|---:|---:|---:|---:|",
            *metric_lines,
            "",
            "## Failures",
            "",
            markdown_table(summary["failures"], columns),
            "## Worst Full IoU",
            "",
            markdown_table(summary["worst_full_iou"], columns),
            "## Largest Overlap",
            "",
            markdown_table(summary["largest_overlap"], columns),
            "## Largest Gap",
            "",
            markdown_table(summary["largest_gap"], columns),
            "## Most Low-IoU Faces",
            "",
            markdown_table(summary["most_low_iou_faces"], columns),
            "## Highest Atom Per Face",
            "",
            markdown_table(summary["highest_atom_per_face"], columns),
        ]
    )


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    summary = build_summary(rows, top_k=int(args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        args.output.write_text(to_markdown(summary), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
