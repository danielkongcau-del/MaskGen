from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual-rule explainer benchmark JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(rows: List[dict], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows if key in row]
    return float(mean(values)) if values else 0.0


def _ratio(rows: List[dict], key: str, value=True) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if row.get(key) == value) / len(rows))


def _histogram(rows: List[dict], key: str) -> Dict[str, int]:
    output: Dict[str, int] = {}
    for row in rows:
        for name, count in (row.get(key) or {}).items():
            output[str(name)] = int(output.get(str(name), 0) + int(count))
    return dict(sorted(output.items()))


def _top(rows: List[dict], key: str, *, reverse: bool = True, n: int = 20) -> List[dict]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0) or 0.0), reverse=reverse)[:n]


def _table(rows: List[dict], columns: List[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _aggregate_uncovered_pairs(rows: List[dict]) -> List[dict]:
    buckets: Dict[str, dict] = {}
    for row in rows:
        for item in row.get("uncovered_contact_pairs", []) or []:
            key = str(item.get("pair"))
            bucket = buckets.setdefault(
                key,
                {
                    "pair": key,
                    "labels": item.get("labels", []),
                    "label_names": item.get("label_names", []),
                    "sample_count": 0,
                    "contact_count": 0,
                    "shared_length": 0.0,
                    "arc_count": 0,
                },
            )
            bucket["sample_count"] = int(bucket["sample_count"]) + 1
            bucket["contact_count"] = int(bucket["contact_count"]) + int(item.get("contact_count", 0) or 0)
            bucket["shared_length"] = float(bucket["shared_length"]) + float(item.get("shared_length", 0.0) or 0.0)
            bucket["arc_count"] = int(bucket["arc_count"]) + int(item.get("arc_count", 0) or 0)
    return sorted(
        buckets.values(),
        key=lambda item: (float(item["shared_length"]), int(item["contact_count"]), int(item["sample_count"])),
        reverse=True,
    )


def summarize(rows: List[dict]) -> str:
    ok_rows = [row for row in rows if "error" not in row]
    uncovered_pairs = _aggregate_uncovered_pairs(ok_rows)
    lines = [
        "# Manual Rule Explainer Benchmark Summary",
        "",
        f"- samples: {len(rows)}",
        f"- successful rows: {len(ok_rows)}",
        f"- valid ratio: {_ratio(ok_rows, 'validation_is_valid'):.3f}",
        f"- all_faces_owned_exactly_once ratio: {_ratio(ok_rows, 'all_faces_owned_exactly_once'):.3f}",
        f"- relation refs valid ratio: {_ratio(ok_rows, 'relation_reference_valid'):.3f}",
        f"- mean residual_face_count: {_mean(ok_rows, 'residual_face_count'):.3f}",
        f"- mean residual_area_ratio: {_mean(ok_rows, 'residual_area_ratio'):.6f}",
        f"- mean duplicate_owned_face_count: {_mean(ok_rows, 'duplicate_owned_face_count'):.3f}",
        f"- mean unowned_face_count: {_mean(ok_rows, 'unowned_face_count'):.3f}",
        f"- mean node_count: {_mean(ok_rows, 'node_count'):.3f}",
        f"- mean relation_count: {_mean(ok_rows, 'relation_count'):.3f}",
        f"- mean runtime_ms: {_mean(ok_rows, 'runtime_ms'):.3f}",
        f"- mean uncovered_contact_pair_count: {_mean(ok_rows, 'uncovered_contact_pair_count'):.3f}",
        "",
        "## Operation Histogram",
        "",
        "```json",
        json.dumps(_histogram(ok_rows, "operation_histogram"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Role Histogram",
        "",
        "```json",
        json.dumps(_histogram(ok_rows, "role_histogram"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Uncovered Contact Pairs",
        "",
        _table(
            uncovered_pairs[:20],
            ["pair", "labels", "label_names", "sample_count", "contact_count", "shared_length", "arc_count"],
        ),
        "",
        "## Top Residual Area Ratio",
        "",
        _table(
            _top(ok_rows, "residual_area_ratio"),
            [
                "source",
                "validation_is_valid",
                "residual_area_ratio",
                "residual_face_count",
                "duplicate_owned_face_count",
                "unowned_face_count",
                "uncovered_contact_pair_count",
            ],
        ),
        "",
        "## Top Duplicate Owned Faces",
        "",
        _table(
            _top(ok_rows, "duplicate_owned_face_count"),
            [
                "source",
                "validation_is_valid",
                "duplicate_owned_face_count",
                "unowned_face_count",
                "residual_face_count",
                "operation_histogram",
            ],
        ),
        "",
        "## Top Unowned Faces",
        "",
        _table(
            _top(ok_rows, "unowned_face_count"),
            [
                "source",
                "validation_is_valid",
                "unowned_face_count",
                "duplicate_owned_face_count",
                "residual_face_count",
                "operation_histogram",
            ],
        ),
    ]
    if len(ok_rows) < len(rows):
        errors = [row for row in rows if "error" in row]
        lines.extend(
            [
                "",
                "## Errors",
                "",
                _table(errors[:20], ["source", "error"]),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summarize(rows), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
