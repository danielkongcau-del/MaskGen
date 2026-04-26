from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize operation explainer benchmark JSONL.")
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


def summarize(rows: List[dict]) -> str:
    ok_rows = [row for row in rows if "error" not in row]
    cost_profiles = sorted({str(row.get("cost_profile", "unknown")) for row in ok_rows})
    lines = [
        "# Operation Explainer Benchmark Summary",
        "",
        f"- samples: {len(rows)}",
        f"- successful rows: {len(ok_rows)}",
        f"- valid ratio: {_ratio(ok_rows, 'validation_is_valid'):.3f}",
        f"- OR-Tools optimal ratio: {_ratio(ok_rows, 'global_optimal'):.3f}",
        f"- greedy fallback ratio: {sum(1 for row in ok_rows if row.get('selection_method') == 'greedy_fallback') / len(ok_rows) if ok_rows else 0.0:.3f}",
        f"- mean residual_face_count: {_mean(ok_rows, 'residual_face_count'):.3f}",
        f"- mean residual_area_ratio: {_mean(ok_rows, 'residual_area_ratio'):.6f}",
        f"- mean total_compression_gain: {_mean(ok_rows, 'total_compression_gain'):.3f}",
        f"- mean false_cover_ratio_max: {_mean(ok_rows, 'false_cover_ratio_max'):.6f}",
        "",
        "## Operation Histogram",
        "",
        "```json",
        json.dumps(_histogram(ok_rows, "operation_histogram"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Cost Profile Groups",
    ]
    for profile in cost_profiles:
        profile_rows = [row for row in ok_rows if str(row.get("cost_profile", "unknown")) == profile]
        lines.extend(
            [
                "",
                f"### {profile}",
                "",
                f"- samples: {len(profile_rows)}",
                f"- valid ratio: {_ratio(profile_rows, 'validation_is_valid'):.3f}",
                f"- OR-Tools optimal ratio: {_ratio(profile_rows, 'global_optimal'):.3f}",
                f"- mean residual_area_ratio: {_mean(profile_rows, 'residual_area_ratio'):.6f}",
                f"- mean total_compression_gain: {_mean(profile_rows, 'total_compression_gain'):.3f}",
                f"- mean false_cover_ratio_max: {_mean(profile_rows, 'false_cover_ratio_max'):.6f}",
                "```json",
                json.dumps(_histogram(profile_rows, "operation_histogram"), indent=2, ensure_ascii=False),
                "```",
            ]
        )
    lines.extend(
        [
        "",
        "## Top False Cover",
        "",
        _table(_top(ok_rows, "false_cover_ratio_max"), ["source", "false_cover_ratio_max", "false_cover_area_total", "validation_is_valid"]),
        "",
        "## Top Residual Area Ratio",
        "",
        _table(_top(ok_rows, "residual_area_ratio"), ["source", "residual_area_ratio", "residual_face_count", "operation_histogram"]),
        "",
        "## Lowest Compression Gain",
        "",
        _table(_top(ok_rows, "total_compression_gain", reverse=False), ["source", "total_compression_gain", "residual_area_ratio", "operation_histogram"]),
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
