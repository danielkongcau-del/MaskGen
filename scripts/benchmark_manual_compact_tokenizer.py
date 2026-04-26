from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_target_token_stats import analyze_manual_target_token_stats  # noqa: E402
from partition_gen.parse_graph_compact_tokenizer import compact_tokenizer_diagnostics  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compact tokenizer v1 on manual-rule generator targets.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", "--output-jsonl", dest="output_jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_target_paths(root: Path, split: str | None = None) -> Iterable[Path]:
    graph_root = root / split / "graphs" if split else None
    if graph_root and graph_root.exists():
        yield from sorted(graph_root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    split_root = root / split if split else None
    if split_root and split_root.exists():
        yield from sorted(split_root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    if root.name == "graphs":
        yield from sorted(root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    yield from sorted(root.rglob("*.json"), key=lambda path: (str(path.parent), len(path.stem), path.stem))


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * percentile))
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
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


def benchmark_target(path: Path, target: dict, *, config: ParseGraphTokenizerConfig | None = None) -> Dict[str, object]:
    config = config or ParseGraphTokenizerConfig()
    stats = analyze_manual_target_token_stats(target, tokenizer_config=config)
    compact = compact_tokenizer_diagnostics(target, config=config)
    old_relation_tokens = int(stats["relation_token_count"])
    compact_relation_tokens = int(compact["relation_token_count_compact"])
    return {
        "path": str(path.as_posix()),
        "stem": path.stem,
        "old_total_tokens": int(compact["old_total_tokens"]),
        "compact_total_tokens": int(compact["compact_total_tokens"]),
        "token_reduction": int(compact["token_reduction"]),
        "token_reduction_ratio": float(compact["token_reduction_ratio"]),
        "node_count": int(compact["node_count"]),
        "relation_count": int(compact["relation_count"]),
        "contains_relation_count": int(compact["contains_relation_count"]),
        "skipped_contains_relation_count": int(compact["skipped_contains_relation_count"]),
        "inserted_in_count": int(compact["inserted_in_count"]),
        "divides_count": int(compact["divides_count"]),
        "adjacent_to_count": int(compact["adjacent_to_count"]),
        "polygon_token_count_estimate": int(stats["polygon_token_count"]),
        "node_token_count_estimate": int(stats["node_header_token_count"] + stats["geometry_token_count"]),
        "relation_token_count_estimate_old": int(old_relation_tokens),
        "relation_token_count_estimate_compact": int(compact_relation_tokens),
        "relation_token_reduction": int(old_relation_tokens - compact_relation_tokens),
        "relation_token_reduction_ratio": float((old_relation_tokens - compact_relation_tokens) / old_relation_tokens)
        if old_relation_tokens
        else 0.0,
    }


def run_benchmark(paths: Sequence[Path]) -> list[Dict[str, object]]:
    config = ParseGraphTokenizerConfig()
    rows: list[Dict[str, object]] = []
    for path in paths:
        target = load_json(path)
        if target.get("format") != "maskgen_generator_target_v1" or target.get("target_type") != "parse_graph":
            continue
        rows.append(benchmark_target(path, target, config=config))
    return rows


def summarize_rows(rows: Sequence[Dict[str, object]], *, top_k: int = 20) -> str:
    old_lengths = [int(row["old_total_tokens"]) for row in rows]
    compact_lengths = [int(row["compact_total_tokens"]) for row in rows]
    reductions = [float(row["token_reduction_ratio"]) for row in rows]
    relation_reductions = [float(row["relation_token_reduction_ratio"]) for row in rows]
    sample_37 = [row for row in rows if str(row.get("stem")) == "37"]
    lines = [
        "# Manual Compact Tokenizer Benchmark",
        "",
        f"- samples: {len(rows)}",
        f"- old_mean_tokens: {mean(old_lengths):.2f}" if old_lengths else "- old_mean_tokens: n/a",
        f"- old_p50_tokens: {median(old_lengths):.2f}" if old_lengths else "- old_p50_tokens: n/a",
        f"- old_p90_tokens: {_percentile(old_lengths, 0.90)}",
        f"- old_p95_tokens: {_percentile(old_lengths, 0.95)}",
        f"- old_max_tokens: {max(old_lengths) if old_lengths else 'n/a'}",
        f"- compact_mean_tokens: {mean(compact_lengths):.2f}" if compact_lengths else "- compact_mean_tokens: n/a",
        f"- compact_p50_tokens: {median(compact_lengths):.2f}" if compact_lengths else "- compact_p50_tokens: n/a",
        f"- compact_p90_tokens: {_percentile(compact_lengths, 0.90)}",
        f"- compact_p95_tokens: {_percentile(compact_lengths, 0.95)}",
        f"- compact_max_tokens: {max(compact_lengths) if compact_lengths else 'n/a'}",
        f"- mean_token_reduction_ratio: {mean(reductions):.6f}" if reductions else "- mean_token_reduction_ratio: n/a",
        f"- mean_relation_token_reduction_ratio: {mean(relation_reductions):.6f}" if relation_reductions else "- mean_relation_token_reduction_ratio: n/a",
        f"- total_skipped_contains_relations: {sum(int(row['skipped_contains_relation_count']) for row in rows)}",
        "",
        "## Sample 37",
        "",
        _table(sample_37, ["stem", "old_total_tokens", "compact_total_tokens", "token_reduction", "relation_token_reduction"]),
        "",
        "## Longest After Compact",
        "",
        _table(
            sorted(rows, key=lambda row: int(row["compact_total_tokens"]), reverse=True)[:top_k],
            [
                "stem",
                "old_total_tokens",
                "compact_total_tokens",
                "token_reduction_ratio",
                "relation_token_count_estimate_old",
                "relation_token_count_estimate_compact",
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = list(iter_target_paths(args.target_root, split=args.split))
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    rows = run_benchmark(paths)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(summarize_rows(rows, top_k=int(args.top_k)), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output_jsonl}")
    print(f"wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
