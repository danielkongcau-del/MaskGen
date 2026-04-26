from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_target_geometry_simplify import (  # noqa: E402
    ManualTargetSimplifyConfig,
    simplify_manual_generator_target,
)
from partition_gen.manual_target_token_stats import analyze_manual_target_token_stats  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark manual-rule target polygon simplification profiles.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--profiles", nargs="+", default=["light", "medium", "aggressive"])
    parser.add_argument("--write-simplified-samples", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def benchmark_target(
    target: Dict[str, object],
    *,
    profile: str,
    tokenizer_config: ParseGraphTokenizerConfig | None = None,
) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    tokenizer_config = tokenizer_config or ParseGraphTokenizerConfig()
    original_stats = analyze_manual_target_token_stats(target, tokenizer_config=tokenizer_config)
    simplified, simplify_diag = simplify_manual_generator_target(
        target,
        config=ManualTargetSimplifyConfig(profile=profile),
    )
    simplified_stats = analyze_manual_target_token_stats(simplified, tokenizer_config=tokenizer_config)
    original_tokens = int(original_stats["total_tokens"])
    simplified_tokens = int(simplified_stats["total_tokens"])
    original_vertices = int(original_stats["polygon_vertex_count"])
    simplified_vertices = int(simplified_stats["polygon_vertex_count"])
    token_reduction = original_tokens - simplified_tokens
    vertex_reduction = original_vertices - simplified_vertices
    row = {
        "profile": profile,
        "valid": bool(int(simplify_diag["invalid_geometry_count"]) == 0),
        "original_total_tokens": int(original_tokens),
        "simplified_total_tokens": int(simplified_tokens),
        "token_reduction": int(token_reduction),
        "token_reduction_ratio": float(token_reduction / original_tokens) if original_tokens else 0.0,
        "original_polygon_vertex_count": int(original_vertices),
        "simplified_polygon_vertex_count": int(simplified_vertices),
        "vertex_reduction": int(vertex_reduction),
        "vertex_reduction_ratio": float(vertex_reduction / original_vertices) if original_vertices else 0.0,
        "node_count": int(original_stats["node_count"]),
        "relation_count": int(original_stats["relation_count"]),
        "simplified_node_count": int(simplify_diag["simplified_node_count"]),
        "failed_node_count": int(simplify_diag["failed_node_count"]),
        "invalid_geometry_count": int(simplify_diag["invalid_geometry_count"]),
        "area_error_total": float(simplify_diag["area_error_total"]),
        "area_error_ratio_mean": float(simplify_diag["area_error_ratio_mean"]),
        "max_node_area_error_ratio": float(simplify_diag["max_node_area_error_ratio"]),
        "top_changed_nodes": simplify_diag.get("top_changed_nodes", [])[:20],
    }
    return row, simplified, simplify_diag


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


def summarize_profile(rows: Sequence[Dict[str, object]], profile: str) -> str:
    token_after = [int(row["simplified_total_tokens"]) for row in rows]
    token_reductions = [float(row["token_reduction_ratio"]) for row in rows]
    vertex_reductions = [float(row["vertex_reduction_ratio"]) for row in rows]
    lines = [
        f"# Manual Target Simplification Summary: {profile}",
        "",
        f"- samples: {len(rows)}",
        f"- valid_samples: {sum(1 for row in rows if row.get('valid'))}",
        f"- mean_token_reduction_ratio: {mean(token_reductions):.6f}" if token_reductions else "- mean_token_reduction_ratio: n/a",
        f"- p50_tokens_after: {_percentile(token_after, 0.50)}",
        f"- p90_tokens_after: {_percentile(token_after, 0.90)}",
        f"- p95_tokens_after: {_percentile(token_after, 0.95)}",
        f"- max_tokens_after: {max(token_after) if token_after else 'n/a'}",
        f"- mean_vertex_reduction_ratio: {mean(vertex_reductions):.6f}" if vertex_reductions else "- mean_vertex_reduction_ratio: n/a",
        f"- invalid_geometry_count_total: {sum(int(row['invalid_geometry_count']) for row in rows)}",
        f"- failed_node_count_total: {sum(int(row['failed_node_count']) for row in rows)}",
        f"- area_error_ratio_mean: {mean([float(row['area_error_ratio_mean']) for row in rows]):.6f}" if rows else "- area_error_ratio_mean: n/a",
        "",
        "## Largest Token Reduction",
        "",
        _table(
            sorted(rows, key=lambda item: float(item["token_reduction_ratio"]), reverse=True)[:20],
            ["stem", "original_total_tokens", "simplified_total_tokens", "token_reduction", "token_reduction_ratio"],
        ),
        "",
        "## Largest Area Error",
        "",
        _table(
            sorted(rows, key=lambda item: float(item["max_node_area_error_ratio"]), reverse=True)[:20],
            ["stem", "max_node_area_error_ratio", "area_error_ratio_mean", "failed_node_count", "invalid_geometry_count"],
        ),
        "",
        "## Simplification Failures",
        "",
        _table(
            [row for row in rows if int(row["failed_node_count"]) > 0][:20],
            ["stem", "failed_node_count", "invalid_geometry_count", "original_total_tokens", "simplified_total_tokens"],
        ),
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    target_paths: Sequence[Path],
    *,
    profiles: Sequence[str],
    output_root: Path | None = None,
    write_simplified_samples: bool = False,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    tokenizer_config = ParseGraphTokenizerConfig()
    for path in target_paths:
        target = load_json(path)
        if target.get("format") != "maskgen_generator_target_v1" or target.get("target_type") != "parse_graph":
            continue
        for profile in profiles:
            row, simplified, _ = benchmark_target(target, profile=profile, tokenizer_config=tokenizer_config)
            row = {"path": str(path.as_posix()), "stem": path.stem, **row}
            rows.append(row)
            if output_root is not None and write_simplified_samples:
                dump_json(output_root / str(profile) / "graphs" / f"{path.stem}.json", simplified)
    return rows


def main() -> None:
    args = parse_args()
    paths = list(iter_target_paths(args.target_root, split=args.split))
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(
        paths,
        profiles=[str(profile) for profile in args.profiles],
        output_root=args.output_root,
        write_simplified_samples=bool(args.write_simplified_samples),
    )
    result_path = args.output_root / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    for profile in args.profiles:
        profile_rows = [row for row in rows if row["profile"] == profile]
        (args.output_root / f"summary_{profile}.md").write_text(summarize_profile(profile_rows, str(profile)), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {result_path}")
    print(f"wrote summaries to {args.output_root}")


if __name__ == "__main__":
    main()
