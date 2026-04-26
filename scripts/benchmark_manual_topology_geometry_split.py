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

from partition_gen.manual_target_split import build_topology_geometry_split_targets  # noqa: E402
from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target, encode_topology_target  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, encode_generator_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark topology/geometry split manual-rule targets.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", "--output-jsonl", dest="output_jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--write-split-samples", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
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


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * percentile))
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def benchmark_target(path: Path, target: dict, *, config: ParseGraphTokenizerConfig | None = None) -> tuple[dict, dict, list[dict]]:
    config = config or ParseGraphTokenizerConfig()
    old_tokens = encode_generator_target(target, config=config)
    topology_target, geometry_targets, diagnostics = build_topology_geometry_split_targets(
        target,
        source_target=str(path.as_posix()),
    )
    topology_tokens = len(encode_topology_target(topology_target, config=config))
    geometry_lengths = [len(encode_geometry_target(geometry_target, config=config)) for geometry_target in geometry_targets]
    geometry_tokens_total = int(sum(geometry_lengths))
    geometry_tokens_max = int(max(geometry_lengths)) if geometry_lengths else 0
    longest_geometry_targets = sorted(
        [
            {
                "source_node_id": geometry_target.get("source_node_id"),
                "role": geometry_target.get("role"),
                "label": geometry_target.get("label"),
                "geometry_model": geometry_target.get("geometry_model"),
                "tokens": int(length),
            }
            for geometry_target, length in zip(geometry_targets, geometry_lengths)
        ],
        key=lambda item: int(item["tokens"]),
        reverse=True,
    )[:20]
    row = {
        "path": str(path.as_posix()),
        "stem": path.stem,
        "old_total_tokens": int(len(old_tokens)),
        "topology_tokens": int(topology_tokens),
        "geometry_target_count": int(len(geometry_targets)),
        "geometry_tokens_total": int(geometry_tokens_total),
        "geometry_tokens_mean": float(geometry_tokens_total / len(geometry_lengths)) if geometry_lengths else 0.0,
        "geometry_tokens_p95": int(_percentile(geometry_lengths, 0.95) or 0),
        "geometry_tokens_max": int(geometry_tokens_max),
        "max_single_sequence_tokens": int(max(topology_tokens, geometry_tokens_max)),
        "node_count": int(diagnostics["topology_node_count"]),
        "relation_count": int(diagnostics["topology_relation_count"]),
        "renderable_node_count": int(diagnostics["renderable_node_count"]),
        "non_renderable_node_count": int(diagnostics["non_renderable_node_count"]),
        "reference_only_count": int(diagnostics["reference_only_count"]),
        "longest_geometry_targets": longest_geometry_targets,
    }
    return row, topology_target, geometry_targets


def run_benchmark(
    paths: Sequence[Path],
    *,
    output_root: Path | None = None,
    write_split_samples: bool = False,
) -> list[Dict[str, object]]:
    config = ParseGraphTokenizerConfig()
    rows: list[Dict[str, object]] = []
    for path in paths:
        target = load_json(path)
        if target.get("format") != "maskgen_generator_target_v1" or target.get("target_type") != "parse_graph":
            continue
        row, topology_target, geometry_targets = benchmark_target(path, target, config=config)
        rows.append(row)
        if output_root is not None and write_split_samples:
            dump_json(output_root / "topology" / "graphs" / f"{path.stem}.json", topology_target)
            for geometry_target in geometry_targets:
                node_id = str(geometry_target.get("source_node_id"))
                dump_json(output_root / "geometry" / path.stem / f"{node_id}.json", geometry_target)
    return rows


def summarize_rows(rows: Sequence[Dict[str, object]], *, top_k: int = 20) -> str:
    old_lengths = [int(row["old_total_tokens"]) for row in rows]
    topology_lengths = [int(row["topology_tokens"]) for row in rows]
    geometry_max_lengths = [int(row["geometry_tokens_max"]) for row in rows]
    max_single = [int(row["max_single_sequence_tokens"]) for row in rows]
    sample_37 = [row for row in rows if str(row.get("stem")) == "37"]
    longest_geometry = []
    for row in rows:
        for item in row.get("longest_geometry_targets", [])[:3]:
            longest_geometry.append({"stem": row.get("stem"), **item})
    longest_geometry = sorted(longest_geometry, key=lambda item: int(item["tokens"]), reverse=True)[:top_k]
    lines = [
        "# Manual Topology/Geometry Split Benchmark",
        "",
        f"- samples: {len(rows)}",
        f"- old_mean_tokens: {mean(old_lengths):.2f}" if old_lengths else "- old_mean_tokens: n/a",
        f"- old_p95_tokens: {_percentile(old_lengths, 0.95)}",
        f"- old_max_tokens: {max(old_lengths) if old_lengths else 'n/a'}",
        f"- topology_mean_tokens: {mean(topology_lengths):.2f}" if topology_lengths else "- topology_mean_tokens: n/a",
        f"- topology_p95_tokens: {_percentile(topology_lengths, 0.95)}",
        f"- topology_max_tokens: {max(topology_lengths) if topology_lengths else 'n/a'}",
        f"- geometry_max_mean_tokens: {mean(geometry_max_lengths):.2f}" if geometry_max_lengths else "- geometry_max_mean_tokens: n/a",
        f"- geometry_max_p95_tokens: {_percentile(geometry_max_lengths, 0.95)}",
        f"- geometry_max_max_tokens: {max(geometry_max_lengths) if geometry_max_lengths else 'n/a'}",
        f"- max_single_sequence_mean_tokens: {mean(max_single):.2f}" if max_single else "- max_single_sequence_mean_tokens: n/a",
        f"- max_single_sequence_p95_tokens: {_percentile(max_single, 0.95)}",
        f"- max_single_sequence_max_tokens: {max(max_single) if max_single else 'n/a'}",
        f"- samples_over_4096: {sum(1 for value in max_single if value > 4096)}",
        f"- samples_over_6144: {sum(1 for value in max_single if value > 6144)}",
        "",
        "## Sample 37",
        "",
        _table(sample_37, ["stem", "old_total_tokens", "topology_tokens", "geometry_tokens_max", "max_single_sequence_tokens"]),
        "",
        "## Longest Topology",
        "",
        _table(
            sorted(rows, key=lambda row: int(row["topology_tokens"]), reverse=True)[:top_k],
            ["stem", "old_total_tokens", "topology_tokens", "geometry_tokens_max", "max_single_sequence_tokens"],
        ),
        "",
        "## Longest Geometry Targets",
        "",
        _table(longest_geometry, ["stem", "source_node_id", "role", "label", "geometry_model", "tokens"]),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = list(iter_target_paths(args.target_root, split=args.split))
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    rows = run_benchmark(
        paths,
        output_root=args.output_root,
        write_split_samples=bool(args.write_split_samples),
    )
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
