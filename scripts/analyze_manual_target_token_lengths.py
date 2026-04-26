from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_target_token_stats import (  # noqa: E402
    analyze_manual_target_token_stats,
    summarize_manual_token_stat_rows,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze manual-rule generator target token length drivers.")
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


def iter_target_paths(root: Path, split: str | None = None):
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


def analyze_paths(paths, *, tokenizer_config: ParseGraphTokenizerConfig) -> list[dict]:
    rows = []
    for path in paths:
        target = load_json(path)
        if target.get("format") != "maskgen_generator_target_v1" or target.get("target_type") != "parse_graph":
            continue
        stats = analyze_manual_target_token_stats(target, tokenizer_config=tokenizer_config)
        rows.append({"path": str(path.as_posix()), "stem": path.stem, **stats})
    return rows


def main() -> None:
    args = parse_args()
    paths = list(iter_target_paths(args.target_root, split=args.split))
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    rows = analyze_paths(paths, tokenizer_config=ParseGraphTokenizerConfig())
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(summarize_manual_token_stat_rows(rows, top_k=int(args.top_k)), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output_jsonl}")
    print(f"wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
