from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_layout_ar import build_layout_sequence_rows  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary, save_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize topology-conditioned manual layout/frame sequences.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-token-ids", action="store_true")
    parser.add_argument("--max-int", type=int, default=4096)
    return parser.parse_args()


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    config = ParseGraphTokenizerConfig(max_int=int(args.max_int))
    vocab = build_token_vocabulary(config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_vocabulary(args.output_root / "vocab.json", vocab, config=config)
    rows, summary = build_layout_sequence_rows(
        args.split_root,
        config=config,
        vocab=vocab,
        max_tokens=args.max_tokens,
        include_token_ids=not bool(args.no_token_ids),
    )
    sequences_path = args.output_root / "layout_sequences.jsonl"
    manifest_path = args.output_root / "manifest.jsonl"
    with sequences_path.open("w", encoding="utf-8") as sequence_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for row in rows:
            write_jsonl_row(sequence_handle, row)
            write_jsonl_row(
                manifest_handle,
                {
                    "stem": row.get("stem"),
                    "source_topology": row.get("source_topology"),
                    "length": row.get("length"),
                    "topology_length": row.get("topology_length"),
                    "layout_length": row.get("layout_length"),
                    "layout_node_count": row.get("layout_node_count"),
                    "loss_start_index": row.get("loss_start_index"),
                },
            )
    summary.update(
        {
            "output_root": str(args.output_root.as_posix()),
            "config": asdict(config),
            "vocab_size": int(len(vocab)),
            "sequences_path": str(sequences_path.as_posix()),
            "manifest_path": str(manifest_path.as_posix()),
        }
    )
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"tokenized layout samples={summary['sample_count']} "
        f"sequences={summary['written_layout']} "
        f"skipped_too_long={summary['skipped_too_long']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
