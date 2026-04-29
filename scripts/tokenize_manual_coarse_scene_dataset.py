from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_coarse_scene_ar import build_coarse_scene_sequence_rows  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary, save_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize parent-first manual coarse scene sequences.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-token-ids", action="store_true")
    parser.add_argument("--max-int", type=int, default=4096)
    parser.add_argument("--coarse-grid-bins", type=int, default=8)
    parser.add_argument("--coarse-size-bins", type=int, default=8)
    parser.add_argument("--coarse-aspect-bins", type=int, default=8)
    parser.add_argument("--coarse-angle-bins", type=int, default=8)
    parser.add_argument("--coarse-relation-bins", type=int, default=8)
    return parser.parse_args()


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def main() -> None:
    args = parse_args()
    config = ParseGraphTokenizerConfig(
        max_int=int(args.max_int),
        coarse_grid_bins=int(args.coarse_grid_bins),
        coarse_size_bins=int(args.coarse_size_bins),
        coarse_aspect_bins=int(args.coarse_aspect_bins),
        coarse_angle_bins=int(args.coarse_angle_bins),
        coarse_relation_bins=int(args.coarse_relation_bins),
    )
    vocab = build_token_vocabulary(config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_vocabulary(args.output_root / "vocab.json", vocab, config=config)
    rows, summary = build_coarse_scene_sequence_rows(
        args.split_root,
        config=config,
        vocab=vocab,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        include_token_ids=not bool(args.no_token_ids),
    )
    sequences_path = args.output_root / "coarse_scene_sequences.jsonl"
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
                    "action_count": row.get("action_count"),
                    "action_histogram": row.get("action_histogram"),
                    "anchor_mode_histogram": row.get("anchor_mode_histogram"),
                    "relation_histogram": row.get("relation_histogram"),
                    "dependency_fallback_count": row.get("dependency_fallback_count"),
                    "forward_reference_count": row.get("forward_reference_count"),
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
        f"tokenized coarse scene samples={summary['sample_count']} "
        f"sequences={summary['written_coarse_scene']} "
        f"skipped_too_long={summary['skipped_too_long']} "
        f"forward_refs={summary['forward_reference_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
