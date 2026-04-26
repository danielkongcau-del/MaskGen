from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.parse_graph_tokenizer import (  # noqa: E402
    ParseGraphTokenizerConfig,
    build_token_vocabulary,
    encode_generator_target,
    iter_training_manifest_rows,
    load_generator_target,
    save_vocabulary,
    tokens_to_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize generator target parse_graph JSON files.")
    parser.add_argument("--target-root", type=Path, default=Path("data/remote_256_generator_targets_cdt_greedy"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_generator_tokens_cdt_greedy"))
    parser.add_argument("--include-non-usable", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--coord-bins", type=int, default=1024)
    parser.add_argument("--area-bins", type=int, default=2048)
    parser.add_argument("--max-int", type=int, default=4096)
    parser.add_argument("--no-token-ids", action="store_true")
    return parser.parse_args()


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    config = ParseGraphTokenizerConfig(
        coord_bins=int(args.coord_bins),
        area_bins=int(args.area_bins),
        max_int=int(args.max_int),
    )
    vocab = build_token_vocabulary(config)
    output_split_root = args.output_root / args.split
    output_split_root.mkdir(parents=True, exist_ok=True)
    save_vocabulary(args.output_root / "vocab.json", vocab, config=config)

    manifest_path = args.target_root / args.split / "manifest.jsonl"
    sequences_path = output_split_root / "sequences.jsonl"
    token_manifest_path = output_split_root / "manifest.jsonl"

    processed = 0
    written = 0
    skipped_too_long = 0
    skipped_missing = 0
    length_values: list[int] = []
    with sequences_path.open("w", encoding="utf-8") as seq_handle, token_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for row in iter_training_manifest_rows(manifest_path, include_non_usable=bool(args.include_non_usable)):
            if args.max_samples is not None and processed >= int(args.max_samples):
                break
            processed += 1
            target_path_value = row.get("target_path")
            if not target_path_value:
                skipped_missing += 1
                continue
            target_path = Path(str(target_path_value))
            if not target_path.exists():
                skipped_missing += 1
                continue
            target = load_generator_target(target_path)
            tokens = encode_generator_target(target, config=config)
            if args.max_tokens is not None and len(tokens) > int(args.max_tokens):
                skipped_too_long += 1
                write_jsonl_row(
                    manifest_handle,
                    {
                        "source_target": str(target_path.as_posix()),
                        "stem": row.get("stem"),
                        "written": False,
                        "skip_reason": "too_long",
                        "length": int(len(tokens)),
                    },
                )
                continue
            ids = [] if args.no_token_ids else tokens_to_ids(tokens, vocab)
            sequence_row = {
                "format": "maskgen_tokenized_parse_graph_v1",
                "tokenizer": "weak_parse_graph_structured_v1",
                "source_target": str(target_path.as_posix()),
                "split": args.split,
                "stem": row.get("stem"),
                "length": int(len(tokens)),
                "tokens": tokens,
            }
            if not args.no_token_ids:
                sequence_row["ids"] = ids
            write_jsonl_row(seq_handle, sequence_row)
            write_jsonl_row(
                manifest_handle,
                {
                    "source_target": str(target_path.as_posix()),
                    "stem": row.get("stem"),
                    "written": True,
                    "skip_reason": None,
                    "length": int(len(tokens)),
                    "node_count": row.get("node_count"),
                    "semantic_face_count": row.get("semantic_face_count"),
                    "convex_atom_count": row.get("convex_atom_count"),
                    "relation_count": row.get("relation_count"),
                    "full_iou": row.get("full_iou"),
                    "mask_pixel_accuracy": row.get("mask_pixel_accuracy"),
                },
            )
            written += 1
            length_values.append(len(tokens))

    summary = {
        "format": "maskgen_tokenized_parse_graph_summary_v1",
        "target_root": str(args.target_root.as_posix()),
        "split": args.split,
        "output_root": str(args.output_root.as_posix()),
        "config": asdict(config),
        "vocab_size": int(len(vocab)),
        "processed": int(processed),
        "written": int(written),
        "skipped_missing": int(skipped_missing),
        "skipped_too_long": int(skipped_too_long),
        "min_length": int(min(length_values)) if length_values else None,
        "max_length": int(max(length_values)) if length_values else None,
        "mean_length": float(sum(length_values) / len(length_values)) if length_values else None,
        "sequences_path": str(sequences_path.as_posix()),
        "manifest_path": str(token_manifest_path.as_posix()),
        "vocab_path": str((args.output_root / "vocab.json").as_posix()),
    }
    summary_path = output_split_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"tokenized split={args.split}: processed={processed} written={written} "
        f"skipped_missing={skipped_missing} skipped_too_long={skipped_too_long} "
        f"mean_length={summary['mean_length']} max_length={summary['max_length']} "
        f"vocab_size={len(vocab)} output={sequences_path}"
    )


if __name__ == "__main__":
    main()
