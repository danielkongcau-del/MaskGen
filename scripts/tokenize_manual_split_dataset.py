from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target, encode_topology_target  # noqa: E402
from partition_gen.parse_graph_tokenizer import (  # noqa: E402
    ParseGraphTokenizerConfig,
    build_token_vocabulary,
    save_vocabulary,
    tokens_to_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize topology/geometry split manual-rule dataset.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-token-ids", action="store_true")
    parser.add_argument("--coord-bins", type=int, default=1024)
    parser.add_argument("--area-bins", type=int, default=2048)
    parser.add_argument("--max-int", type=int, default=4096)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _tokenize_target(target_path: Path, *, target_kind: str, config: ParseGraphTokenizerConfig) -> list[str]:
    target = load_json(target_path)
    if target_kind == "topology":
        return encode_topology_target(target, config=config)
    if target_kind == "geometry":
        return encode_geometry_target(target, config=config)
    raise ValueError(f"Unknown target kind: {target_kind}")


def main() -> None:
    args = parse_args()
    config = ParseGraphTokenizerConfig(coord_bins=int(args.coord_bins), area_bins=int(args.area_bins), max_int=int(args.max_int))
    vocab = build_token_vocabulary(config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_vocabulary(args.output_root / "vocab.json", vocab, config=config)

    manifest_path = args.split_root / "manifest.jsonl"
    topology_sequences_path = args.output_root / "topology_sequences.jsonl"
    geometry_sequences_path = args.output_root / "geometry_sequences.jsonl"
    token_manifest_path = args.output_root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    topology_lengths: list[int] = []
    geometry_lengths: list[int] = []
    skipped_too_long = 0
    written_topology = 0
    written_geometry = 0
    with (
        topology_sequences_path.open("w", encoding="utf-8") as topology_handle,
        geometry_sequences_path.open("w", encoding="utf-8") as geometry_handle,
        token_manifest_path.open("w", encoding="utf-8") as manifest_handle,
    ):
        for row in rows:
            topology_path = Path(str(row["topology_path"]))
            topology_tokens = _tokenize_target(topology_path, target_kind="topology", config=config)
            topology_written = args.max_tokens is None or len(topology_tokens) <= int(args.max_tokens)
            if topology_written:
                sequence_row = {
                    "format": "maskgen_tokenized_parse_graph_v1",
                    "tokenizer": "manual_topology_v1",
                    "source_target": str(topology_path.as_posix()),
                    "stem": row.get("stem"),
                    "length": int(len(topology_tokens)),
                    "tokens": topology_tokens,
                }
                if not args.no_token_ids:
                    sequence_row["ids"] = tokens_to_ids(topology_tokens, vocab)
                write_jsonl_row(topology_handle, sequence_row)
                topology_lengths.append(len(topology_tokens))
                written_topology += 1
            else:
                skipped_too_long += 1

            geometry_written_count = 0
            for geometry_path_value in row.get("geometry_paths", []) or []:
                geometry_path = Path(str(geometry_path_value))
                geometry_tokens = _tokenize_target(geometry_path, target_kind="geometry", config=config)
                geometry_written = args.max_tokens is None or len(geometry_tokens) <= int(args.max_tokens)
                if geometry_written:
                    geometry_target = load_json(geometry_path)
                    sequence_row = {
                        "format": "maskgen_tokenized_parse_graph_v1",
                        "tokenizer": "manual_geometry_v1",
                        "source_target": str(geometry_path.as_posix()),
                        "stem": row.get("stem"),
                        "source_node_id": geometry_target.get("source_node_id"),
                        "length": int(len(geometry_tokens)),
                        "tokens": geometry_tokens,
                    }
                    if not args.no_token_ids:
                        sequence_row["ids"] = tokens_to_ids(geometry_tokens, vocab)
                    write_jsonl_row(geometry_handle, sequence_row)
                    geometry_lengths.append(len(geometry_tokens))
                    written_geometry += 1
                    geometry_written_count += 1
                else:
                    skipped_too_long += 1

            write_jsonl_row(
                manifest_handle,
                {
                    "stem": row.get("stem"),
                    "topology_path": row.get("topology_path"),
                    "topology_written": bool(topology_written),
                    "geometry_written_count": int(geometry_written_count),
                    "geometry_target_count": int(row.get("geometry_target_count", 0)),
                    "max_single_sequence_tokens": row.get("max_single_sequence_tokens"),
                },
            )

    summary = {
        "format": "maskgen_manual_split_tokenized_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "config": asdict(config),
        "vocab_size": int(len(vocab)),
        "sample_count": int(len(rows)),
        "written_topology": int(written_topology),
        "written_geometry": int(written_geometry),
        "skipped_too_long": int(skipped_too_long),
        "topology_mean_length": float(mean(topology_lengths)) if topology_lengths else None,
        "topology_max_length": int(max(topology_lengths)) if topology_lengths else None,
        "geometry_mean_length": float(mean(geometry_lengths)) if geometry_lengths else None,
        "geometry_max_length": int(max(geometry_lengths)) if geometry_lengths else None,
        "topology_sequences_path": str(topology_sequences_path.as_posix()),
        "geometry_sequences_path": str(geometry_sequences_path.as_posix()),
        "manifest_path": str(token_manifest_path.as_posix()),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"tokenized split dataset samples={len(rows)} topology={written_topology} "
        f"geometry={written_geometry} skipped_too_long={skipped_too_long} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
