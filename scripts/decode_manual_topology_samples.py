from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_topology_sample_validation import validate_topology_tokens  # noqa: E402
from partition_gen.parse_graph_compact_tokenizer import decode_topology_tokens_to_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode sampled MANUAL_TOPOLOGY_V1 token rows into topology target JSON files.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]

    manifest_rows = []
    skipped_invalid = 0
    for fallback_index, row in enumerate(sample_rows):
        tokens = [str(token) for token in row.get("tokens", [])]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not bool(args.include_invalid):
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target["metadata"].update(
            {
                "source_sample_index": sample_index,
                "source_samples": str(args.samples.as_posix()),
                "checkpoint": row.get("checkpoint"),
                "semantic_valid": bool(validation["semantic_valid"]),
            }
        )
        output_path = args.output_root / "topology" / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, topology_target)
        manifest_rows.append(
            {
                "format": "maskgen_manual_topology_decode_row_v1",
                "sample_index": sample_index,
                "source_samples": str(args.samples.as_posix()),
                "output_path": str(output_path.as_posix()),
                "valid": bool(validation["valid"]),
                "semantic_valid": bool(validation["semantic_valid"]),
                "node_count": int(validation["node_count_actual"]),
                "relation_counts": validation["relation_counts"],
                "length": int(validation["length"]),
            }
        )

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_manual_topology_decode_summary_v1",
        "samples": str(args.samples.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(sample_rows)),
        "decoded_count": int(len(manifest_rows)),
        "skipped_invalid_count": int(skipped_invalid),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"decoded topology samples={summary['decoded_count']} "
        f"skipped_invalid={summary['skipped_invalid_count']} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
