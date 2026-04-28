from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_split_materialize import materialize_manual_split_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize topology/geometry split rows into full parse-graph targets.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def main() -> None:
    args = parse_args()
    targets = materialize_manual_split_targets(args.split_root, max_samples=args.max_samples)
    manifest_rows: list[dict] = []
    attached = 0
    missing = 0
    attach_modes: Counter[str] = Counter()
    for index, target in enumerate(targets):
        output_path = args.output_root / "graphs" / f"sample_{index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append({"sample_index": int(index), "output_path": str(output_path.as_posix())})
        metadata = target.get("metadata", {}) or {}
        attached += int(metadata.get("attached_geometry_count", 0))
        missing += int(metadata.get("missing_geometry_count", 0))
        attach_modes.update(metadata.get("attach_modes", {}) or {})

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_manual_split_materialized_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(targets)),
        "output_count": int(len(manifest_rows)),
        "attached_geometry_count": int(attached),
        "missing_geometry_count": int(missing),
        "attach_modes": dict(attach_modes),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"materialized split targets samples={summary['output_count']} "
        f"attached={attached} missing={missing} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
