from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_topology_placeholder_geometry import (  # noqa: E402
    GeometryPlaceholderLibrary,
    build_placeholder_targets_from_sample_rows,
    iter_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach retrieved placeholder geometry to generated manual topology samples.")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL topology samples containing token rows.")
    parser.add_argument("--geometry-split-root", type=Path, required=True, help="Manual target split root containing manifest.jsonl.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-geometry-targets", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]
    library = GeometryPlaceholderLibrary.from_split_manifest(
        args.geometry_split_root,
        seed=int(args.seed),
        max_geometry_targets=args.max_geometry_targets,
    )
    targets, summary = build_placeholder_targets_from_sample_rows(
        sample_rows,
        library,
        include_invalid=bool(args.include_invalid),
    )
    manifest_rows = []
    for index, target in enumerate(targets):
        sample_index = int(target.get("metadata", {}).get("sample_index", index))
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append(
            {
                "format": "maskgen_placeholder_geometry_manifest_row_v1",
                "sample_index": sample_index,
                "output_path": str(output_path.as_posix()),
                "node_count": int(len(target.get("parse_graph", {}).get("nodes", []) or [])),
                "relation_count": int(len(target.get("parse_graph", {}).get("relations", []) or [])),
                "attached_geometry_count": int(target.get("metadata", {}).get("attached_geometry_count", 0)),
                "missing_geometry_count": int(target.get("metadata", {}).get("missing_geometry_count", 0)),
                "attach_modes": target.get("metadata", {}).get("attach_modes", {}),
            }
        )
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary.update(
        {
            "samples": str(args.samples.as_posix()),
            "geometry_split_root": str(args.geometry_split_root.as_posix()),
            "output_root": str(args.output_root.as_posix()),
            "seed": int(args.seed),
            "geometry_library_size": int(len(library.targets)),
        }
    )
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached placeholder geometry samples={summary['output_count']} "
        f"skipped_invalid={summary['skipped_invalid_count']} "
        f"geometry_library={summary['geometry_library_size']} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
