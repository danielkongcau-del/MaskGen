from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_split_validator import validate_topology_geometry_split  # noqa: E402
from partition_gen.manual_target_split import merge_topology_geometry_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a manual topology/geometry split dataset into full parse graph targets.")
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="Root containing manifest.jsonl, topology/, and geometry/. Tokenized roots with summary.json are also accepted.",
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split name when --split-root is a dataset root.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-missing-geometry", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_split_root_and_manifest(split_root: Path, *, split: str | None = None) -> tuple[Path, Path]:
    candidates: list[Path] = []

    summary_path = split_root / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary_split_root = summary.get("split_root")
        if summary_split_root is not None:
            candidates.append(Path(str(summary_split_root)) / "manifest.jsonl")
            if split:
                candidates.append(Path(str(summary_split_root)) / split / "manifest.jsonl")

    candidates.append(split_root / "manifest.jsonl")
    if split:
        candidates.append(split_root / split / "manifest.jsonl")

    for manifest_path in candidates:
        if manifest_path.exists():
            return manifest_path.parent, manifest_path

    searched = "\n".join(f"  - {path.as_posix()}" for path in candidates)
    raise FileNotFoundError(
        "Could not find a manual topology/geometry split manifest.jsonl.\n"
        f"Searched:\n{searched}\n"
        "This replay step needs the target split JSON dataset, not only topology/geometry token JSONL files. "
        "If the target split dataset is missing, rebuild it with scripts/build_manual_split_dataset.py first."
    )


def resolve_manifest_path(value: object, *, split_root: Path, manifest_parent: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    for base in (manifest_parent, split_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def replay_row(row: dict, *, split_root: Path, manifest_parent: Path, output_root: Path, require_all_geometry: bool) -> dict:
    topology_path = resolve_manifest_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_parent)
    if int(row.get("geometry_target_count", 0)) > 0 and not row.get("geometry_paths"):
        raise ValueError(
            "Manifest row has geometry_target_count > 0 but no geometry_paths. "
            "This looks like a tokenized split manifest; replay needs the target split manifest."
        )
    geometry_paths = [
        resolve_manifest_path(value, split_root=split_root, manifest_parent=manifest_parent)
        for value in row.get("geometry_paths", []) or []
    ]
    topology_target = load_json(topology_path)
    geometry_targets = [load_json(path) for path in geometry_paths]
    validation = validate_topology_geometry_split(topology_target, geometry_targets)
    merged_target = merge_topology_geometry_targets(
        topology_target,
        geometry_targets,
        require_all_geometry=require_all_geometry,
    )

    stem = str(row.get("stem") or topology_path.stem)
    output_path = output_root / "graphs" / f"{stem}.json"
    dump_json(output_path, merged_target)

    graph = merged_target["parse_graph"]
    return {
        "format": "maskgen_manual_split_replay_row_v1",
        "stem": stem,
        "source_target": row.get("source_target"),
        "topology_path": str(topology_path.as_posix()),
        "geometry_target_count": int(len(geometry_targets)),
        "output_path": str(output_path.as_posix()),
        "split_valid": bool(validation["is_valid"]),
        "missing_geometry_ref_count": int(len(validation["missing_geometry_ref_ids"])),
        "extra_geometry_target_count": int(len(validation["extra_geometry_target_ids"])),
        "duplicate_geometry_target_count": int(len(validation["duplicate_geometry_target_ids"])),
        "invalid_relation_ref_count": int(len(validation["invalid_relation_refs"])),
        "node_count": int(len(graph.get("nodes", []) or [])),
        "relation_count": int(len(graph.get("relations", []) or [])),
        "attached_geometry_count": int(merged_target.get("metadata", {}).get("attached_geometry_count", 0)),
    }


def summarize(rows: Sequence[dict], *, split_root: Path, output_root: Path) -> dict:
    node_counts = [int(row["node_count"]) for row in rows]
    relation_counts = [int(row["relation_count"]) for row in rows]
    return {
        "format": "maskgen_manual_split_replay_summary_v1",
        "split_root": str(split_root.as_posix()),
        "output_root": str(output_root.as_posix()),
        "sample_count": int(len(rows)),
        "split_valid_count": int(sum(1 for row in rows if bool(row["split_valid"]))),
        "missing_geometry_ref_count": int(sum(int(row["missing_geometry_ref_count"]) for row in rows)),
        "invalid_relation_ref_count": int(sum(int(row["invalid_relation_ref_count"]) for row in rows)),
        "geometry_target_count": int(sum(int(row["geometry_target_count"]) for row in rows)),
        "attached_geometry_count": int(sum(int(row["attached_geometry_count"]) for row in rows)),
        "node_count_mean": float(mean(node_counts)) if node_counts else 0.0,
        "relation_count_mean": float(mean(relation_counts)) if relation_counts else 0.0,
    }


def main() -> None:
    args = parse_args()
    split_root, manifest_path = resolve_split_root_and_manifest(args.split_root, split=args.split)
    rows = list(iter_jsonl(manifest_path))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    output_rows = [
        replay_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
            output_root=args.output_root,
            require_all_geometry=not bool(args.allow_missing_geometry),
        )
        for row in rows
    ]
    write_jsonl(args.output_root / "manifest.jsonl", output_rows)
    summary = summarize(output_rows, split_root=split_root, output_root=args.output_root)
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"replayed samples={summary['sample_count']} "
        f"split_valid={summary['split_valid_count']} "
        f"attached_geometry={summary['attached_geometry_count']} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
