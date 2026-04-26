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
from partition_gen.manual_target_split import build_topology_geometry_split_targets  # noqa: E402
from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target, encode_topology_target  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, encode_generator_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a formal topology/geometry split dataset from manual-rule targets.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--only-training-usable", action="store_true")
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


def _target_training_usable(target: dict) -> bool:
    return bool(target.get("metadata", {}).get("training_usable", True))


def build_split_dataset_row(
    target_path: Path,
    *,
    output_split_root: Path,
    config: ParseGraphTokenizerConfig | None = None,
) -> tuple[dict, dict, list[dict]]:
    config = config or ParseGraphTokenizerConfig()
    target = load_json(target_path)
    topology_target, geometry_targets, split_diagnostics = build_topology_geometry_split_targets(
        target,
        source_target=str(target_path.as_posix()),
    )
    validation = validate_topology_geometry_split(topology_target, geometry_targets)
    topology_path = output_split_root / "topology" / "graphs" / f"{target_path.stem}.json"
    geometry_dir = output_split_root / "geometry" / target_path.stem
    geometry_paths = [geometry_dir / f"{geometry_target['source_node_id']}.json" for geometry_target in geometry_targets]

    old_total_tokens = len(encode_generator_target(target, config=config))
    topology_tokens = len(encode_topology_target(topology_target, config=config))
    geometry_lengths = [len(encode_geometry_target(geometry_target, config=config)) for geometry_target in geometry_targets]
    geometry_tokens_max = max(geometry_lengths) if geometry_lengths else 0

    row = {
        "source_target": str(target_path.as_posix()),
        "stem": target_path.stem,
        "target_written": True,
        "training_usable": bool(_target_training_usable(target) and validation["is_valid"]),
        "topology_path": str(topology_path.as_posix()),
        "geometry_dir": str(geometry_dir.as_posix()),
        "geometry_paths": [str(path.as_posix()) for path in geometry_paths],
        "old_total_tokens": int(old_total_tokens),
        "topology_tokens": int(topology_tokens),
        "geometry_target_count": int(len(geometry_targets)),
        "geometry_tokens_total": int(sum(geometry_lengths)),
        "geometry_tokens_mean": float(sum(geometry_lengths) / len(geometry_lengths)) if geometry_lengths else 0.0,
        "geometry_tokens_max": int(geometry_tokens_max),
        "max_single_sequence_tokens": int(max(topology_tokens, geometry_tokens_max)),
        "split_valid": bool(validation["is_valid"]),
        "missing_geometry_ref_count": int(len(validation["missing_geometry_ref_ids"])),
        "invalid_relation_ref_count": int(len(validation["invalid_relation_refs"])),
        "topology_nodes_with_geometry_payload_count": int(len(validation["topology_nodes_with_geometry_payload"])),
        **split_diagnostics,
    }
    return row, topology_target, geometry_targets


def _summary(rows: Sequence[dict], *, target_root: Path, split: str, output_root: Path) -> dict:
    max_single = [int(row["max_single_sequence_tokens"]) for row in rows]
    topology_lengths = [int(row["topology_tokens"]) for row in rows]
    geometry_max = [int(row["geometry_tokens_max"]) for row in rows]
    return {
        "format": "maskgen_manual_split_dataset_summary_v1",
        "target_root": str(target_root.as_posix()),
        "split": split,
        "output_root": str(output_root.as_posix()),
        "sample_count": int(len(rows)),
        "training_usable_count": int(sum(1 for row in rows if bool(row.get("training_usable")))),
        "split_valid_count": int(sum(1 for row in rows if bool(row.get("split_valid")))),
        "geometry_target_count": int(sum(int(row["geometry_target_count"]) for row in rows)),
        "max_single_sequence_tokens_mean": float(mean(max_single)) if max_single else 0.0,
        "max_single_sequence_tokens_max": int(max(max_single)) if max_single else 0,
        "topology_tokens_mean": float(mean(topology_lengths)) if topology_lengths else 0.0,
        "topology_tokens_max": int(max(topology_lengths)) if topology_lengths else 0,
        "geometry_tokens_max_mean": float(mean(geometry_max)) if geometry_max else 0.0,
        "geometry_tokens_max_max": int(max(geometry_max)) if geometry_max else 0,
        "samples_over_4096": int(sum(1 for value in max_single if value > 4096)),
        "samples_over_6144": int(sum(1 for value in max_single if value > 6144)),
    }


def build_split_dataset(
    target_paths: Sequence[Path],
    *,
    output_split_root: Path,
    target_root: Path,
    split: str,
    only_training_usable: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    config = ParseGraphTokenizerConfig()
    for target_path in target_paths:
        target = load_json(target_path)
        if target.get("format") != "maskgen_generator_target_v1" or target.get("target_type") != "parse_graph":
            continue
        if only_training_usable and not _target_training_usable(target):
            continue
        row, topology_target, geometry_targets = build_split_dataset_row(
            target_path,
            output_split_root=output_split_root,
            config=config,
        )
        dump_json(Path(row["topology_path"]), topology_target)
        for geometry_target, geometry_path in zip(geometry_targets, row["geometry_paths"]):
            dump_json(Path(geometry_path), geometry_target)
        rows.append(row)

    manifest_path = output_split_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = _summary(rows, target_root=target_root, split=split, output_root=output_split_root.parent)
    dump_json(output_split_root / "summary.json", summary)
    return rows


def main() -> None:
    args = parse_args()
    paths = list(iter_target_paths(args.target_root, split=args.split))
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    output_split_root = args.output_root / args.split
    rows = build_split_dataset(
        paths,
        output_split_root=output_split_root,
        target_root=args.target_root,
        split=args.split,
        only_training_usable=bool(args.only_training_usable),
    )
    print(
        f"built split dataset split={args.split} samples={len(rows)} "
        f"geometry_targets={sum(int(row['geometry_target_count']) for row in rows)} "
        f"manifest={output_split_root / 'manifest.jsonl'}"
    )


if __name__ == "__main__":
    main()
