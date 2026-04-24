from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import build_dual_graph_payload, dump_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build face-level dual graphs from partition-graph JSON files.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/remote_256_partition"),
        help="Root containing split/graphs JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/remote_256_dual"),
        help="Output root for dual-graph JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for debugging.",
    )
    return parser.parse_args()


def copy_meta(input_root: Path, output_root: Path) -> None:
    for filename in ["class_map.json", "summary.json"]:
        source = input_root / "meta" / filename
        if not source.exists():
            continue
        target = output_root / "meta" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def process_split(split: str, input_root: Path, output_root: Path, max_samples: int | None) -> Dict[str, object]:
    input_graph_dir = input_root / split / "graphs"
    output_graph_dir = output_root / split / "graphs"
    graph_paths = sorted(input_graph_dir.glob("*.json"))
    if max_samples is not None:
        graph_paths = graph_paths[:max_samples]

    num_faces: List[int] = []
    num_edges: List[int] = []
    max_prev_neighbors: List[int] = []

    for index, graph_path in enumerate(graph_paths, start=1):
        graph_data = load_json(graph_path)
        payload = build_dual_graph_payload(
            graph_data,
            source_path=str(graph_path.relative_to(input_root).as_posix()),
        )
        dump_json(output_graph_dir / graph_path.name, payload)
        num_faces.append(int(payload["stats"]["num_faces"]))
        num_edges.append(int(payload["stats"]["num_edges"]))
        max_prev_neighbors.append(int(payload["stats"]["max_prev_neighbors"]))

        if index % 100 == 0 or index == len(graph_paths):
            mean_faces = sum(num_faces) / max(1, len(num_faces))
            mean_edges = sum(num_edges) / max(1, len(num_edges))
            print(
                f"[{split}] processed {index}/{len(graph_paths)} dual graphs "
                f"(mean faces={mean_faces:.1f}, mean edges={mean_edges:.1f})"
            )

    return {
        "split": split,
        "num_samples": len(graph_paths),
        "mean_faces": float(sum(num_faces) / max(1, len(num_faces))),
        "mean_edges": float(sum(num_edges) / max(1, len(num_edges))),
        "mean_max_prev_neighbors": float(sum(max_prev_neighbors) / max(1, len(max_prev_neighbors))),
        "max_faces": int(max(num_faces, default=0)),
        "max_edges": int(max(num_edges, default=0)),
        "max_prev_neighbors": int(max(max_prev_neighbors, default=0)),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_meta(args.input_root, args.output_root)

    split_summaries = []
    for split in args.splits:
        split_summaries.append(
            process_split(
                split=split,
                input_root=args.input_root,
                output_root=args.output_root,
                max_samples=args.max_samples,
            )
        )

    summary = {
        "input_root": str(args.input_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "splits": split_summaries,
    }
    dump_json(args.output_root / "meta" / "dual_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
