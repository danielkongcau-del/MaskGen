from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json
from partition_gen.joint_render import _draw_ring


Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared-boundary raster/segment targets from partition and dual graphs.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), default=str)


def copy_meta(dual_root: Path, output_root: Path) -> None:
    for filename in ["class_map.json", "dual_summary.json", "dual_stats.json", "ar_binners.json"]:
        source = dual_root / "meta" / filename
        if not source.exists():
            continue
        target = output_root / "meta" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def edge_key(face_ids: List[int]) -> Tuple[int, ...]:
    if len(face_ids) <= 1:
        return tuple(sorted(face_ids))
    a, b = sorted(face_ids[:2])
    return (a, b)


def process_split(
    *,
    split: str,
    partition_root: Path,
    dual_root: Path,
    output_root: Path,
    max_samples: int | None,
) -> Dict[str, object]:
    partition_dir = partition_root / split / "graphs"
    dual_dir = dual_root / split / "graphs"
    graph_paths = sorted(dual_dir.glob("*.json"))
    if max_samples is not None:
        graph_paths = graph_paths[:max_samples]

    total_segments = 0
    shared_segments = 0
    border_segments = 0
    boundary_pixels = []

    for index, dual_path in enumerate(graph_paths, start=1):
        dual_graph = load_json(dual_path)
        partition_path = partition_dir / dual_path.name
        partition_graph = load_json(partition_path)
        height, width = (int(value) for value in dual_graph["size"])
        size = (height, width)
        vertices: List[Point] = [tuple(int(value) for value in vertex) for vertex in partition_graph["vertices"]]
        orig_to_new = {int(face["orig_face_id"]): int(face["id"]) for face in dual_graph["faces"]}

        edge_groups: Dict[Tuple[int, ...], List[Dict[str, object]]] = {}
        boundary_mask = np.zeros(size, dtype=bool)

        for edge in partition_graph["edges"]:
            start_id, end_id = edge["vertices"]
            start = vertices[start_id]
            end = vertices[end_id]
            face_ids = [orig_to_new[int(face_id)] for face_id in edge.get("faces", []) if int(face_id) in orig_to_new]
            key = edge_key(face_ids)
            item = {
                "edge_id": int(edge["id"]),
                "faces": list(key),
                "vertices": [[int(start[0]), int(start[1])], [int(end[0]), int(end[1])]],
                "length": int(edge["length"]),
            }
            edge_groups.setdefault(key, []).append(item)
            _draw_ring(boundary_mask, [start, end])
            total_segments += 1
            if len(key) >= 2:
                shared_segments += 1
            else:
                border_segments += 1

        seeds = []
        for face in dual_graph["faces"]:
            seeds.append(
                {
                    "face_id": int(face["id"]),
                    "label": int(face["label"]),
                    "centroid": [float(face["centroid"][0]), float(face["centroid"][1])],
                    "centroid_ratio": [float(face["centroid_ratio"][0]), float(face["centroid_ratio"][1])],
                }
            )

        payload = {
            "source_partition_graph": str(partition_path.relative_to(partition_root).as_posix()),
            "source_dual_graph": str(dual_path.relative_to(dual_root).as_posix()),
            "source_mask": dual_graph.get("source_mask"),
            "size": [height, width],
            "seeds": seeds,
            "edge_groups": [
                {
                    "faces": list(key),
                    "segments": segments,
                    "num_segments": len(segments),
                    "shared_length": int(sum(int(segment["length"]) for segment in segments)),
                }
                for key, segments in sorted(edge_groups.items(), key=lambda item: (len(item[0]), item[0]))
            ],
            "stats": {
                "num_faces": int(len(dual_graph["faces"])),
                "num_edge_groups": int(len(edge_groups)),
                "boundary_pixels": int(boundary_mask.sum()),
            },
        }
        save_json(output_root / split / "graphs" / dual_path.name, payload)
        boundary_path = output_root / split / "boundary_masks" / f"{dual_path.stem}.png"
        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(boundary_mask.astype(np.uint8) * 255, mode="L").save(boundary_path)
        boundary_pixels.append(int(boundary_mask.sum()))

        if index % 100 == 0 or index == len(graph_paths):
            mean_boundary = float(np.mean(boundary_pixels)) if boundary_pixels else 0.0
            print(
                f"[{split}] processed {index}/{len(graph_paths)} boundary graphs "
                f"(mean_boundary_pixels={mean_boundary:.1f})"
            )

    return {
        "split": split,
        "num_samples": len(graph_paths),
        "mean_boundary_pixels": float(np.mean(boundary_pixels)) if boundary_pixels else 0.0,
        "max_boundary_pixels": int(max(boundary_pixels, default=0)),
        "total_segments": int(total_segments),
        "shared_segments": int(shared_segments),
        "border_segments": int(border_segments),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_meta(args.dual_root, args.output_root)

    summaries = []
    for split in args.splits:
        summaries.append(
            process_split(
                split=split,
                partition_root=args.partition_root,
                dual_root=args.dual_root,
                output_root=args.output_root,
                max_samples=args.max_samples,
            )
        )

    summary = {
        "partition_root": str(args.partition_root.as_posix()),
        "dual_root": str(args.dual_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "splits": summaries,
    }
    save_json(args.output_root / "meta" / "boundary_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
