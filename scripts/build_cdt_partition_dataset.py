from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.cdt_partition import CdtSimplifyConfig, face_geometry, polygon_payload, simplify_face_polygon
from partition_gen.dual_graph import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CDT-friendly simplified partition dataset.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_partition_cdt"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerances", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--min-iou", type=float, default=0.995)
    return parser.parse_args()


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), indent=2)


def process_graph(path: Path, *, config: CdtSimplifyConfig) -> Dict[str, object]:
    graph_data = load_json(path)
    faces: List[Dict[str, object]] = []
    total_original_vertices = 0
    total_simplified_vertices = 0
    ious: List[float] = []

    for face_data in graph_data["faces"]:
        polygon = face_geometry(graph_data, face_data)
        simplified = simplify_face_polygon(polygon, config=config)
        total_original_vertices += int(simplified["original_vertex_count"])
        total_simplified_vertices += int(simplified["simplified_vertex_count"])
        ious.append(float(simplified["iou"]))
        face_payload = {
            "id": int(face_data["id"]),
            "label": int(face_data["label"]),
            "area": int(face_data["area"]),
            "bbox": [int(value) for value in face_data["bbox"]],
            "original_hole_count": int(len(face_data["holes"])),
            "original_vertex_count": int(simplified["original_vertex_count"]),
            "simplified_vertex_count": int(simplified["simplified_vertex_count"]),
            "simplified_hole_count": int(simplified["hole_count"]),
            "simplify_iou": float(simplified["iou"]),
            "selected_tolerance": float(simplified["tolerance"]),
        }
        face_payload.update(polygon_payload(simplified["polygon"], eps=config.trim_collinear_eps))
        faces.append(face_payload)

    return {
        "format": "cdt_partition_v1",
        "source_partition_graph": str(path.as_posix()),
        "source_mask": graph_data.get("source_mask"),
        "size": graph_data["size"],
        "config": {
            "tolerances": [float(value) for value in config.tolerances],
            "min_iou": float(config.min_iou),
            "trim_collinear_eps": float(config.trim_collinear_eps),
        },
        "faces": faces,
        "stats": {
            "num_faces": int(len(faces)),
            "total_original_vertices": int(total_original_vertices),
            "total_simplified_vertices": int(total_simplified_vertices),
            "vertex_reduction_ratio": float(total_simplified_vertices / max(total_original_vertices, 1)),
            "mean_face_iou": float(sum(ious) / max(len(ious), 1)),
        },
    }


def main() -> None:
    args = parse_args()
    config = CdtSimplifyConfig(tolerances=tuple(float(value) for value in args.tolerances), min_iou=float(args.min_iou))

    for split in args.splits:
        graph_paths = sorted((args.partition_root / split / "graphs").glob("*.json"))
        if args.stems:
            stem_set = {str(value) for value in args.stems}
            graph_paths = [path for path in graph_paths if path.stem in stem_set]
        if args.max_samples is not None:
            graph_paths = graph_paths[: args.max_samples]
        split_rows = []
        for index, graph_path in enumerate(graph_paths, start=1):
            payload = process_graph(graph_path, config=config)
            dump_json(args.output_root / split / "graphs" / graph_path.name, payload)
            split_rows.append(payload["stats"])
            if index % 50 == 0 or index == len(graph_paths):
                mean_ratio = sum(row["vertex_reduction_ratio"] for row in split_rows) / max(len(split_rows), 1)
                mean_iou = sum(row["mean_face_iou"] for row in split_rows) / max(len(split_rows), 1)
                print(f"[{split}] {index}/{len(graph_paths)} vertex_ratio={mean_ratio:.3f} mean_iou={mean_iou:.4f}")

        if split_rows:
            dump_json(
                args.output_root / "meta" / f"{split}_summary.json",
                {
                    "split": split,
                    "num_graphs": int(len(split_rows)),
                    "mean_vertex_reduction_ratio": float(
                        sum(row["vertex_reduction_ratio"] for row in split_rows) / len(split_rows)
                    ),
                    "mean_face_iou": float(sum(row["mean_face_iou"] for row in split_rows) / len(split_rows)),
                },
            )


if __name__ == "__main__":
    main()
