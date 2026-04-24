from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simplified polygon geometry targets for dual-graph faces.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_geometry"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--simplify-tolerance", type=float, default=2.0)
    parser.add_argument("--max-holes", type=int, default=0, help="0 keeps hole faces unsupported.")
    parser.add_argument("--max-hole-vertices", type=int, default=0, help="Only used when max-holes > 0.")
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


def face_polygon(face_data: Dict[str, object], vertices: List[Tuple[int, int]]) -> Polygon:
    outer = [vertices[index] for index in face_data["outer"]]
    holes = [[vertices[index] for index in hole] for hole in face_data["holes"]]
    return Polygon(outer, holes)


def canonicalize_ring(coords: List[Point]) -> List[Point]:
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if not coords:
        return coords
    start_index = min(range(len(coords)), key=lambda index: (coords[index][1], coords[index][0], index))
    return coords[start_index:] + coords[:start_index]


def normalize_vertices(coords: List[Point], centroid: Tuple[float, float], bbox_width: float, bbox_height: float) -> List[List[float]]:
    scale_x = max(bbox_width / 2.0, 1.0)
    scale_y = max(bbox_height / 2.0, 1.0)
    normalized = []
    for x, y in coords:
        normalized.append([
            float((x - centroid[0]) / scale_x),
            float((y - centroid[1]) / scale_y),
        ])
    return normalized


def global_normalize_vertices(coords: List[Point], size: int) -> List[List[float]]:
    return [[float(x) / float(size), float(y) / float(size)] for x, y in coords]


def _sorted_hole_rings(polygon: Polygon) -> List[List[Point]]:
    holes = []
    for interior in polygon.interiors:
        coords = canonicalize_ring(list(interior.coords))
        hole_poly = Polygon(coords)
        holes.append((float(abs(hole_poly.area)), coords))
    holes.sort(key=lambda item: item[0], reverse=True)
    return [coords for _, coords in holes]


def build_face_geometry(
    face: Dict[str, object],
    partition_face: Dict[str, object],
    partition_vertices: List[Tuple[int, int]],
    size: int,
    simplify_tolerance: float,
    *,
    max_holes: int,
    max_hole_vertices: int,
) -> Dict[str, object]:
    polygon = face_polygon(partition_face, partition_vertices)
    polygon = orient(polygon, sign=1.0)
    simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
    original_hole_count = int(len(partition_face["holes"]))

    supported = True
    reason = None
    if simplified.is_empty:
        supported = False
        reason = "empty_after_simplify"
    elif simplified.geom_type != "Polygon":
        supported = False
        reason = f"geom_type_{simplified.geom_type}"
    elif max_holes <= 0 and len(simplified.interiors) > 0:
        supported = False
        reason = "holes"

    geometry: Dict[str, object] = {
        "supported": bool(supported),
        "reason": reason,
        "simplify_tolerance": float(simplify_tolerance),
        "original_hole_count": original_hole_count,
        "vertex_count": 0,
        "vertices_local": [],
        "vertices_global": [],
        "hole_count": 0,
        "hole_vertex_counts": [],
        "hole_vertices_local": [],
        "hole_vertices_global": [],
    }
    if not supported:
        return geometry

    coords = list(simplified.exterior.coords)
    coords = canonicalize_ring(coords)
    hole_rings = _sorted_hole_rings(simplified)
    if len(hole_rings) > 0:
        if max_holes <= 0:
            geometry["supported"] = False
            geometry["reason"] = "holes"
            return geometry
        if len(hole_rings) > max_holes:
            geometry["supported"] = False
            geometry["reason"] = "too_many_holes"
            return geometry

    bbox = face["bbox"]
    centroid = (float(face["centroid"][0]), float(face["centroid"][1]))
    bbox_width = float(face["bbox_width"])
    bbox_height = float(face["bbox_height"])

    geometry["vertex_count"] = int(len(coords))
    geometry["vertices_local"] = normalize_vertices(coords, centroid, bbox_width, bbox_height)
    geometry["vertices_global"] = global_normalize_vertices(coords, size=size)
    geometry["hole_count"] = int(len(hole_rings))

    for hole_coords in hole_rings:
        hole_vertex_count = int(len(hole_coords))
        if max_hole_vertices > 0 and hole_vertex_count > max_hole_vertices:
            geometry["supported"] = False
            geometry["reason"] = "hole_vertex_cap"
            geometry["hole_count"] = 0
            geometry["hole_vertex_counts"] = []
            geometry["hole_vertices_local"] = []
            geometry["hole_vertices_global"] = []
            return geometry
        geometry["hole_vertex_counts"].append(hole_vertex_count)
        geometry["hole_vertices_local"].append(normalize_vertices(hole_coords, centroid, bbox_width, bbox_height))
        geometry["hole_vertices_global"].append(global_normalize_vertices(hole_coords, size=size))
    return geometry


def process_split(
    split: str,
    partition_root: Path,
    dual_root: Path,
    output_root: Path,
    simplify_tolerance: float,
    max_samples: int | None,
    *,
    max_holes: int,
    max_hole_vertices: int,
) -> Dict[str, object]:
    partition_dir = partition_root / split / "graphs"
    dual_dir = dual_root / split / "graphs"
    output_dir = output_root / split / "graphs"
    graph_paths = sorted(dual_dir.glob("*.json"))
    if max_samples is not None:
        graph_paths = graph_paths[:max_samples]

    supported_faces = 0
    total_faces = 0
    supported_vertex_counts: List[int] = []
    supported_hole_face_count = 0
    supported_hole_counts: List[int] = []
    supported_hole_vertex_counts: List[int] = []
    unsupported_reasons: Dict[str, int] = {}

    for index, dual_path in enumerate(graph_paths, start=1):
        partition_path = partition_dir / dual_path.name
        dual_data = load_json(dual_path)
        partition_data = load_json(partition_path)
        size = int(dual_data["size"][0])
        partition_vertices = [tuple(vertex) for vertex in partition_data["vertices"]]
        partition_faces = {int(face["id"]): face for face in partition_data["faces"]}

        face_geometries = []
        for face in dual_data["faces"]:
            partition_face = partition_faces[int(face["orig_face_id"])]
            geometry = build_face_geometry(
                face=face,
                partition_face=partition_face,
                partition_vertices=partition_vertices,
                size=size,
                simplify_tolerance=simplify_tolerance,
                max_holes=max_holes,
                max_hole_vertices=max_hole_vertices,
            )
            face_geometries.append(
                {
                    "id": int(face["id"]),
                    "orig_face_id": int(face["orig_face_id"]),
                    **geometry,
                }
            )
            total_faces += 1
            if geometry["supported"]:
                supported_faces += 1
                supported_vertex_counts.append(int(geometry["vertex_count"]))
                if int(geometry["hole_count"]) > 0:
                    supported_hole_face_count += 1
                    supported_hole_counts.append(int(geometry["hole_count"]))
                    supported_hole_vertex_counts.extend(int(count) for count in geometry["hole_vertex_counts"])
            else:
                key = str(geometry["reason"])
                unsupported_reasons[key] = unsupported_reasons.get(key, 0) + 1

        payload = {
            "source_dual_graph": str(dual_path.relative_to(dual_root).as_posix()),
            "source_partition_graph": str(partition_path.relative_to(partition_root).as_posix()),
            "size": dual_data["size"],
            "simplify_tolerance": float(simplify_tolerance),
            "faces": face_geometries,
        }
        save_json(output_dir / dual_path.name, payload)

        if index % 100 == 0 or index == len(graph_paths):
            ratio = supported_faces / max(total_faces, 1)
            mean_vertices = float(np.mean(supported_vertex_counts)) if supported_vertex_counts else 0.0
            print(
                f"[{split}] processed {index}/{len(graph_paths)} geometry graphs "
                f"(support={ratio:.3f}, mean_vertices={mean_vertices:.1f})"
            )

    summary = {
        "split": split,
        "num_samples": len(graph_paths),
        "total_faces": int(total_faces),
        "supported_faces": int(supported_faces),
        "supported_ratio": float(supported_faces / max(total_faces, 1)),
        "mean_supported_vertices": float(np.mean(supported_vertex_counts)) if supported_vertex_counts else 0.0,
        "max_supported_vertices": int(max(supported_vertex_counts, default=0)),
        "supported_hole_faces": int(supported_hole_face_count),
        "mean_supported_holes_per_face": float(np.mean(supported_hole_counts)) if supported_hole_counts else 0.0,
        "mean_supported_hole_vertices": float(np.mean(supported_hole_vertex_counts)) if supported_hole_vertex_counts else 0.0,
        "max_supported_hole_vertices": int(max(supported_hole_vertex_counts, default=0)),
        "unsupported_reasons": unsupported_reasons,
    }
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_meta(args.dual_root, args.output_root)

    split_summaries = []
    for split in args.splits:
        split_summaries.append(
            process_split(
                split=split,
                partition_root=args.partition_root,
                dual_root=args.dual_root,
                output_root=args.output_root,
                simplify_tolerance=args.simplify_tolerance,
                max_samples=args.max_samples,
                max_holes=args.max_holes,
                max_hole_vertices=args.max_hole_vertices,
            )
        )

    summary = {
        "partition_root": str(args.partition_root.as_posix()),
        "dual_root": str(args.dual_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "simplify_tolerance": float(args.simplify_tolerance),
        "max_holes": int(args.max_holes),
        "max_hole_vertices": int(args.max_hole_vertices),
        "splits": split_summaries,
    }
    save_json(args.output_root / "meta" / "geometry_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
