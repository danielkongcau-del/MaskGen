from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from shapely import contains_xy
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize


Point = Tuple[int, int]
Segment = Tuple[Point, Point]


@dataclass(frozen=True)
class GraphEdge:
    edge_id: int
    vertices: Tuple[int, int]
    length: int
    faces: Tuple[int, ...]


@dataclass(frozen=True)
class FaceRecord:
    face_id: int
    label: int
    area: int
    bbox: Tuple[int, int, int, int]
    outer: Tuple[int, ...]
    holes: Tuple[Tuple[int, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert label masks into planar partition graphs and verify round-trip rasterization."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/remote_256"),
        help="Dataset root containing split/masks_id directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/remote_256_partition"),
        help="Output root for extracted graph JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for debugging.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rasterize extracted faces and require an exact match with the source mask.",
    )
    parser.add_argument(
        "--save-rendered",
        action="store_true",
        help="Save reconstructed masks alongside JSON files for debugging.",
    )
    return parser.parse_args()


def canonical_segment(a: Point, b: Point) -> Segment:
    return (a, b) if a <= b else (b, a)


def segment_length(segment: Segment) -> int:
    (x0, y0), (x1, y1) = segment
    return abs(x1 - x0) + abs(y1 - y0)


def is_collinear(prev_point: Point, point: Point, next_point: Point) -> bool:
    return (point[0] - prev_point[0]) * (next_point[1] - point[1]) == (
        point[1] - prev_point[1]
    ) * (next_point[0] - point[0])


def trim_ring(points: Sequence[Tuple[float, float]]) -> List[Point]:
    ring = [tuple(int(round(value)) for value in point) for point in points]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    changed = True
    while changed and len(ring) >= 3:
        changed = False
        kept: List[Point] = []
        for index, point in enumerate(ring):
            prev_point = ring[index - 1]
            next_point = ring[(index + 1) % len(ring)]
            if is_collinear(prev_point, point, next_point):
                changed = True
                continue
            kept.append(point)
        ring = kept
    return ring


def ring_points(points: Sequence[Tuple[float, float]]) -> List[Point]:
    ring = [tuple(int(round(value)) for value in point) for point in points]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def extract_unit_segments(mask: np.ndarray) -> List[Segment]:
    height, width = mask.shape
    segments: set[Segment] = set()
    for y in range(height):
        for x in range(width):
            if y == 0:
                segments.add(canonical_segment((x, y), (x + 1, y)))
            if y == height - 1:
                segments.add(canonical_segment((x, y + 1), (x + 1, y + 1)))
            if x == 0:
                segments.add(canonical_segment((x, y), (x, y + 1)))
            if x == width - 1:
                segments.add(canonical_segment((x + 1, y), (x + 1, y + 1)))
            if x + 1 < width and int(mask[y, x]) != int(mask[y, x + 1]):
                segments.add(canonical_segment((x + 1, y), (x + 1, y + 1)))
            if y + 1 < height and int(mask[y, x]) != int(mask[y + 1, x]):
                segments.add(canonical_segment((x, y + 1), (x + 1, y + 1)))
    return sorted(segments)


def build_adjacency(segments: Iterable[Segment]) -> Dict[Point, set[Point]]:
    adjacency: Dict[Point, set[Point]] = defaultdict(set)
    for point_a, point_b in segments:
        adjacency[point_a].add(point_b)
        adjacency[point_b].add(point_a)
    return adjacency


def is_straight_pass_through(vertex: Point, adjacency: Dict[Point, set[Point]]) -> bool:
    neighbors = list(adjacency[vertex])
    if len(neighbors) != 2:
        return False
    first, second = neighbors
    return (first[0] == vertex[0] == second[0]) or (first[1] == vertex[1] == second[1])


def compress_segments(unit_segments: Sequence[Segment]) -> List[Segment]:
    adjacency = build_adjacency(unit_segments)
    anchors = {vertex for vertex in adjacency if not is_straight_pass_through(vertex, adjacency)}
    visited: set[Segment] = set()
    coarse_segments: List[Segment] = []

    for start in sorted(anchors):
        for next_vertex in sorted(adjacency[start]):
            unit_key = canonical_segment(start, next_vertex)
            if unit_key in visited:
                continue
            visited.add(unit_key)
            prev_vertex = start
            current_vertex = next_vertex

            while current_vertex not in anchors:
                candidates = [candidate for candidate in adjacency[current_vertex] if candidate != prev_vertex]
                if not candidates:
                    break
                next_step = candidates[0]
                visited.add(canonical_segment(current_vertex, next_step))
                prev_vertex, current_vertex = current_vertex, next_step

            coarse_segments.append(canonical_segment(start, current_vertex))

    return sorted(set(coarse_segments))


def build_vertices_and_edges(segments: Sequence[Segment]) -> Tuple[List[Point], Dict[Segment, int], List[GraphEdge]]:
    vertex_to_id: Dict[Point, int] = {}
    vertices: List[Point] = []

    def get_vertex_id(point: Point) -> int:
        if point not in vertex_to_id:
            vertex_to_id[point] = len(vertices)
            vertices.append(point)
        return vertex_to_id[point]

    segment_to_edge_id: Dict[Segment, int] = {}
    edges: List[GraphEdge] = []
    for segment in segments:
        point_a, point_b = segment
        vertex_pair = (get_vertex_id(point_a), get_vertex_id(point_b))
        edge_id = len(edges)
        segment_to_edge_id[segment] = edge_id
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                vertices=vertex_pair,
                length=segment_length(segment),
                faces=tuple(),
            )
        )

    return vertices, segment_to_edge_id, edges


def build_vertex_line_index(vertices: Sequence[Point]) -> Tuple[Dict[int, List[Point]], Dict[int, List[Point]]]:
    vertical: Dict[int, List[Point]] = defaultdict(list)
    horizontal: Dict[int, List[Point]] = defaultdict(list)
    for point in vertices:
        vertical[point[0]].append(point)
        horizontal[point[1]].append(point)
    for points in vertical.values():
        points.sort(key=lambda item: item[1])
    for points in horizontal.values():
        points.sort(key=lambda item: item[0])
    return vertical, horizontal


def expand_segment_vertices(
    point_a: Point,
    point_b: Point,
    vertical_index: Dict[int, List[Point]],
    horizontal_index: Dict[int, List[Point]],
) -> List[Point]:
    if point_a[0] == point_b[0]:
        candidates = [
            point
            for point in vertical_index[point_a[0]]
            if min(point_a[1], point_b[1]) <= point[1] <= max(point_a[1], point_b[1])
        ]
        candidates.sort(key=lambda item: item[1], reverse=point_a[1] > point_b[1])
        return candidates
    if point_a[1] == point_b[1]:
        candidates = [
            point
            for point in horizontal_index[point_a[1]]
            if min(point_a[0], point_b[0]) <= point[0] <= max(point_a[0], point_b[0])
        ]
        candidates.sort(key=lambda item: item[0], reverse=point_a[0] > point_b[0])
        return candidates
    raise ValueError(f"Non axis-aligned edge detected: {point_a} -> {point_b}")


def expand_ring_vertices(
    ring: Sequence[Point],
    vertical_index: Dict[int, List[Point]],
    horizontal_index: Dict[int, List[Point]],
) -> List[Point]:
    expanded: List[Point] = []
    for index, point_a in enumerate(ring):
        point_b = ring[(index + 1) % len(ring)]
        segment_vertices = expand_segment_vertices(point_a, point_b, vertical_index, horizontal_index)
        if not segment_vertices:
            raise ValueError(f"Unable to expand ring segment: {point_a} -> {point_b}")
        if not expanded:
            expanded.extend(segment_vertices)
        else:
            expanded.extend(segment_vertices[1:])
    if expanded and expanded[0] == expanded[-1]:
        expanded = expanded[:-1]
    return expanded


def polygon_to_vertex_rings(
    polygon: Polygon,
    vertex_to_id: Dict[Point, int],
    vertical_index: Dict[int, List[Point]],
    horizontal_index: Dict[int, List[Point]],
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    polygon = orient(polygon, sign=1.0)
    outer_coords = expand_ring_vertices(ring_points(polygon.exterior.coords), vertical_index, horizontal_index)
    outer_ring = tuple(vertex_to_id[point] for point in outer_coords)

    holes: List[Tuple[int, ...]] = []
    for interior in polygon.interiors:
        hole_coords = expand_ring_vertices(ring_points(interior.coords), vertical_index, horizontal_index)
        holes.append(tuple(vertex_to_id[point] for point in hole_coords))

    return outer_ring, tuple(holes)


def face_geometry_from_rings(face: FaceRecord, vertices: Sequence[Point]) -> Polygon:
    outer = [vertices[vertex_id] for vertex_id in face.outer]
    holes = [[vertices[vertex_id] for vertex_id in hole] for hole in face.holes]
    return Polygon(outer, holes)


def extract_faces(mask: np.ndarray, vertices: Sequence[Point], segment_to_edge_id: Dict[Segment, int]) -> Tuple[List[FaceRecord], List[GraphEdge], List[Dict[str, int]]]:
    coarse_segments = [segment for segment in segment_to_edge_id]
    polygons = list(polygonize([LineString(segment) for segment in coarse_segments]))
    vertex_to_id = {point: index for index, point in enumerate(vertices)}
    vertical_index, horizontal_index = build_vertex_line_index(vertices)
    edge_faces: Dict[int, set[int]] = defaultdict(set)
    faces: List[FaceRecord] = []

    for polygon in polygons:
        point = polygon.representative_point()
        sample_x = min(mask.shape[1] - 1, max(0, int(math.floor(point.x))))
        sample_y = min(mask.shape[0] - 1, max(0, int(math.floor(point.y))))
        label = int(mask[sample_y, sample_x])
        outer_ring, hole_rings = polygon_to_vertex_rings(
            polygon,
            vertex_to_id,
            vertical_index,
            horizontal_index,
        )
        face_id = len(faces)

        for ring in (outer_ring, *hole_rings):
            for index, current_vertex in enumerate(ring):
                next_vertex = ring[(index + 1) % len(ring)]
                segment = canonical_segment(vertices[current_vertex], vertices[next_vertex])
                edge_faces[segment_to_edge_id[segment]].add(face_id)

        bounds = tuple(int(round(value)) for value in polygon.bounds)
        faces.append(
            FaceRecord(
                face_id=face_id,
                label=label,
                area=int(round(polygon.area)),
                bbox=bounds,
                outer=outer_ring,
                holes=hole_rings,
            )
        )

    edges: List[GraphEdge] = []
    for segment, edge_id in sorted(segment_to_edge_id.items(), key=lambda item: item[1]):
        faces_for_edge = tuple(sorted(edge_faces.get(edge_id, set())))
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                vertices=(
                    vertex_to_id[segment[0]],
                    vertex_to_id[segment[1]],
                ),
                length=segment_length(segment),
                faces=faces_for_edge,
            )
        )

    adjacency_totals: Dict[Tuple[int, int], int] = defaultdict(int)
    for edge in edges:
        if len(edge.faces) == 2:
            adjacency_totals[(edge.faces[0], edge.faces[1])] += edge.length

    adjacency = [
        {"faces": [face_a, face_b], "shared_length": shared_length}
        for (face_a, face_b), shared_length in sorted(adjacency_totals.items())
    ]

    return faces, edges, adjacency


def rasterize_faces(faces: Sequence[FaceRecord], vertices: Sequence[Point], shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    rendered = np.full((height, width), -1, dtype=np.int16)

    for face in sorted(faces, key=lambda item: item.area, reverse=True):
        geometry = face_geometry_from_rings(face, vertices)
        min_x, min_y, max_x, max_y = face.bbox
        xs = np.arange(min_x, max_x) + 0.5
        ys = np.arange(min_y, max_y) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = contains_xy(geometry, grid_x, grid_y)
        rendered[min_y:max_y, min_x:max_x][inside] = face.label

    return rendered


def serialize_partition(mask_path: Path, mask: np.ndarray) -> Dict[str, object]:
    unit_segments = extract_unit_segments(mask)
    coarse_segments = compress_segments(unit_segments)
    vertices, segment_to_edge_id, _ = build_vertices_and_edges(coarse_segments)
    faces, edges, adjacency = extract_faces(mask, vertices, segment_to_edge_id)

    return {
        "source_mask": str(mask_path.as_posix()),
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "vertices": [[int(x), int(y)] for x, y in vertices],
        "edges": [
            {
                "id": edge.edge_id,
                "vertices": [int(edge.vertices[0]), int(edge.vertices[1])],
                "length": int(edge.length),
                "faces": [int(face_id) for face_id in edge.faces],
            }
            for edge in edges
        ],
        "faces": [
            {
                "id": face.face_id,
                "label": int(face.label),
                "area": int(face.area),
                "bbox": [int(value) for value in face.bbox],
                "outer": [int(vertex_id) for vertex_id in face.outer],
                "holes": [[int(vertex_id) for vertex_id in hole] for hole in face.holes],
            }
            for face in faces
        ],
        "adjacency": adjacency,
        "stats": {
            "num_unit_segments": int(len(unit_segments)),
            "num_edges": int(len(edges)),
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
        },
    }


def deserialize_faces(graph_data: Dict[str, object]) -> Tuple[List[Point], List[FaceRecord], Tuple[int, int]]:
    vertices = [tuple(vertex) for vertex in graph_data["vertices"]]
    size = tuple(int(value) for value in graph_data["size"])
    faces = [
        FaceRecord(
            face_id=int(face["id"]),
            label=int(face["label"]),
            area=int(face["area"]),
            bbox=tuple(int(value) for value in face["bbox"]),
            outer=tuple(int(vertex_id) for vertex_id in face["outer"]),
            holes=tuple(tuple(int(vertex_id) for vertex_id in hole) for hole in face["holes"]),
        )
        for face in graph_data["faces"]
    ]
    return vertices, faces, size


def verify_round_trip(graph_data: Dict[str, object], original_mask: np.ndarray) -> None:
    vertices, faces, shape = deserialize_faces(graph_data)
    rendered = rasterize_faces(faces, vertices, shape)
    if np.any(rendered < 0):
        raise ValueError("Round-trip rasterization left unfilled pixels.")
    if not np.array_equal(rendered.astype(original_mask.dtype), original_mask):
        raise ValueError("Round-trip rasterization does not match the source mask exactly.")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def copy_class_map_if_available(input_root: Path, output_root: Path) -> None:
    source = input_root / "meta" / "class_map.json"
    if not source.exists():
        return
    target = output_root / "meta" / "class_map.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def process_split(
    split: str,
    input_root: Path,
    output_root: Path,
    max_samples: int | None,
    verify: bool,
    save_rendered: bool,
) -> Dict[str, object]:
    mask_dir = input_root / split / "masks_id"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing masks directory: {mask_dir}")

    output_graph_dir = output_root / split / "graphs"
    output_render_dir = output_root / split / "renders"
    graph_paths = sorted(mask_dir.glob("*.png"))
    if max_samples is not None:
        graph_paths = graph_paths[:max_samples]

    split_vertices: List[int] = []
    split_edges: List[int] = []
    split_faces: List[int] = []

    for index, mask_path in enumerate(graph_paths, start=1):
        mask = np.array(Image.open(mask_path))
        graph_data = serialize_partition(mask_path.relative_to(input_root), mask)
        if verify:
            verify_round_trip(graph_data, mask)
        write_json(output_graph_dir / f"{mask_path.stem}.json", graph_data)

        if save_rendered:
            vertices, faces, shape = deserialize_faces(graph_data)
            rendered = rasterize_faces(faces, vertices, shape).astype(np.uint8)
            output_render_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered, mode="L").save(output_render_dir / f"{mask_path.stem}.png")

        split_vertices.append(graph_data["stats"]["num_vertices"])
        split_edges.append(graph_data["stats"]["num_edges"])
        split_faces.append(graph_data["stats"]["num_faces"])

        if index % 100 == 0 or index == len(graph_paths):
            print(
                f"[{split}] processed {index}/{len(graph_paths)} masks "
                f"(mean faces={np.mean(split_faces):.1f}, mean edges={np.mean(split_edges):.1f})"
            )

    return {
        "split": split,
        "num_samples": len(graph_paths),
        "mean_vertices": float(np.mean(split_vertices)) if split_vertices else 0.0,
        "mean_edges": float(np.mean(split_edges)) if split_edges else 0.0,
        "mean_faces": float(np.mean(split_faces)) if split_faces else 0.0,
        "max_vertices": int(max(split_vertices)) if split_vertices else 0,
        "max_edges": int(max(split_edges)) if split_edges else 0,
        "max_faces": int(max(split_faces)) if split_faces else 0,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_class_map_if_available(args.input_root, args.output_root)

    split_summaries = []
    for split in args.splits:
        summary = process_split(
            split=split,
            input_root=args.input_root,
            output_root=args.output_root,
            max_samples=args.max_samples,
            verify=args.verify,
            save_rendered=args.save_rendered,
        )
        split_summaries.append(summary)

    summary_payload = {
        "input_root": str(args.input_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "verify": bool(args.verify),
        "save_rendered": bool(args.save_rendered),
        "splits": split_summaries,
    }
    write_json(args.output_root / "meta" / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
