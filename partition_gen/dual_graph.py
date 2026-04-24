from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import Polygon


Point = Tuple[int, int]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vertices(graph_data: Dict[str, object]) -> List[Point]:
    return [tuple(int(value) for value in vertex) for vertex in graph_data["vertices"]]


def ring_coords(vertex_ids: Sequence[int], vertices: Sequence[Point]) -> List[Point]:
    return [vertices[vertex_id] for vertex_id in vertex_ids]


def face_polygon(face_data: Dict[str, object], vertices: Sequence[Point]) -> Polygon:
    outer = ring_coords(face_data["outer"], vertices)
    holes = [ring_coords(hole, vertices) for hole in face_data["holes"]]
    return Polygon(outer, holes)


def ring_perimeter(coords: Sequence[Point]) -> int:
    perimeter = 0
    for index, point_a in enumerate(coords):
        point_b = coords[(index + 1) % len(coords)]
        perimeter += abs(point_b[0] - point_a[0]) + abs(point_b[1] - point_a[1])
    return perimeter


def border_overlap(point_a: Point, point_b: Point, width: int, height: int) -> int:
    if point_a[0] == point_b[0]:
        x = point_a[0]
        if x not in (0, width):
            return 0
        return abs(point_b[1] - point_a[1])
    if point_a[1] == point_b[1]:
        y = point_a[1]
        if y not in (0, height):
            return 0
        return abs(point_b[0] - point_a[0])
    return 0


def face_border_length(face_data: Dict[str, object], vertices: Sequence[Point], width: int, height: int) -> int:
    border_length = 0
    rings = [face_data["outer"], *face_data["holes"]]
    for ring in rings:
        coords = ring_coords(ring, vertices)
        for index, point_a in enumerate(coords):
            point_b = coords[(index + 1) % len(coords)]
            border_length += border_overlap(point_a, point_b, width=width, height=height)
    return border_length


def build_adjacency_map(adjacency_items: Iterable[Dict[str, int]]) -> Dict[int, Dict[int, int]]:
    adjacency_map: Dict[int, Dict[int, int]] = defaultdict(dict)
    for item in adjacency_items:
        face_a, face_b = (int(value) for value in item["faces"])
        shared_length = int(item["shared_length"])
        adjacency_map[face_a][face_b] = shared_length
        adjacency_map[face_b][face_a] = shared_length
    return adjacency_map


def build_face_records(graph_data: Dict[str, object]) -> List[Dict[str, object]]:
    height, width = (int(value) for value in graph_data["size"])
    vertices = load_vertices(graph_data)
    records: List[Dict[str, object]] = []
    for face_data in graph_data["faces"]:
        polygon = face_polygon(face_data, vertices)
        outer_coords = ring_coords(face_data["outer"], vertices)
        hole_coords = [ring_coords(hole, vertices) for hole in face_data["holes"]]
        bbox = [int(value) for value in face_data["bbox"]]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        perimeter = ring_perimeter(outer_coords) + sum(ring_perimeter(hole) for hole in hole_coords)
        border_length = face_border_length(face_data, vertices, width=width, height=height)
        centroid = [float(polygon.centroid.x), float(polygon.centroid.y)]
        records.append(
            {
                "orig_face_id": int(face_data["id"]),
                "label": int(face_data["label"]),
                "area": int(face_data["area"]),
                "area_ratio": float(face_data["area"]) / float(width * height),
                "bbox": bbox,
                "bbox_ratio": [
                    bbox[0] / float(width),
                    bbox[1] / float(height),
                    bbox[2] / float(width),
                    bbox[3] / float(height),
                ],
                "bbox_width": int(bbox_width),
                "bbox_height": int(bbox_height),
                "bbox_width_ratio": float(bbox_width) / float(width),
                "bbox_height_ratio": float(bbox_height) / float(height),
                "centroid": centroid,
                "centroid_ratio": [centroid[0] / float(width), centroid[1] / float(height)],
                "perimeter": int(perimeter),
                "perimeter_ratio": float(perimeter) / float(2 * (width + height)),
                "border_length": int(border_length),
                "border_ratio": float(border_length) / float(2 * (width + height)),
                "touches_border": bool(border_length > 0),
                "outer_vertices": int(len(face_data["outer"])),
                "hole_count": int(len(face_data["holes"])),
                "hole_vertices": int(sum(len(hole) for hole in face_data["holes"])),
                "total_vertices": int(len(face_data["outer"]) + sum(len(hole) for hole in face_data["holes"])),
            }
        )
    return records


def root_sort_key(face: Dict[str, object]) -> Tuple[float, ...]:
    return (
        -int(bool(face["touches_border"])),
        -float(face["area"]),
        float(face["centroid"][1]),
        float(face["centroid"][0]),
        int(face["label"]),
        int(face["orig_face_id"]),
    )


def neighbor_sort_key(
    neighbor_face: Dict[str, object],
    shared_length: int,
) -> Tuple[float, ...]:
    return (
        -int(shared_length),
        -int(bool(neighbor_face["touches_border"])),
        -float(neighbor_face["area"]),
        float(neighbor_face["centroid"][1]),
        float(neighbor_face["centroid"][0]),
        int(neighbor_face["label"]),
        int(neighbor_face["orig_face_id"]),
    )


def canonical_face_order(
    face_records: Sequence[Dict[str, object]],
    adjacency_map: Dict[int, Dict[int, int]],
) -> List[int]:
    faces_by_id = {int(face["orig_face_id"]): face for face in face_records}
    remaining = set(faces_by_id.keys())
    order: List[int] = []

    while remaining:
        root = sorted((faces_by_id[face_id] for face_id in remaining), key=root_sort_key)[0]
        root_id = int(root["orig_face_id"])
        queue: deque[int] = deque([root_id])
        remaining.remove(root_id)

        while queue:
            current = queue.popleft()
            order.append(current)
            neighbor_ids = [
                neighbor_id for neighbor_id in adjacency_map.get(current, {}) if neighbor_id in remaining
            ]
            neighbor_ids.sort(
                key=lambda neighbor_id: neighbor_sort_key(
                    faces_by_id[neighbor_id],
                    adjacency_map[current][neighbor_id],
                )
            )
            for neighbor_id in neighbor_ids:
                if neighbor_id not in remaining:
                    continue
                remaining.remove(neighbor_id)
                queue.append(neighbor_id)

    return order


def build_dual_graph_payload(
    graph_data: Dict[str, object],
    source_path: str | None = None,
) -> Dict[str, object]:
    height, width = (int(value) for value in graph_data["size"])
    face_records = build_face_records(graph_data)
    adjacency_map = build_adjacency_map(graph_data["adjacency"])
    order = canonical_face_order(face_records, adjacency_map)
    faces_by_old_id = {int(face["orig_face_id"]): face for face in face_records}
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(order)}

    dual_edges: List[Dict[str, int]] = []
    for old_face_a, neighbors in adjacency_map.items():
        for old_face_b, shared_length in neighbors.items():
            if old_face_a >= old_face_b:
                continue
            new_face_a = int(old_to_new[old_face_a])
            new_face_b = int(old_to_new[old_face_b])
            if new_face_a > new_face_b:
                new_face_a, new_face_b = new_face_b, new_face_a
            dual_edges.append(
                {
                    "u": new_face_a,
                    "v": new_face_b,
                    "shared_length": int(shared_length),
                }
            )
    dual_edges.sort(key=lambda item: (item["u"], item["v"]))

    dual_faces: List[Dict[str, object]] = []
    prev_neighbor_lengths: List[int] = []
    max_prev_neighbors = 0

    for new_face_id, old_face_id in enumerate(order):
        face = dict(faces_by_old_id[old_face_id])
        neighbor_items = []
        for neighbor_old_id, shared_length in adjacency_map.get(old_face_id, {}).items():
            neighbor_items.append((old_to_new[neighbor_old_id], int(shared_length)))
        neighbor_items.sort(key=lambda item: (item[0], item[1]))

        previous_neighbors = [
            {"id": int(neighbor_id), "shared_length": int(shared_length)}
            for neighbor_id, shared_length in neighbor_items
            if neighbor_id < new_face_id
        ]
        next_neighbors = [
            {"id": int(neighbor_id), "shared_length": int(shared_length)}
            for neighbor_id, shared_length in neighbor_items
            if neighbor_id > new_face_id
        ]
        prev_neighbor_lengths.extend(item["shared_length"] for item in previous_neighbors)
        max_prev_neighbors = max(max_prev_neighbors, len(previous_neighbors))

        dual_faces.append(
            {
                "id": int(new_face_id),
                "orig_face_id": int(old_face_id),
                "label": int(face["label"]),
                "area": int(face["area"]),
                "area_ratio": float(face["area_ratio"]),
                "centroid": [float(face["centroid"][0]), float(face["centroid"][1])],
                "centroid_ratio": [float(face["centroid_ratio"][0]), float(face["centroid_ratio"][1])],
                "bbox": [int(value) for value in face["bbox"]],
                "bbox_ratio": [float(value) for value in face["bbox_ratio"]],
                "bbox_width": int(face["bbox_width"]),
                "bbox_height": int(face["bbox_height"]),
                "bbox_width_ratio": float(face["bbox_width_ratio"]),
                "bbox_height_ratio": float(face["bbox_height_ratio"]),
                "perimeter": int(face["perimeter"]),
                "perimeter_ratio": float(face["perimeter_ratio"]),
                "border_length": int(face["border_length"]),
                "border_ratio": float(face["border_ratio"]),
                "touches_border": bool(face["touches_border"]),
                "outer_vertices": int(face["outer_vertices"]),
                "hole_count": int(face["hole_count"]),
                "hole_vertices": int(face["hole_vertices"]),
                "total_vertices": int(face["total_vertices"]),
                "degree": int(len(neighbor_items)),
                "neighbors": [
                    {"id": int(neighbor_id), "shared_length": int(shared_length)}
                    for neighbor_id, shared_length in neighbor_items
                ],
                "prev_neighbors": previous_neighbors,
                "next_neighbors": next_neighbors,
            }
        )

    return {
        "source_partition_graph": source_path,
        "source_mask": graph_data.get("source_mask"),
        "size": [height, width],
        "ordering": {
            "type": "bfs_border_area",
            "face_ids": [int(old_face_id) for old_face_id in order],
        },
        "faces": dual_faces,
        "edges": dual_edges,
        "stats": {
            "num_faces": int(len(dual_faces)),
            "num_edges": int(len(dual_edges)),
            "max_prev_neighbors": int(max_prev_neighbors),
            "mean_prev_shared_length": float(sum(prev_neighbor_lengths) / len(prev_neighbor_lengths))
            if prev_neighbor_lengths
            else 0.0,
        },
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
