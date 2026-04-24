from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from partition_gen.geometry_render import (
    clamp_vertices,
    denormalize_local_vertices,
    repair_polygon,
)


Point = Tuple[float, float]
IntPoint = Tuple[int, int]


def _line_pixels(point_a: Point, point_b: Point, size: Tuple[int, int]) -> List[IntPoint]:
    height, width = size
    x0, y0 = point_a
    x1, y1 = point_b
    steps = max(int(np.ceil(abs(x1 - x0))), int(np.ceil(abs(y1 - y0))), 1) * 2
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    coords: List[IntPoint] = []
    for x, y in zip(xs, ys):
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        if not coords or coords[-1] != (ix, iy):
            coords.append((ix, iy))
    return coords


def _draw_ring(mask: np.ndarray, coords: Sequence[Point]) -> None:
    if len(coords) < 2:
        return
    for index in range(len(coords)):
        point_a = coords[index]
        point_b = coords[(index + 1) % len(coords)]
        for x, y in _line_pixels(point_a, point_b, size=mask.shape):
            mask[y, x] = True


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def hole_vertices_from_prediction(
    face_meta: Dict[str, object],
    face_pred: Dict[str, object],
    size: Tuple[int, int],
) -> List[List[Point]]:
    hole_count = int(face_pred.get("hole_count_pred", 0))
    hole_vertex_counts = face_pred.get("hole_vertex_counts_pred", [])
    hole_vertices_local = face_pred.get("hole_vertices_local_pred", [])
    holes: List[List[Point]] = []
    for hole_index in range(min(hole_count, len(hole_vertex_counts), len(hole_vertices_local))):
        vertex_count = int(hole_vertex_counts[hole_index])
        if vertex_count < 3:
            continue
        vertices = denormalize_local_vertices(face_meta, hole_vertices_local[hole_index][:vertex_count])
        holes.append(clamp_vertices(vertices, size=size))
    return holes


def polygon_from_prediction(
    face_meta: Dict[str, object],
    face_pred: Dict[str, object],
    size: Tuple[int, int],
) -> Polygon | None:
    vertex_count = int(face_pred.get("vertex_count_pred", 0))
    vertices_local = face_pred.get("vertices_local_pred", [])[:vertex_count]
    if len(vertices_local) < 3:
        return None
    vertices = denormalize_local_vertices(face_meta, vertices_local)
    vertices = clamp_vertices(vertices, size=size)
    holes = hole_vertices_from_prediction(face_meta, face_pred, size=size)
    return repair_polygon(vertices, holes=holes)


def build_boundary_mask_from_prediction(
    *,
    dual_graph: Dict[str, object],
    prediction: Dict[str, object],
    size: Tuple[int, int],
    boundary_dilation: int = 1,
) -> Tuple[np.ndarray, Dict[str, int]]:
    boundary = np.zeros(size, dtype=bool)
    face_metas = {int(face["id"]): face for face in dual_graph["faces"]}
    used_faces = 0

    for face_pred in prediction["faces"]:
        if int(face_pred.get("support_pred", 0)) <= 0:
            continue
        face_meta = face_metas.get(int(face_pred["id"]))
        if face_meta is None:
            continue
        polygon = polygon_from_prediction(face_meta, face_pred, size=size)
        if polygon is None:
            continue
        used_faces += 1
        _draw_ring(boundary, list(polygon.exterior.coords))
        for interior in polygon.interiors:
            _draw_ring(boundary, list(interior.coords))

    boundary = _dilate(boundary, radius=boundary_dilation)
    meta = {
        "used_faces": used_faces,
        "boundary_pixels": int(boundary.sum()),
    }
    return boundary, meta


def _nearest_open_pixel(boundary: np.ndarray, start_xy: Point, max_radius: int = 32) -> IntPoint:
    height, width = boundary.shape
    sx = int(np.clip(round(start_xy[0]), 0, width - 1))
    sy = int(np.clip(round(start_xy[1]), 0, height - 1))
    if not boundary[sy, sx]:
        return sx, sy
    for radius in range(1, max_radius + 1):
        x0 = max(0, sx - radius)
        x1 = min(width - 1, sx + radius)
        y0 = max(0, sy - radius)
        y1 = min(height - 1, sy + radius)
        candidates: List[IntPoint] = []
        for x in range(x0, x1 + 1):
            candidates.append((x, y0))
            candidates.append((x, y1))
        for y in range(y0 + 1, y1):
            candidates.append((x0, y))
            candidates.append((x1, y))
        best: IntPoint | None = None
        best_dist = None
        for x, y in candidates:
            if boundary[y, x]:
                continue
            dist = abs(x - sx) + abs(y - sy)
            if best is None or dist < best_dist:
                best = (x, y)
                best_dist = dist
        if best is not None:
            return best
    return sx, sy


def render_partition_from_boundaries(
    *,
    dual_graph: Dict[str, object],
    boundary_mask: np.ndarray,
    size: Tuple[int, int],
    use_all_faces: bool = True,
    prediction: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    height, width = size
    face_owner = np.full((height, width), -1, dtype=np.int32)
    distance = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[Tuple[int, int]] = deque()

    allowed_faces = None
    if not use_all_faces and prediction is not None:
        allowed_faces = {
            int(face_pred["id"])
            for face_pred in prediction["faces"]
            if int(face_pred.get("support_pred", 0)) > 0
        }

    seed_records = []
    seed_collisions = 0
    for face in dual_graph["faces"]:
        face_id = int(face["id"])
        if allowed_faces is not None and face_id not in allowed_faces:
            continue
        x, y = _nearest_open_pixel(boundary_mask, (float(face["centroid"][0]), float(face["centroid"][1])))
        if boundary_mask[y, x]:
            continue
        if face_owner[y, x] != -1:
            seed_collisions += 1
            continue
        face_owner[y, x] = face_id
        distance[y, x] = 0
        queue.append((x, y))
        seed_records.append({"face_id": face_id, "x": x, "y": y, "label": int(face["label"])})

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        x, y = queue.popleft()
        owner = face_owner[y, x]
        next_distance = distance[y, x] + 1
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if boundary_mask[ny, nx]:
                continue
            if next_distance < distance[ny, nx]:
                distance[ny, nx] = next_distance
                face_owner[ny, nx] = owner
                queue.append((nx, ny))

    label_map = np.full((height, width), -1, dtype=np.int16)
    labels = np.asarray([int(face["label"]) for face in dual_graph["faces"]], dtype=np.int16)
    open_pixels = face_owner >= 0
    label_map[open_pixels] = labels[face_owner[open_pixels]]

    boundary_pixels = np.argwhere(boundary_mask)
    for y, x in boundary_pixels:
        neighbor_labels = []
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            label = label_map[ny, nx]
            if label >= 0:
                neighbor_labels.append(int(label))
        if neighbor_labels:
            values, counts = np.unique(np.asarray(neighbor_labels), return_counts=True)
            label_map[y, x] = int(values[counts.argmax()])

    missing = label_map < 0
    if missing.any():
        centroids = np.asarray(
            [[float(face["centroid"][0]), float(face["centroid"][1])] for face in dual_graph["faces"]],
            dtype=np.float32,
        )
        missing_y, missing_x = np.nonzero(missing)
        pixel_centers = np.stack([missing_x.astype(np.float32), missing_y.astype(np.float32)], axis=1)
        squared_dist = ((pixel_centers[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        nearest = squared_dist.argmin(axis=1)
        label_map[missing_y, missing_x] = labels[nearest]

    summary = {
        "seed_count": len(seed_records),
        "seed_collisions": seed_collisions,
        "boundary_pixels": int(boundary_mask.sum()),
        "unfilled_pixels": int((label_map < 0).sum()),
    }
    return label_map.astype(np.uint8), {"seeds": seed_records, "summary": summary}


def render_joint_partition_prediction(
    *,
    dual_graph: Dict[str, object],
    prediction: Dict[str, object],
    size: Tuple[int, int],
    boundary_dilation: int = 1,
    use_all_faces: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    boundary_mask, boundary_meta = build_boundary_mask_from_prediction(
        dual_graph=dual_graph,
        prediction=prediction,
        size=size,
        boundary_dilation=boundary_dilation,
    )
    label_map, render_meta = render_partition_from_boundaries(
        dual_graph=dual_graph,
        boundary_mask=boundary_mask,
        size=size,
        use_all_faces=use_all_faces,
        prediction=prediction,
    )
    meta = {
        "boundary": boundary_meta,
        "summary": {
            **render_meta["summary"],
            "used_faces": boundary_meta["used_faces"],
            "filled_pixels": int((label_map >= 0).sum()),
        },
        "seeds": render_meta["seeds"],
        "boundary_mask": boundary_mask,
    }
    return label_map, meta
