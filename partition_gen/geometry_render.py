from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon


Point = Tuple[float, float]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def denormalize_local_vertices(face_meta: Dict[str, object], vertices_local: Sequence[Sequence[float]]) -> List[Point]:
    cx, cy = (float(face_meta["centroid"][0]), float(face_meta["centroid"][1]))
    scale_x = max(float(face_meta["bbox_width"]) / 2.0, 1.0)
    scale_y = max(float(face_meta["bbox_height"]) / 2.0, 1.0)
    return [(cx + float(x) * scale_x, cy + float(y) * scale_y) for x, y in vertices_local]


def clamp_vertices(vertices: Sequence[Point], size: Tuple[int, int]) -> List[Point]:
    height, width = size
    return [
        (
            min(max(float(x), 0.0), float(width)),
            min(max(float(y), 0.0), float(height)),
        )
        for x, y in vertices
    ]


def repair_polygon(vertices: Sequence[Point], holes: Sequence[Sequence[Point]] | None = None) -> Polygon | None:
    if len(vertices) < 3:
        return None
    clean_holes = []
    if holes is not None:
        for hole in holes:
            if len(hole) >= 3:
                clean_holes.append(hole)
    polygon = Polygon(vertices, holes=clean_holes)
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(polygon, Polygon):
        return None
    if polygon.area <= 0:
        return None
    return polygon


def polygon_pixels(polygon: Polygon, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = size
    min_x, min_y, max_x, max_y = polygon.bounds
    x0 = max(0, int(math.floor(min_x)))
    y0 = max(0, int(math.floor(min_y)))
    x1 = min(width, int(math.ceil(max_x)))
    y1 = min(height, int(math.ceil(max_y)))
    if x0 >= x1 or y0 >= y1:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=bool)
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = contains_xy(polygon, grid_x, grid_y)
    return np.arange(x0, x1), np.arange(y0, y1), inside


def render_geometry_prediction(
    *,
    dual_graph: Dict[str, object],
    prediction: Dict[str, object],
    size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    height, width = size
    label_map = np.full((height, width), -1, dtype=np.int16)
    score_map = np.full((height, width), np.inf, dtype=np.float32)
    face_metas = {int(face["id"]): face for face in dual_graph["faces"]}
    rendered_faces = []

    for face_pred in prediction["faces"]:
        face_id = int(face_pred["id"])
        face_meta = face_metas.get(face_id)
        if face_meta is None:
            continue
        if int(face_pred.get("support_pred", 0)) <= 0:
            continue
        vertex_count = int(face_pred.get("vertex_count_pred", 0))
        vertices_local = face_pred.get("vertices_local_pred", [])[:vertex_count]
        if len(vertices_local) < 3:
            continue
        vertices = denormalize_local_vertices(face_meta, vertices_local)
        vertices = clamp_vertices(vertices, size=size)
        hole_count = int(face_pred.get("hole_count_pred", 0))
        hole_vertex_counts = face_pred.get("hole_vertex_counts_pred", [])
        hole_vertices_local = face_pred.get("hole_vertices_local_pred", [])
        holes = []
        for hole_index in range(min(hole_count, len(hole_vertices_local), len(hole_vertex_counts))):
            hv_count = int(hole_vertex_counts[hole_index])
            if hv_count < 3:
                continue
            hole_vertices = denormalize_local_vertices(face_meta, hole_vertices_local[hole_index][:hv_count])
            holes.append(clamp_vertices(hole_vertices, size=size))
        polygon = repair_polygon(vertices, holes=holes)
        if polygon is None:
            continue

        xs, ys, inside = polygon_pixels(polygon, size=size)
        if inside.size == 0:
            continue

        cx, cy = (float(face_meta["centroid"][0]), float(face_meta["centroid"][1]))
        norm_x = max(float(face_meta["bbox_width"]) / 2.0, 1.0)
        norm_y = max(float(face_meta["bbox_height"]) / 2.0, 1.0)
        grid_x, grid_y = np.meshgrid(xs + 0.5, ys + 0.5)
        distances = ((grid_x - cx) / norm_x) ** 2 + ((grid_y - cy) / norm_y) ** 2
        region_scores = score_map[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        region_labels = label_map[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]

        update = inside & (distances < region_scores)
        region_scores[update] = distances[update]
        region_labels[update] = int(face_meta["label"])

        rendered_faces.append(
            {
                "id": face_id,
                "label": int(face_meta["label"]),
                "polygon_area": float(polygon.area),
                "vertex_count_pred": vertex_count,
                "hole_count_pred": hole_count,
            }
        )

    centroids = np.asarray(
        [[float(face["centroid"][0]), float(face["centroid"][1])] for face in dual_graph["faces"]],
        dtype=np.float32,
    )
    labels = np.asarray([int(face["label"]) for face in dual_graph["faces"]], dtype=np.int16)
    missing = label_map < 0
    if missing.any():
        missing_y, missing_x = np.nonzero(missing)
        pixel_centers = np.stack([missing_x + 0.5, missing_y + 0.5], axis=1).astype(np.float32)
        squared_dist = ((pixel_centers[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        nearest = squared_dist.argmin(axis=1)
        label_map[missing_y, missing_x] = labels[nearest]

    summary = {
        "rendered_faces": len(rendered_faces),
        "total_faces": len(dual_graph["faces"]),
        "filled_pixels": int((label_map >= 0).sum()),
        "unfilled_pixels": int((label_map < 0).sum()),
    }
    return label_map.astype(np.uint8), {"faces": rendered_faces, "summary": summary}


def remap_ids_to_values(mask_id: np.ndarray, id_to_value: Dict[str, int]) -> np.ndarray:
    out = np.zeros_like(mask_id, dtype=np.uint8)
    for key, value in id_to_value.items():
        out[mask_id == int(key)] = int(value)
    return out
