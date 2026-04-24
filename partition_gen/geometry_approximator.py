from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from partition_gen.cdt_partition import polygon_payload
from partition_gen.primitive_decomposition import (
    decompose_partition_face,
    face_geometry,
    primitives_union_geometry,
)


@dataclass(frozen=True)
class GeometryApproximationConfig:
    simplify_tolerance: float = 1.5
    area_epsilon: float = 1e-3
    trim_collinear_eps: float = 1e-9


def _vertex_count(polygon: Polygon, *, eps: float) -> int:
    polygon = orient(polygon, sign=1.0)
    total = 0
    outer = polygon_payload(polygon, eps=eps)["outer"]
    total += len(outer)
    for ring in polygon_payload(polygon, eps=eps)["holes"]:
        total += len(ring)
    return int(total)


def approximate_face_from_partition_graph(
    graph_data: Dict[str, object],
    face_data: Dict[str, object],
    *,
    config: GeometryApproximationConfig | None = None,
) -> Dict[str, object]:
    config = config or GeometryApproximationConfig()
    original_geometry = face_geometry(graph_data, face_data)
    base_payload = decompose_partition_face(
        graph_data,
        face_data,
        simplify_tolerance=config.simplify_tolerance,
        area_epsilon=config.area_epsilon,
    )
    approx_geometry = primitives_union_geometry(base_payload["primitives"])
    approx_geometry = orient(approx_geometry, sign=1.0) if isinstance(approx_geometry, Polygon) else approx_geometry

    payload = {
        "face_id": int(face_data["id"]),
        "label": int(face_data["label"]),
        "bbox": [int(value) for value in face_data["bbox"]],
        "original_hole_count": int(len(face_data["holes"])),
        "base_primitive_count": int(base_payload["primitive_count"]),
        "base_triangle_count": int(base_payload["triangle_count"]),
        "base_quad_count": int(base_payload["quad_count"]),
        "simplify_tolerance": float(config.simplify_tolerance),
        "original_area": float(original_geometry.area),
        "approx_area": float(approx_geometry.area) if not approx_geometry.is_empty else 0.0,
        "approx_iou": float(base_payload["approx_iou"]),
        "original_vertex_count": int(_vertex_count(original_geometry, eps=config.trim_collinear_eps)),
        "approx_vertex_count": int(_vertex_count(approx_geometry, eps=config.trim_collinear_eps))
        if not approx_geometry.is_empty
        else 0,
        "base_primitives": base_payload["primitives"],
    }
    payload["original_geometry"] = polygon_payload(original_geometry, eps=config.trim_collinear_eps)
    payload["approx_geometry"] = (
        polygon_payload(approx_geometry, eps=config.trim_collinear_eps)
        if not approx_geometry.is_empty
        else {"outer": [], "holes": []}
    )
    return payload
