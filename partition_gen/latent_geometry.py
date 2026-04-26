from __future__ import annotations

from typing import Dict, List, Sequence

from shapely.geometry import Polygon
from shapely.ops import unary_union

from partition_gen.operation_geometry import largest_polygon, polygon_from_face, union_face_polygons
from partition_gen.operation_types import DIVIDE_BY_REGION, OVERLAY_INSERT, OperationExplainerConfig


def _valid_candidate(policy: str, geometry, *, extra_cost: float, metadata: Dict[str, object] | None = None) -> Dict[str, object]:
    if geometry is None or geometry.is_empty or float(geometry.area) <= 0.0:
        return {
            "policy": policy,
            "geometry": Polygon(),
            "extra_cost": float(extra_cost),
            "valid": False,
            "failure_reason": "empty_geometry",
            "metadata": metadata or {},
        }
    fixed = geometry if geometry.is_valid else geometry.buffer(0)
    if fixed.is_empty or float(fixed.area) <= 0.0:
        return {
            "policy": policy,
            "geometry": Polygon(),
            "extra_cost": float(extra_cost),
            "valid": False,
            "failure_reason": "invalid_geometry",
            "metadata": metadata or {},
        }
    return {
        "policy": policy,
        "geometry": fixed,
        "extra_cost": float(extra_cost),
        "valid": True,
        "failure_reason": None,
        "metadata": metadata or {},
    }


def build_latent_geometry_candidates(
    support_faces: Sequence[Dict[str, object]],
    child_faces: Sequence[Dict[str, object]],
    operation_type: str,
    config: OperationExplainerConfig,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    support_union = union_face_polygons(support_faces)
    child_union = union_face_polygons(child_faces)

    if config.enable_visible_union:
        candidates.append(_valid_candidate("visible_union", support_union, extra_cost=0.0))

    if operation_type == OVERLAY_INSERT and config.enable_union_with_children:
        geometry = unary_union([item for item in (support_union, child_union) if not item.is_empty])
        candidates.append(_valid_candidate("union_with_children", geometry, extra_cost=0.5))

    if operation_type == DIVIDE_BY_REGION and config.enable_union_with_divider:
        geometry = unary_union([item for item in (support_union, child_union) if not item.is_empty])
        candidates.append(_valid_candidate("union_with_divider", geometry, extra_cost=0.5))

    if config.enable_convex_hull_fill:
        base = unary_union([item for item in (support_union, child_union) if not item.is_empty])
        candidates.append(_valid_candidate("convex_hull_fill", base.convex_hull if not base.is_empty else Polygon(), extra_cost=2.0))

    if config.enable_buffer_close_fill:
        base = unary_union([item for item in (support_union, child_union) if not item.is_empty])
        radius = float(config.buffer_close_radius)
        geometry = base.buffer(radius).buffer(-radius) if not base.is_empty else Polygon()
        candidates.append(
            _valid_candidate(
                "buffer_close_fill",
                geometry,
                extra_cost=2.5 + 0.1 * radius,
                metadata={"radius": radius},
            )
        )

    unique: List[Dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["policy"],
            round(float(candidate["geometry"].area), 6) if candidate["valid"] else 0.0,
            round(float(largest_polygon(candidate["geometry"]).length), 6) if candidate["valid"] else 0.0,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def face_geometry(face: Dict[str, object]):
    return polygon_from_face(face)
