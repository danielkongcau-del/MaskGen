from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely import coverage_union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


Point = Tuple[float, float]


@dataclass(frozen=True)
class WeakRenderConfig:
    area_eps: float = 1e-8
    validity_eps: float = 1e-6
    raster_fill_background: int = 0


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1e-9:
        ring = ring[:-1]
    return ring


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            if not polygon.is_empty:
                yield polygon
        return
    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def _fix_geometry(geometry):
    if geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    return geometry.buffer(0)


def _polygonal_union(geometries: Sequence):
    polygons = []
    for geometry in geometries:
        polygons.extend([polygon for polygon in _iter_polygons(geometry) if polygon.area > 0.0])
    if not polygons:
        return Polygon()
    unioned = unary_union(polygons)
    unioned = _fix_geometry(unioned)
    polygons = [polygon for polygon in _iter_polygons(unioned) if polygon.area > 0.0]
    return unary_union(polygons) if polygons else Polygon()


def _coverage_union(geometries: Sequence):
    polygons = []
    for geometry in geometries:
        polygons.extend([polygon for polygon in _iter_polygons(geometry) if polygon.area > 0.0])
    if not polygons:
        return Polygon()
    try:
        return _fix_geometry(coverage_union_all(polygons))
    except Exception:
        return _polygonal_union(polygons)


def _polygon_to_payload(polygon: Polygon) -> Dict[str, object]:
    polygon = orient(polygon, sign=1.0)
    return {
        "outer": [[float(x), float(y)] for x, y in _trim_ring(polygon.exterior.coords)],
        "holes": [
            [[float(x), float(y)] for x, y in _trim_ring(interior.coords)]
            for interior in polygon.interiors
            if len(_trim_ring(interior.coords)) >= 3
        ],
    }


def _geometry_to_payload(geometry) -> List[Dict[str, object]]:
    return [_polygon_to_payload(polygon) for polygon in _iter_polygons(geometry) if polygon.area > 0.0]


def _polygon_from_evidence_face(face: Dict[str, object]) -> Polygon:
    geometry = face.get("geometry") or {}
    outer = _trim_ring(geometry.get("outer", []))
    holes = [_trim_ring(ring) for ring in geometry.get("holes", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer, holes), sign=1.0)
    except Exception:
        fixed = Polygon(outer, holes).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _local_to_world(point: Sequence[float], frame: Dict[str, object]) -> List[float]:
    cx, cy = [float(value) for value in frame["origin"]]
    scale = max(float(frame["scale"]), 1e-8)
    theta = float(frame["orientation"])
    x = float(point[0]) * scale
    y = float(point[1]) * scale
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return [float(cx + x * cos_t - y * sin_t), float(cy + x * sin_t + y * cos_t)]


def _local_ring_to_world(ring: Sequence[Sequence[float]], frame: Dict[str, object]) -> List[List[float]]:
    return [_local_to_world(point, frame) for point in ring]


def _polygon_from_local_geometry(geometry: Dict[str, object], frame: Dict[str, object]) -> Polygon:
    outer = _local_ring_to_world(geometry.get("outer_local", []), frame)
    holes = [_local_ring_to_world(ring, frame) for ring in geometry.get("holes_local", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer, holes), sign=1.0)
    except Exception:
        fixed = Polygon(outer, holes).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _atom_polygon(atom_node: Dict[str, object]) -> Polygon:
    geometry = atom_node.get("geometry") or {}
    frame = atom_node.get("frame") or {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0}
    outer = _local_ring_to_world(geometry.get("outer_local", []), frame)
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer), sign=1.0)
    except Exception:
        fixed = Polygon(outer).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _iou(left, right, *, eps: float) -> float:
    if left.is_empty and right.is_empty:
        return 1.0
    union_area = float(left.union(right).area)
    if union_area <= eps:
        return 0.0
    return float(left.intersection(right).area / union_area)


def _pairwise_overlap_area(geometries: Sequence) -> float:
    overlap = 0.0
    for left_index, left in enumerate(geometries):
        if left.is_empty:
            continue
        for right in geometries[left_index + 1 :]:
            if right.is_empty:
                continue
            inter = left.intersection(right)
            if not inter.is_empty:
                overlap += float(inter.area)
    return float(overlap)


def _mask_from_geometries(rendered_faces: Sequence[Dict[str, object]], *, size: Sequence[int], background: int) -> np.ndarray:
    height, width = [int(value) for value in size]
    image = Image.new("I", (width, height), int(background))
    draw = ImageDraw.Draw(image)
    for face in sorted(rendered_faces, key=lambda item: -float(item.get("area", 0.0))):
        label = int(face.get("label", 0))
        for polygon in _iter_polygons(face["geometry"]):
            outer = [(float(x), float(y)) for x, y in polygon.exterior.coords]
            if len(outer) >= 3:
                draw.polygon(outer, fill=label)
            for interior in polygon.interiors:
                hole = [(float(x), float(y)) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=int(background))
    return np.asarray(image, dtype=np.int32)


def _mask_iou(rendered: np.ndarray, target: np.ndarray) -> float:
    if rendered.shape != target.shape:
        return 0.0
    equal = rendered == target
    return float(equal.sum() / equal.size) if equal.size else 0.0


def _load_target_mask(mask_path: str | Path | None) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    return np.asarray(Image.open(path), dtype=np.int32)


def _source_mask_path(evidence_payload: Dict[str, object], *, mask_root: Path | None = None) -> Path | None:
    source_mask = evidence_payload.get("source_mask")
    if source_mask:
        path = Path(str(source_mask))
        if path.exists():
            return path
        if mask_root is not None and len(path.parts) >= 2:
            candidate = mask_root / path
            if candidate.exists():
                return candidate
            if len(path.parts) >= 3:
                candidate = mask_root / path.parts[-3] / path.parts[-2] / path.name
                if candidate.exists():
                    return candidate
    return None


def render_weak_explanation_payload(
    weak_payload: Dict[str, object],
    *,
    evidence_payload: Dict[str, object] | None = None,
    config: WeakRenderConfig | None = None,
    mask_root: Path | None = None,
) -> Dict[str, object]:
    config = config or WeakRenderConfig()
    target = weak_payload.get("generator_target", {})
    graph = target.get("parse_graph", {})
    nodes = graph.get("nodes", [])
    relations = graph.get("relations", [])
    size = target.get("size", [0, 0])
    evidence_by_face = {
        int(face["id"]): face
        for face in (evidence_payload or {}).get("faces", [])
    }
    evidence_polygons = {
        face_id: _polygon_from_evidence_face(face)
        for face_id, face in evidence_by_face.items()
    }

    nodes_by_id = {str(node["id"]): node for node in nodes}
    atom_ids_by_face: Dict[str, List[str]] = {}
    for relation in relations:
        if relation.get("type") == "atom_part_of":
            atom_ids_by_face.setdefault(str(relation.get("face")), []).append(str(relation.get("atom")))

    rendered_faces: List[Dict[str, object]] = []
    per_face_validation: List[Dict[str, object]] = []
    for face_node in [node for node in nodes if node.get("role") == "semantic_face"]:
        face_id = str(face_node["id"])
        source_face_id = int(face_node.get("source_face_id", -1))
        atom_polygons = []
        for atom_id in atom_ids_by_face.get(face_id, face_node.get("atom_ids", [])):
            atom_node = nodes_by_id.get(str(atom_id))
            if not atom_node:
                continue
            polygon = _fix_geometry(_atom_polygon(atom_node))
            if not polygon.is_empty and polygon.area > config.area_eps:
                atom_polygons.append(polygon)
        if atom_polygons:
            rendered_geometry = _polygonal_union(atom_polygons)
            render_source = "convex_atoms"
        else:
            geometry_model = str(face_node.get("geometry_model"))
            if geometry_model == "polygon_code":
                rendered_geometry = _fix_geometry(_polygon_from_local_geometry(face_node.get("geometry", {}), face_node.get("frame", {})))
                render_source = "semantic_face_polygon_code"
            elif source_face_id in evidence_polygons:
                rendered_geometry = evidence_polygons[source_face_id]
                render_source = "evidence_face_fallback"
            else:
                rendered_geometry = Polygon()
                render_source = "missing_geometry"
        evidence_polygon = evidence_polygons.get(source_face_id, Polygon())
        face_iou = _iou(rendered_geometry, evidence_polygon, eps=config.area_eps) if evidence_payload else None
        missing_area = float(max(0.0, evidence_polygon.difference(rendered_geometry).area)) if evidence_payload else None
        extra_area = float(max(0.0, rendered_geometry.difference(evidence_polygon).area)) if evidence_payload else None
        rendered_faces.append(
            {
                "id": face_id,
                "source_face_id": source_face_id,
                "label": int(face_node.get("label", 0)),
                "render_source": render_source,
                "geometry": rendered_geometry,
                "area": float(rendered_geometry.area) if not rendered_geometry.is_empty else 0.0,
                "atom_ids": list(atom_ids_by_face.get(face_id, face_node.get("atom_ids", []))),
            }
        )
        per_face_validation.append(
            {
                "face_node_id": face_id,
                "source_face_id": source_face_id,
                "label": int(face_node.get("label", 0)),
                "render_source": render_source,
                "atom_count": int(len(atom_polygons)),
                "rendered_area": float(rendered_geometry.area) if not rendered_geometry.is_empty else 0.0,
                "evidence_area": float(evidence_polygon.area) if not evidence_polygon.is_empty else None,
                "iou": face_iou,
                "missing_area": missing_area,
                "extra_area": extra_area,
                "is_valid": bool(not rendered_geometry.is_empty and rendered_geometry.is_valid),
            }
        )

    rendered_geometries = [item["geometry"] for item in rendered_faces if not item["geometry"].is_empty]
    rendered_union = _coverage_union(rendered_geometries) if rendered_geometries else Polygon()
    evidence_union = _coverage_union(list(evidence_polygons.values())) if evidence_polygons else Polygon()
    overlap_area = _pairwise_overlap_area(rendered_geometries)
    gap_area = float(evidence_union.difference(rendered_union).area) if evidence_payload else None
    extra_area = float(rendered_union.difference(evidence_union).area) if evidence_payload else None
    full_iou = _iou(rendered_union, evidence_union, eps=config.area_eps) if evidence_payload else None
    rendered_mask = _mask_from_geometries(rendered_faces, size=size, background=config.raster_fill_background)
    target_mask = _load_target_mask(_source_mask_path(evidence_payload or {}, mask_root=mask_root))
    mask_iou = _mask_iou(rendered_mask, target_mask) if target_mask is not None else None
    invalid_faces = [item["face_node_id"] for item in per_face_validation if not item["is_valid"]]
    low_iou_faces = [
        item["face_node_id"]
        for item in per_face_validation
        if item["iou"] is not None and float(item["iou"]) < 0.999
    ]
    validation = {
        "is_valid": bool(not invalid_faces and overlap_area <= config.validity_eps and (full_iou is None or full_iou >= 0.999)),
        "face_count": int(len(rendered_faces)),
        "invalid_face_ids": invalid_faces,
        "low_iou_face_ids": low_iou_faces,
        "full_iou": full_iou,
        "mask_pixel_accuracy": mask_iou,
        "overlap_area": float(overlap_area),
        "gap_area": gap_area,
        "extra_area": extra_area,
        "rendered_area": float(rendered_union.area) if not rendered_union.is_empty else 0.0,
        "evidence_area": float(evidence_union.area) if not evidence_union.is_empty else None,
        "per_face": per_face_validation,
    }
    return {
        "format": "maskgen_weak_rendered_partition_v1",
        "source_explanation": weak_payload.get("source_evidence"),
        "size": size,
        "faces": [
            {
                "id": item["id"],
                "source_face_id": item["source_face_id"],
                "label": item["label"],
                "render_source": item["render_source"],
                "atom_ids": item["atom_ids"],
                "area": item["area"],
                "polygons": _geometry_to_payload(item["geometry"]),
            }
            for item in rendered_faces
        ],
        "validation": validation,
        "config": asdict(config),
        "_rendered_mask": rendered_mask,
    }


def save_render_outputs(
    render_payload: Dict[str, object],
    *,
    partition_path: Path,
    mask_path: Path | None = None,
    validation_path: Path | None = None,
) -> None:
    mask = render_payload.pop("_rendered_mask")
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_path.write_text(
        __import__("json").dumps(render_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if validation_path is not None:
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            __import__("json").dumps(render_payload.get("validation", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if mask_path is not None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)).save(mask_path)
