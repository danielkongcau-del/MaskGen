from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import LineString, Point as ShapelyPoint, Polygon
from shapely.geometry.polygon import orient

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    build_bridged_convex_partition_from_geometry_payload,
)
from partition_gen.global_approx_partition import EXTERIOR_FACE_ID


Point = Tuple[float, float]


@dataclass(frozen=True)
class ExplanationEvidenceConfig:
    convex_backend: str = "auto"
    convex_cgal_cli: str | None = None
    convex_max_bridge_sets: int = 256
    convex_cut_slit_scale: float = 1e-6
    convex_validity_eps: float = 1e-7
    convex_area_eps: float = 1e-8
    thin_aspect_ratio: float = 4.0
    compactness_threshold: float = 0.45
    min_area_eps: float = 1e-8
    validity_eps: float = 1e-6


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and _points_close(ring[0], ring[-1], eps=1e-9):
        ring = ring[:-1]
    return ring


def _points_close(left: Point, right: Point, *, eps: float) -> bool:
    return math.hypot(left[0] - right[0], left[1] - right[1]) <= eps


def _polygon_from_face(face: Dict[str, object]) -> Polygon:
    outer = _trim_ring(face.get("outer", []))
    holes = [_trim_ring(ring) for ring in face.get("holes", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer, holes), sign=1.0)
    except Exception:
        fixed = Polygon(outer, holes).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _polygon_outer(polygon: Polygon) -> List[List[float]]:
    if polygon.is_empty:
        return []
    return [[float(x), float(y)] for x, y in _trim_ring(polygon.exterior.coords)]


def _polygon_holes(polygon: Polygon) -> List[List[List[float]]]:
    if polygon.is_empty:
        return []
    holes: List[List[List[float]]] = []
    for interior in polygon.interiors:
        ring = _trim_ring(interior.coords)
        if len(ring) >= 3:
            holes.append([[float(x), float(y)] for x, y in ring])
    return holes


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(
                float(points[index + 1][0]) - float(points[index][0]),
                float(points[index + 1][1]) - float(points[index][1]),
            )
            for index in range(len(points) - 1)
        )
    )


def _endpoint_distance(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        math.hypot(
            float(points[-1][0]) - float(points[0][0]),
            float(points[-1][1]) - float(points[0][1]),
        )
    )


def _max_point_line_distance(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    line = LineString([points[0], points[-1]])
    if line.length <= 1e-12:
        return 0.0
    return float(max(line.distance(ShapelyPoint(point)) for point in points[1:-1]))


def _compactness(area: float, perimeter: float, *, eps: float) -> float:
    if perimeter <= eps:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _oriented_bbox_dimensions(polygon: Polygon) -> Tuple[float, float]:
    if polygon.is_empty:
        return 0.0, 0.0
    rectangle = polygon.minimum_rotated_rectangle
    coords = _trim_ring(rectangle.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0
    lengths = [
        math.hypot(coords[(index + 1) % len(coords)][0] - coords[index][0], coords[(index + 1) % len(coords)][1] - coords[index][1])
        for index in range(len(coords))
    ]
    unique = sorted(lengths)
    if not unique:
        return 0.0, 0.0
    width = float(unique[0])
    height = float(unique[-1])
    return width, height


def _aspect_ratio(width: float, height: float, *, eps: float) -> float:
    short = min(width, height)
    long = max(width, height)
    if short <= eps:
        return float("inf") if long > eps else 0.0
    return float(long / short)


def _face_adjacency_from_arcs(arcs: Sequence[Dict[str, object]], labels_by_face: Dict[int, int]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, int], Dict[str, object]] = {}
    for arc in arcs:
        semantic_faces = sorted(int(face_id) for face_id in arc.get("incident_faces", []) if int(face_id) != EXTERIOR_FACE_ID)
        if len(semantic_faces) != 2:
            continue
        key = (semantic_faces[0], semantic_faces[1])
        item = grouped.setdefault(
            key,
            {
                "faces": [int(key[0]), int(key[1])],
                "labels": [int(labels_by_face.get(key[0], -1)), int(labels_by_face.get(key[1], -1))],
                "arc_ids": [],
                "shared_length": 0.0,
                "arc_count": 0,
            },
        )
        item["arc_ids"].append(int(arc["id"]))
        item["shared_length"] = float(item["shared_length"]) + float(arc.get("features", {}).get("length", arc.get("length", 0.0)))
        item["arc_count"] = int(item["arc_count"]) + 1
    return sorted(grouped.values(), key=lambda item: (item["faces"][0], item["faces"][1]))


def _face_adjacency_from_geometry(
    polygons_by_face: Dict[int, Polygon],
    labels_by_face: Dict[int, int],
    *,
    eps: float,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    face_ids = sorted(polygons_by_face)
    for left_index, left_id in enumerate(face_ids):
        left = polygons_by_face[left_id]
        if left.is_empty:
            continue
        for right_id in face_ids[left_index + 1 :]:
            right = polygons_by_face[right_id]
            if right.is_empty:
                continue
            if not left.bounds or not right.bounds:
                continue
            if left.boundary.distance(right.boundary) > eps:
                continue
            shared = left.boundary.intersection(right.boundary)
            shared_length = float(shared.length) if not shared.is_empty else 0.0
            if shared_length <= eps:
                continue
            output.append(
                {
                    "faces": [int(left_id), int(right_id)],
                    "labels": [int(labels_by_face.get(left_id, -1)), int(labels_by_face.get(right_id, -1))],
                    "arc_ids": [],
                    "shared_length": shared_length,
                    "arc_count": 0,
                    "source": "geometry_fallback",
                }
            )
    return output


def _arc_evidence(arc: Dict[str, object]) -> Dict[str, object]:
    points = [[float(x), float(y)] for x, y in arc.get("points", [])]
    incident_faces = [int(face_id) for face_id in arc.get("incident_faces", [])]
    semantic_faces = [face_id for face_id in incident_faces if face_id != EXTERIOR_FACE_ID]
    length = float(arc.get("length", _polyline_length(points)))
    original_vertex_count = int(arc.get("original_vertex_count", len(points)))
    vertex_count = int(arc.get("vertex_count", len(points)))
    compression_ratio = float(vertex_count / original_vertex_count) if original_vertex_count > 0 else None
    return {
        "id": int(arc["id"]),
        "incident_faces": incident_faces,
        "is_shared": bool(len(semantic_faces) == 2),
        "is_border": bool(len(semantic_faces) == 1 or EXTERIOR_FACE_ID in incident_faces),
        "source_edge_ids": [int(value) for value in arc.get("source_edge_ids", [])],
        "points": points,
        "features": {
            "length": length,
            "vertex_count": vertex_count,
            "original_vertex_count": original_vertex_count,
            "vertex_reduction": int(original_vertex_count - vertex_count),
            "compression_ratio": compression_ratio,
            "straight_distance": _max_point_line_distance(points),
            "endpoint_distance": _endpoint_distance(points),
            "is_simplified": bool(arc.get("simplified", False)),
            "touches_border": bool(len(semantic_faces) == 1 or EXTERIOR_FACE_ID in incident_faces),
        },
        "metadata": {
            "method": arc.get("method"),
            "owner_face_id": arc.get("owner_face_id"),
            "owner_distance": arc.get("owner_distance"),
            "regularized": arc.get("regularized"),
            "regularization_method": arc.get("regularization_method"),
        },
    }


def _atom_features(atom: Dict[str, object], *, face_area: float, eps: float) -> Dict[str, object]:
    polygon = Polygon(atom.get("outer", []))
    area = float(polygon.area) if not polygon.is_empty else float(atom.get("area", 0.0))
    perimeter = float(polygon.length) if not polygon.is_empty else 0.0
    minx, miny, maxx, maxy = polygon.bounds if not polygon.is_empty else (0.0, 0.0, 0.0, 0.0)
    bbox_aspect = _aspect_ratio(maxx - minx, maxy - miny, eps=eps)
    obb_w, obb_h = _oriented_bbox_dimensions(polygon)
    return {
        "area_ratio_in_face": float(area / face_area) if face_area > eps else 0.0,
        "bbox_aspect_ratio": float(bbox_aspect),
        "oriented_aspect_ratio": float(_aspect_ratio(obb_w, obb_h, eps=eps)),
        "compactness": _compactness(area, perimeter, eps=eps),
    }


def _convex_partition_for_face(
    global_payload: Dict[str, object],
    face: Dict[str, object],
    *,
    config: ExplanationEvidenceConfig,
) -> Dict[str, object]:
    geometry_payload = {
        "source_partition_graph": global_payload.get("source_partition_graph"),
        "face_id": int(face["id"]),
        "label": int(face["label"]),
        "bbox": face.get("bbox", []),
        "approx_geometry": {
            "outer": face.get("outer", []),
            "holes": face.get("holes", []),
        },
    }
    bridged_config = BridgedPartitionConfig(
        max_bridge_sets=int(config.convex_max_bridge_sets),
        area_eps=float(config.convex_area_eps),
        validity_eps=float(config.convex_validity_eps),
        backend=str(config.convex_backend),
        cgal_cli=config.convex_cgal_cli,
        cut_slit_scale=float(config.convex_cut_slit_scale),
    )
    try:
        payload = build_bridged_convex_partition_from_geometry_payload(
            geometry_payload,
            config=bridged_config,
            source_tag=global_payload.get("source_partition_graph"),
        )
    except Exception as exc:
        return {
            "backend": "failed",
            "valid": False,
            "piece_count": 0,
            "triangle_count": None,
            "atoms": [],
            "validation": {"is_valid": False},
            "backend_info": {},
            "failure_reason": str(exc),
        }

    face_area = float(payload.get("validation", {}).get("original_area", 0.0))
    atoms = []
    triangle_count = 0
    for atom in payload.get("primitives", []):
        atom_payload = dict(atom)
        atom_payload["features"] = _atom_features(atom_payload, face_area=face_area, eps=config.min_area_eps)
        if atom_payload.get("type") == "triangle":
            triangle_count += 1
        atoms.append(atom_payload)
    backend_info = dict(payload.get("backend_info") or {})
    return {
        "backend": backend_info.get("backend", "unknown"),
        "valid": bool(payload.get("validation", {}).get("is_valid", False)),
        "piece_count": int(len(atoms)),
        "triangle_count": int(triangle_count),
        "atoms": atoms,
        "validation": payload.get("validation", {}),
        "backend_info": backend_info,
    }


def _face_features(
    face: Dict[str, object],
    polygon: Polygon,
    *,
    image_area: float,
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
    border_face_ids: set[int],
    config: ExplanationEvidenceConfig,
) -> Dict[str, object]:
    area = float(polygon.area) if not polygon.is_empty else float(face.get("approx_area", 0.0))
    perimeter = float(polygon.length) if not polygon.is_empty else 0.0
    centroid = polygon.centroid if not polygon.is_empty else None
    minx, miny, maxx, maxy = polygon.bounds if not polygon.is_empty else (0.0, 0.0, 0.0, 0.0)
    bbox_width = float(maxx - minx)
    bbox_height = float(maxy - miny)
    convex_hull_area = float(polygon.convex_hull.area) if not polygon.is_empty else 0.0
    obb_width, obb_height = _oriented_bbox_dimensions(polygon)
    oriented_aspect = _aspect_ratio(obb_width, obb_height, eps=config.min_area_eps)
    compactness = _compactness(area, perimeter, eps=config.min_area_eps)
    shared_boundary_length = float(sum(float(item.get("shared_length", 0.0)) for item in adjacency_by_face.get(int(face["id"]), [])))
    return {
        "area": area,
        "area_ratio": float(area / image_area) if image_area > config.min_area_eps else 0.0,
        "centroid": [float(centroid.x), float(centroid.y)] if centroid is not None else [0.0, 0.0],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": float(bbox_width * bbox_height),
        "bbox_aspect_ratio": _aspect_ratio(bbox_width, bbox_height, eps=config.min_area_eps),
        "perimeter": perimeter,
        "compactness": compactness,
        "convex_hull_area": convex_hull_area,
        "solidity": float(area / convex_hull_area) if convex_hull_area > config.min_area_eps else 0.0,
        "oriented_bbox_width": float(obb_width),
        "oriented_bbox_height": float(obb_height),
        "oriented_aspect_ratio": float(oriented_aspect),
        "degree": int(len(adjacency_by_face.get(int(face["id"]), []))),
        "shared_boundary_length": shared_boundary_length,
        "touches_border": bool(int(face["id"]) in border_face_ids),
        "hole_count": int(len(face.get("holes", []))),
        "is_thin": bool(oriented_aspect >= config.thin_aspect_ratio),
        "is_compact": bool(compactness >= config.compactness_threshold),
    }


def build_explanation_evidence_payload(
    global_payload: Dict[str, object],
    *,
    config: ExplanationEvidenceConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or ExplanationEvidenceConfig()
    size = [int(value) for value in global_payload.get("size", [0, 0])]
    image_area = float(size[0] * size[1]) if len(size) == 2 else 0.0

    raw_faces = list(global_payload.get("faces", []))
    polygons_by_face = {int(face["id"]): _polygon_from_face(face) for face in raw_faces}
    arcs = [_arc_evidence(arc) for arc in global_payload.get("arcs", [])]
    labels_by_face = {int(face["id"]): int(face.get("label", -1)) for face in raw_faces}
    adjacency = _face_adjacency_from_arcs(arcs, labels_by_face)
    if not adjacency:
        adjacency = _face_adjacency_from_geometry(polygons_by_face, labels_by_face, eps=config.validity_eps)

    adjacency_by_face: Dict[int, List[Dict[str, object]]] = {}
    for item in adjacency:
        left, right = [int(value) for value in item["faces"]]
        adjacency_by_face.setdefault(left, []).append(item)
        adjacency_by_face.setdefault(right, []).append(item)

    border_face_ids = {
        int(face_id)
        for arc in arcs
        if arc["is_border"]
        for face_id in arc["incident_faces"]
        if int(face_id) != EXTERIOR_FACE_ID
    }
    if not arcs and len(size) == 2:
        height, width = size
        for face_id, polygon in polygons_by_face.items():
            if polygon.is_empty:
                continue
            minx, miny, maxx, maxy = polygon.bounds
            if minx <= config.validity_eps or miny <= config.validity_eps or maxx >= width - config.validity_eps or maxy >= height - config.validity_eps:
                border_face_ids.add(int(face_id))

    faces = []
    invalid_face_ids: List[int] = []
    convex_failure_face_ids: List[int] = []
    missing_convex_atoms_face_ids: List[int] = []
    label_histogram: Dict[str, int] = {}

    for face in raw_faces:
        face_id = int(face["id"])
        label = int(face.get("label", -1))
        label_histogram[str(label)] = label_histogram.get(str(label), 0) + 1
        polygon = polygons_by_face[face_id]
        if polygon.is_empty or not polygon.is_valid:
            invalid_face_ids.append(face_id)
        convex_partition = _convex_partition_for_face(global_payload, face, config=config)
        if not convex_partition.get("valid", False):
            convex_failure_face_ids.append(face_id)
        if not convex_partition.get("atoms"):
            missing_convex_atoms_face_ids.append(face_id)
        features = _face_features(
            face,
            polygon,
            image_area=image_area,
            adjacency_by_face=adjacency_by_face,
            border_face_ids=border_face_ids,
            config=config,
        )
        faces.append(
            {
                "id": face_id,
                "label": label,
                "bbox": [float(value) for value in face.get("bbox", [])],
                "outer_arc_refs": face.get("outer_arc_refs", []),
                "hole_arc_refs": face.get("hole_arc_refs", []),
                "geometry": {
                    "outer": _polygon_outer(polygon) if not polygon.is_empty else face.get("outer", []),
                    "holes": _polygon_holes(polygon) if not polygon.is_empty else face.get("holes", []),
                },
                "features": features,
                "convex_partition": convex_partition,
                "source": {
                    "approx_area": float(face.get("approx_area", features["area"])),
                    "is_valid": bool(face.get("is_valid", polygon.is_valid and not polygon.is_empty)),
                },
            }
        )

    total_face_area = float(sum(face["features"]["area"] for face in faces))
    convex_success_count = int(sum(1 for face in faces if face["convex_partition"].get("valid", False)))
    convex_failure_count = int(len(faces) - convex_success_count)
    global_validation = global_payload.get("validation", {})
    global_partition_valid = bool(global_validation.get("is_valid", False))
    evidence_validation = {
        "is_valid": bool(global_partition_valid and not invalid_face_ids and convex_failure_count == 0),
        "usable_for_explainer": bool(global_partition_valid and len(faces) > 0 and convex_success_count > 0),
        "global_partition_valid": global_partition_valid,
        "face_count": int(len(faces)),
        "arc_count": int(len(arcs)),
        "adjacency_count": int(len(adjacency)),
        "convex_success_count": convex_success_count,
        "convex_failure_count": convex_failure_count,
        "invalid_face_ids": invalid_face_ids,
        "convex_failure_face_ids": convex_failure_face_ids,
        "missing_convex_atoms_face_ids": missing_convex_atoms_face_ids,
    }

    statistics = {
        "image_area": image_area,
        "total_face_area": total_face_area,
        "mean_face_area": float(total_face_area / len(faces)) if faces else 0.0,
        "mean_face_degree": float(sum(face["features"]["degree"] for face in faces) / len(faces)) if faces else 0.0,
        "mean_face_convex_piece_count": float(sum(face["convex_partition"]["piece_count"] for face in faces) / len(faces)) if faces else 0.0,
        "total_convex_atom_count": int(sum(face["convex_partition"]["piece_count"] for face in faces)),
        "mean_arc_length": float(sum(arc["features"]["length"] for arc in arcs) / len(arcs)) if arcs else 0.0,
        "shared_arc_count": int(sum(1 for arc in arcs if arc["is_shared"])),
        "border_arc_count": int(sum(1 for arc in arcs if arc["is_border"])),
        "thin_face_count": int(sum(1 for face in faces if face["features"]["is_thin"])),
        "compact_face_count": int(sum(1 for face in faces if face["features"]["is_compact"])),
        "label_histogram": label_histogram,
    }

    return {
        "format": "maskgen_explanation_evidence_v1",
        "source_global_approx": source_tag,
        "source_partition_graph": global_payload.get("source_partition_graph"),
        "source_mask": global_payload.get("source_mask"),
        "size": size,
        "config": asdict(config),
        "faces": faces,
        "arcs": arcs,
        "adjacency": adjacency,
        "global_validation": global_validation,
        "evidence_validation": evidence_validation,
        "statistics": statistics,
    }
