from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


@dataclass(frozen=True)
class MicroFaceAbsorptionConfig:
    max_area: float = 4.0
    max_vertices: int = 4
    min_shared_length: float = 0.5
    labels: tuple[int, ...] = ()
    absorb_small_islands: bool = False
    island_max_area: float = 4.0
    island_min_shared_length: float = 0.5
    island_labels: tuple[int, ...] = ()
    area_eps: float = 1e-8
    validity_eps: float = 1e-6


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


def _polygon_from_face(face: Dict[str, object]) -> Polygon:
    outer = face.get("outer", [])
    if len(outer) < 3:
        return Polygon()
    try:
        polygon = Polygon(outer, face.get("holes", []))
    except ValueError:
        return Polygon()
    return orient(polygon, sign=1.0) if not polygon.is_empty else polygon


def _trim_ring(points: Sequence[Sequence[float]]) -> List[List[float]]:
    output = [[float(point[0]), float(point[1])] for point in points]
    if len(output) >= 2 and output[0] == output[-1]:
        output = output[:-1]
    return output


def _face_payload_from_polygon(face: Dict[str, object], polygon: Polygon) -> Dict[str, object]:
    polygon = orient(polygon, sign=1.0)
    outer = _trim_ring(list(polygon.exterior.coords))
    holes = [_trim_ring(list(ring.coords)) for ring in polygon.interiors if len(ring.coords) >= 4]
    minx, miny, maxx, maxy = polygon.bounds
    centroid = polygon.centroid
    output = {
        "id": int(face["id"]),
        "label": int(face["label"]),
        "area": float(polygon.area),
        "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
        "outer": outer,
        "holes": holes,
        "approx_area": float(polygon.area),
        "centroid": [float(centroid.x), float(centroid.y)],
        "is_valid": bool(polygon.is_valid and not polygon.is_empty),
    }
    absorbed_ids = list(face.get("absorbed_face_ids", []))
    if absorbed_ids:
        output["absorbed_face_ids"] = [int(value) for value in absorbed_ids]
    return output


def _vertex_count(face: Dict[str, object]) -> int:
    return int(len(face.get("outer", [])) + sum(len(ring) for ring in face.get("holes", [])))


def _shared_lengths_from_arcs(global_payload: Dict[str, object]) -> Dict[int, Dict[int, float]]:
    output: Dict[int, Dict[int, float]] = {}
    for arc in global_payload.get("arcs", []):
        faces = [int(face_id) for face_id in arc.get("incident_faces", []) if int(face_id) >= 0]
        if len(faces) != 2:
            continue
        left, right = faces
        length = float(arc.get("length", 0.0))
        output.setdefault(left, {})[right] = output.setdefault(left, {}).get(right, 0.0) + length
        output.setdefault(right, {})[left] = output.setdefault(right, {}).get(left, 0.0) + length
    return output


def _is_micro_face(face: Dict[str, object], polygon: Polygon, *, config: MicroFaceAbsorptionConfig) -> bool:
    if polygon.is_empty or not polygon.is_valid:
        return False
    if face.get("holes"):
        return False
    if config.labels and int(face["label"]) not in set(int(value) for value in config.labels):
        return False
    if float(polygon.area) > float(config.max_area):
        return False
    if _vertex_count(face) > int(config.max_vertices):
        return False
    return True


def _is_small_island_face(
    face: Dict[str, object],
    polygon: Polygon,
    *,
    shared_lengths: Dict[int, Dict[int, float]],
    config: MicroFaceAbsorptionConfig,
) -> bool:
    if not config.absorb_small_islands:
        return False
    if polygon.is_empty or not polygon.is_valid:
        return False
    if face.get("holes"):
        return False
    if config.island_labels and int(face["label"]) not in set(int(value) for value in config.island_labels):
        return False
    if float(polygon.area) > float(config.island_max_area):
        return False
    neighbors = [
        neighbor_id
        for neighbor_id, shared_length in shared_lengths.get(int(face["id"]), {}).items()
        if float(shared_length) >= float(config.island_min_shared_length)
    ]
    return len(neighbors) == 1


def _target_for_micro_face(
    face_id: int,
    *,
    active_ids: set[int],
    micro_ids: set[int],
    shared_lengths: Dict[int, Dict[int, float]],
    config: MicroFaceAbsorptionConfig,
) -> Tuple[int | None, float]:
    candidates = [
        (neighbor_id, shared_length)
        for neighbor_id, shared_length in shared_lengths.get(int(face_id), {}).items()
        if int(neighbor_id) in active_ids and int(neighbor_id) not in micro_ids and float(shared_length) >= float(config.min_shared_length)
    ]
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda item: (-float(item[1]), int(item[0])))
    return int(candidates[0][0]), float(candidates[0][1])


def _validate_faces(faces: Sequence[Dict[str, object]], reference_union: Polygon, *, config: MicroFaceAbsorptionConfig) -> Dict[str, object]:
    polygons = [_polygon_from_face(face) for face in faces]
    valid_polygons = [polygon for polygon in polygons if not polygon.is_empty]
    all_faces_valid = all(polygon.is_valid and not polygon.is_empty for polygon in polygons)
    total_area = float(sum(polygon.area for polygon in valid_polygons))
    union = unary_union(valid_polygons) if valid_polygons else Polygon()
    union_area = float(union.area) if not union.is_empty else 0.0
    overlap_area = max(0.0, total_area - union_area)
    iou = 0.0
    if not union.is_empty and not reference_union.is_empty:
        merged_union = union.union(reference_union)
        union_denominator = float(merged_union.area)
        if union_denominator > 0:
            iou = float(union.intersection(reference_union).area / union_denominator)
    return {
        "is_valid": bool(all_faces_valid and overlap_area <= config.validity_eps and iou >= 1.0 - config.validity_eps),
        "all_faces_valid": bool(all_faces_valid),
        "face_count": int(len(faces)),
        "union_iou": float(iou),
        "total_face_area": float(total_area),
        "union_area": float(union_area),
        "overlap_area": float(overlap_area),
        "reference_union_area": float(reference_union.area) if not reference_union.is_empty else 0.0,
    }


def absorb_micro_faces_from_global_payload(
    global_payload: Dict[str, object],
    *,
    config: MicroFaceAbsorptionConfig | None = None,
) -> Dict[str, object]:
    config = config or MicroFaceAbsorptionConfig()
    faces_by_id = {int(face["id"]): dict(face) for face in global_payload.get("faces", [])}
    polygons_by_id = {face_id: _polygon_from_face(face) for face_id, face in faces_by_id.items()}
    reference_union = unary_union([polygon for polygon in polygons_by_id.values() if not polygon.is_empty])
    shared_lengths = _shared_lengths_from_arcs(global_payload)

    micro_ids = {
        face_id
        for face_id, face in faces_by_id.items()
        if _is_micro_face(face, polygons_by_id[face_id], config=config)
    }
    island_ids = {
        face_id
        for face_id, face in faces_by_id.items()
        if _is_small_island_face(face, polygons_by_id[face_id], shared_lengths=shared_lengths, config=config)
    }
    candidate_ids = set(micro_ids) | set(island_ids)
    active_ids = set(faces_by_id)
    absorbed: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    for face_id in sorted(candidate_ids, key=lambda value: (float(polygons_by_id[value].area), int(value))):
        if face_id not in active_ids:
            continue
        target_id, shared_length = _target_for_micro_face(
            face_id,
            active_ids=active_ids,
            micro_ids=candidate_ids,
            shared_lengths=shared_lengths,
            config=config,
        )
        if target_id is None:
            rejected.append({"face_id": int(face_id), "reason": "no non-micro adjacent target"})
            continue
        merged = polygons_by_id[target_id].union(polygons_by_id[face_id])
        pieces = [polygon for polygon in _iter_polygons(merged) if polygon.area > config.area_eps]
        if len(pieces) != 1:
            rejected.append(
                {
                    "face_id": int(face_id),
                    "target_face_id": int(target_id),
                    "reason": "merge produced multiple polygons",
                    "piece_count": int(len(pieces)),
                }
            )
            continue
        merged_polygon = pieces[0]
        if not merged_polygon.is_valid or merged_polygon.is_empty:
            rejected.append({"face_id": int(face_id), "target_face_id": int(target_id), "reason": "merged polygon invalid"})
            continue

        target_face = dict(faces_by_id[target_id])
        absorbed_ids = [int(value) for value in target_face.get("absorbed_face_ids", [])]
        absorbed_ids.append(int(face_id))
        target_face["absorbed_face_ids"] = absorbed_ids
        faces_by_id[target_id] = target_face
        polygons_by_id[target_id] = merged_polygon
        active_ids.remove(face_id)
        absorbed.append(
            {
                "face_id": int(face_id),
                "label": int(faces_by_id[face_id]["label"]),
                "area": float(polygons_by_id[face_id].area),
                "vertex_count": int(_vertex_count(faces_by_id[face_id])),
                "reason": "micro_face" if face_id in micro_ids else "small_island",
                "is_micro_face": bool(face_id in micro_ids),
                "is_small_island": bool(face_id in island_ids),
                "target_face_id": int(target_id),
                "target_label": int(faces_by_id[target_id]["label"]),
                "shared_length": float(shared_length),
            }
        )

    output_faces = [
        _face_payload_from_polygon(faces_by_id[face_id], polygons_by_id[face_id])
        for face_id in sorted(active_ids)
    ]
    validation = _validate_faces(output_faces, reference_union, config=config)
    return {
        "format": "micro_face_absorbed_global_faces_v1",
        "source_format": global_payload.get("format"),
        "source_partition_graph": global_payload.get("source_partition_graph"),
        "source_mask": global_payload.get("source_mask"),
        "size": global_payload.get("size"),
        "config": asdict(config),
        "faces": output_faces,
        "micro_face_absorption": {
            "policy": "dominant_adjacent_face_absorption_v1",
            "candidate_count": int(len(candidate_ids)),
            "micro_candidate_count": int(len(micro_ids)),
            "small_island_candidate_count": int(len(island_ids)),
            "absorbed_count": int(len(absorbed)),
            "rejected_count": int(len(rejected)),
            "input_face_count": int(len(faces_by_id)),
            "output_face_count": int(len(output_faces)),
            "absorbed": absorbed,
            "rejected": rejected[:128],
        },
        "validation": validation,
    }
