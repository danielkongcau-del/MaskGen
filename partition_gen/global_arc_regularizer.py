from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from shapely.geometry import LineString, Polygon

from partition_gen.dual_graph import load_json
from partition_gen.global_approx_partition import (
    EXTERIOR_FACE_ID,
    GlobalApproxConfig,
    _face_payloads_from_arcs,
    _line_length,
    _points_close,
    validate_global_approx_partition,
)


Point = Tuple[float, float]


@dataclass(frozen=True)
class GlobalArcRegularizationConfig:
    simplify_tolerance: float = 1.25
    max_distance: float = 1.25
    min_vertex_reduction: int = 1
    min_arc_length: float = 4.0
    enable_subsegment_smoothing: bool = True
    max_subsegment_span: int = 64
    max_candidates_per_arc: int = 64
    enable_face_chain_smoothing: bool = False
    face_chain_max_distance: float = 2.0
    face_chain_min_length: float = 6.0
    face_chain_max_span: int = 96
    max_face_chain_candidates: int = 256
    enable_strip_face_smoothing: bool = False
    strip_min_aspect_ratio: float = 4.0
    strip_max_width: float = 14.0
    strip_min_length: float = 10.0
    strip_max_area_ratio: float = 0.35
    max_strip_face_candidates: int = 64
    allow_polyline_smoothing: bool = False
    include_exterior_arcs: bool = False
    validity_eps: float = 1e-6


def _point_list(points: Sequence[Sequence[float]]) -> List[Point]:
    return [(float(point[0]), float(point[1])) for point in points]


def _dedupe_points(points: Sequence[Point], *, eps: float = 1e-9) -> List[Point]:
    output: List[Point] = []
    for point in points:
        if output and _points_close(output[-1], point, eps=eps):
            continue
        output.append((float(point[0]), float(point[1])))
    return output


def _point_key(point: Point, digits: int = 8) -> Tuple[float, float]:
    return (round(float(point[0]), digits), round(float(point[1]), digits))


def _trim_collinear_points(points: Sequence[Point], *, eps: float = 1e-7) -> List[Point]:
    deduped = _dedupe_points(points, eps=eps)
    if len(deduped) <= 2:
        return deduped
    output = [deduped[0]]
    for index in range(1, len(deduped) - 1):
        prev_point = output[-1]
        point = deduped[index]
        next_point = deduped[index + 1]
        cross = (point[0] - prev_point[0]) * (next_point[1] - point[1]) - (point[1] - prev_point[1]) * (next_point[0] - point[0])
        base = max(math.hypot(next_point[0] - prev_point[0], next_point[1] - prev_point[1]), 1e-12)
        if abs(cross) / base <= eps:
            continue
        output.append(point)
    output.append(deduped[-1])
    return output


def _hausdorff_distance(left: Sequence[Point], right: Sequence[Point]) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("inf")
    left_line = LineString(left)
    right_line = LineString(right)
    return float(max(left_line.hausdorff_distance(right_line), right_line.hausdorff_distance(left_line)))


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    denominator = math.hypot(dx, dy)
    if denominator <= 1e-12:
        return math.hypot(float(point[0]) - float(start[0]), float(point[1]) - float(start[1]))
    return abs(dy * float(point[0]) - dx * float(point[1]) + float(end[0]) * float(start[1]) - float(end[1]) * float(start[0])) / denominator


def _project_point_to_segment_line(point: Point, start: Point, end: Point) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return (float(start[0]), float(start[1]))
    t = ((float(point[0]) - float(start[0])) * dx + (float(point[1]) - float(start[1])) * dy) / denominator
    return (float(start[0]) + t * dx, float(start[1]) + t * dy)


def _nearest_point_on_segment(point: Point, start: Point, end: Point) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return (float(start[0]), float(start[1]))
    t = ((float(point[0]) - float(start[0])) * dx + (float(point[1]) - float(start[1])) * dy) / denominator
    t = max(0.0, min(1.0, t))
    return (float(start[0]) + t * dx, float(start[1]) + t * dy)


def _nearest_point_on_polyline(point: Point, ring: Sequence[Point]) -> Point:
    best: Tuple[float, Point] | None = None
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        projected = _nearest_point_on_segment(point, start, end)
        distance = math.hypot(float(point[0]) - projected[0], float(point[1]) - projected[1])
        if best is None or distance < best[0]:
            best = (distance, projected)
    return best[1] if best is not None else (float(point[0]), float(point[1]))


def _straight_segment_distance(points: Sequence[Point], start_index: int, end_index: int) -> float:
    start = points[start_index]
    end = points[end_index]
    return float(max(_point_line_distance(point, start, end) for point in points[start_index : end_index + 1]))


def _regularize_arc_points(points: Sequence[Point], *, config: GlobalArcRegularizationConfig) -> Tuple[List[Point], float] | None:
    if len(points) < 3:
        return None
    if _line_length(points) < config.min_arc_length:
        return None
    if _points_close(points[0], points[-1], eps=config.validity_eps):
        return None

    straight = [(float(points[0][0]), float(points[0][1])), (float(points[-1][0]), float(points[-1][1]))]
    straight_distance = _hausdorff_distance(points, straight)
    if straight_distance <= float(config.max_distance):
        vertex_reduction = len(points) - len(straight)
        if vertex_reduction >= int(config.min_vertex_reduction):
            return straight, float(straight_distance)

    if not config.allow_polyline_smoothing:
        return None

    simplified = list(LineString(points).simplify(float(config.simplify_tolerance), preserve_topology=False).coords)
    simplified = [(float(x), float(y)) for x, y in simplified]
    if len(simplified) < 2:
        return None

    simplified[0] = (float(points[0][0]), float(points[0][1]))
    simplified[-1] = (float(points[-1][0]), float(points[-1][1]))
    simplified = _dedupe_points(simplified)
    if len(simplified) < 2:
        return None

    vertex_reduction = len(points) - len(simplified)
    if vertex_reduction < int(config.min_vertex_reduction):
        return None

    distance = _hausdorff_distance(points, simplified)
    if distance > float(config.max_distance):
        return None
    return simplified, float(distance)


def _replacement_candidate(
    *,
    arc_id: int,
    points: Sequence[Point],
    start_index: int,
    end_index: int,
    replacement_points: Sequence[Point],
    distance: float,
    mode: str,
) -> Dict[str, object]:
    new_points = list(points[:start_index]) + list(replacement_points) + list(points[end_index + 1 :])
    new_points = _dedupe_points(new_points)
    return {
        "arc_id": int(arc_id),
        "start_point": [float(points[start_index][0]), float(points[start_index][1])],
        "end_point": [float(points[end_index][0]), float(points[end_index][1])],
        "replacement_points": [[float(x), float(y)] for x, y in replacement_points],
        "points": [[float(x), float(y)] for x, y in new_points],
        "vertex_count": int(len(new_points)),
        "length": float(_line_length(new_points)),
        "vertex_reduction": int(len(points) - len(new_points)),
        "distance": float(distance),
        "mode": mode,
        "start_index": int(start_index),
        "end_index": int(end_index),
    }


def _subsegment_regularization_candidates(
    arc: Dict[str, object],
    points: Sequence[Point],
    *,
    config: GlobalArcRegularizationConfig,
) -> List[Dict[str, object]]:
    if not config.enable_subsegment_smoothing or len(points) < 4:
        return []

    arc_id = int(arc["id"])
    candidates: List[Dict[str, object]] = []
    seen: set[Tuple[int, int]] = set()
    max_span = max(3, int(config.max_subsegment_span))
    min_end_delta = max(2, int(config.min_vertex_reduction) + 1)

    for start_index in range(0, len(points) - min_end_delta):
        best: Tuple[int, float] | None = None
        max_end = min(len(points) - 1, start_index + max_span)
        for end_index in range(start_index + min_end_delta, max_end + 1):
            if _line_length(points[start_index : end_index + 1]) < float(config.min_arc_length):
                continue
            distance = _straight_segment_distance(points, start_index, end_index)
            if distance > float(config.max_distance):
                continue
            best = (end_index, distance)
        if best is None:
            continue
        end_index, distance = best
        if (start_index, end_index) in seen:
            continue
        seen.add((start_index, end_index))
        replacement = [points[start_index], points[end_index]]
        candidate = _replacement_candidate(
            arc_id=arc_id,
            points=points,
            start_index=start_index,
            end_index=end_index,
            replacement_points=replacement,
            distance=distance,
            mode="subsegment_straightening",
        )
        if int(candidate["vertex_reduction"]) >= int(config.min_vertex_reduction):
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-int(item["vertex_reduction"]), float(item["distance"]), int(item["start_index"])))
    return candidates[: max(1, int(config.max_candidates_per_arc))]


def _is_regularizable_arc(arc: Dict[str, object], *, config: GlobalArcRegularizationConfig) -> bool:
    incident_faces = [int(face_id) for face_id in arc.get("incident_faces", [])]
    semantic_faces = [face_id for face_id in incident_faces if face_id != EXTERIOR_FACE_ID]
    if len(semantic_faces) < 2 and not config.include_exterior_arcs:
        return False
    return True


def _arc_regularization_candidates(
    arcs: Sequence[Dict[str, object]],
    *,
    config: GlobalArcRegularizationConfig,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for arc in arcs:
        if not _is_regularizable_arc(arc, config=config):
            continue
        points = _point_list(arc.get("points", []))
        result = _regularize_arc_points(points, config=config)
        if result is not None:
            simplified, distance = result
            candidates.append(
                _replacement_candidate(
                    arc_id=int(arc["id"]),
                    points=points,
                    start_index=0,
                    end_index=len(points) - 1,
                    replacement_points=simplified,
                    distance=distance,
                    mode="whole_arc_smoothing",
                )
            )
        candidates.extend(_subsegment_regularization_candidates(arc, points, config=config))
    candidates.sort(
        key=lambda item: (
            -int(item["vertex_reduction"]),
            float(item["distance"]),
            int(item["arc_id"]),
            int(item.get("start_index", 0)),
        )
    )
    return candidates


def _find_point_index(points: Sequence[Point], target: Sequence[float], *, start_index: int = 0, eps: float = 1e-9) -> int | None:
    target_point = (float(target[0]), float(target[1]))
    for index in range(max(0, int(start_index)), len(points)):
        if _points_close(points[index], target_point, eps=eps):
            return index
    return None


def _apply_regularization_candidate(arc: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object] | None:
    output = dict(arc)
    previous_method = output.get("method")
    previous_vertex_count = int(output.get("vertex_count", len(output.get("points", []))))
    current_points = _point_list(output.get("points", []))
    start_index = _find_point_index(current_points, candidate["start_point"], eps=1e-7)
    if start_index is None:
        return None
    end_index = _find_point_index(current_points, candidate["end_point"], start_index=start_index + 1, eps=1e-7)
    if end_index is None:
        return None
    replacement_points = _point_list(candidate["replacement_points"])
    points = current_points[:start_index] + replacement_points + current_points[end_index + 1 :]
    points = _trim_collinear_points(points)
    actual_vertex_reduction = previous_vertex_count - len(points)
    if actual_vertex_reduction <= 0:
        return None
    output.update(
        {
            "points": [[float(x), float(y)] for x, y in points],
            "vertex_count": int(len(points)),
            "length": float(_line_length(points)),
            "regularized": True,
            "regularization_method": str(candidate.get("mode", "rdp_arc_smoothing")),
            "pre_regularization_method": previous_method,
            "pre_regularization_vertex_count": previous_vertex_count,
            "regularization_vertex_reduction": int(actual_vertex_reduction),
            "regularization_distance": float(candidate["distance"]),
        }
    )
    return output


def _face_chain_candidates(
    faces: Sequence[Dict[str, object]],
    *,
    config: GlobalArcRegularizationConfig,
) -> List[Dict[str, object]]:
    if not config.enable_face_chain_smoothing:
        return []
    candidates: List[Dict[str, object]] = []
    for face in faces:
        rings = [("outer", -1, _point_list(face.get("outer", [])))]
        rings.extend(("hole", index, _point_list(ring)) for index, ring in enumerate(face.get("holes", [])))
        for ring_role, ring_index, ring in rings:
            if len(ring) < 4:
                continue
            max_span = min(max(3, int(config.face_chain_max_span)), len(ring) - 1)
            for start_index in range(0, len(ring) - 2):
                best: Tuple[int, float] | None = None
                for end_index in range(start_index + 2, min(len(ring) - 1, start_index + max_span) + 1):
                    segment = ring[start_index : end_index + 1]
                    if _line_length(segment) < float(config.face_chain_min_length):
                        continue
                    distance = _straight_segment_distance(ring, start_index, end_index)
                    if distance > float(config.face_chain_max_distance):
                        continue
                    best = (end_index, distance)
                if best is None:
                    continue
                end_index, distance = best
                segment = ring[start_index : end_index + 1]
                replacement = [_project_point_to_segment_line(point, segment[0], segment[-1]) for point in segment]
                moved_points = sum(1 for before, after in zip(segment, replacement) if not _points_close(before, after, eps=1e-7))
                if moved_points < int(config.min_vertex_reduction):
                    continue
                candidates.append(
                    {
                        "face_id": int(face["id"]),
                        "label": int(face["label"]),
                        "ring_role": str(ring_role),
                        "ring_index": int(ring_index),
                        "start_index": int(start_index),
                        "end_index": int(end_index),
                        "point_updates": [
                            {
                                "from": [float(before[0]), float(before[1])],
                                "to": [float(after[0]), float(after[1])],
                            }
                            for before, after in zip(segment, replacement)
                        ],
                        "moved_point_count": int(moved_points),
                        "distance": float(distance),
                        "length": float(_line_length(segment)),
                        "mode": "face_chain_straightening",
                    }
                )
    candidates.sort(
        key=lambda item: (
            -int(item["moved_point_count"]),
            float(item["distance"]),
            -float(item["length"]),
            int(item["face_id"]),
            str(item.get("ring_role", "")),
            int(item["start_index"]),
        )
    )
    return candidates[: max(1, int(config.max_face_chain_candidates))]


def _apply_face_chain_candidate(
    arcs: Sequence[Dict[str, object]],
    candidate: Dict[str, object],
) -> Tuple[List[Dict[str, object]] | None, Dict[str, object]]:
    update_map = {
        _point_key((float(item["from"][0]), float(item["from"][1]))): (float(item["to"][0]), float(item["to"][1]))
        for item in candidate.get("point_updates", [])
    }
    changed_arc_ids: List[int] = []
    before_vertices = 0
    after_vertices = 0
    next_arcs: List[Dict[str, object]] = []
    for arc in arcs:
        points = _point_list(arc.get("points", []))
        changed = False
        updated_points: List[Point] = []
        for point in points:
            replacement = update_map.get(_point_key(point))
            if replacement is not None:
                updated_points.append(replacement)
                if not _points_close(point, replacement, eps=1e-7):
                    changed = True
            else:
                updated_points.append(point)
        if not changed:
            next_arcs.append(dict(arc))
            continue
        trimmed = _trim_collinear_points(updated_points)
        if len(trimmed) < 2:
            return None, {
                "reason": "face-chain update collapsed an arc",
                "changed_arc_ids": changed_arc_ids,
            }
        updated_arc = dict(arc)
        previous_vertex_count = int(updated_arc.get("vertex_count", len(points)))
        before_vertices += previous_vertex_count
        after_vertices += len(trimmed)
        updated_arc.update(
            {
                "points": [[float(x), float(y)] for x, y in trimmed],
                "vertex_count": int(len(trimmed)),
                "length": float(_line_length(trimmed)),
                "regularized": True,
                "regularization_method": "face_chain_straightening",
                "pre_regularization_method": updated_arc.get("method"),
                "pre_regularization_vertex_count": previous_vertex_count,
                "regularization_vertex_reduction": int(previous_vertex_count - len(trimmed)),
                "regularization_distance": float(candidate.get("distance", candidate.get("max_move", 0.0))),
            }
        )
        changed_arc_ids.append(int(updated_arc["id"]))
        next_arcs.append(updated_arc)
    if not changed_arc_ids:
        return None, {"reason": "face-chain candidate no longer changes any arc", "changed_arc_ids": []}
    return next_arcs, {
        "reason": None,
        "changed_arc_ids": changed_arc_ids,
        "before_vertex_count": int(before_vertices),
        "after_vertex_count": int(after_vertices),
        "vertex_reduction": int(before_vertices - after_vertices),
    }


def _polygon_from_face_payload(face: Dict[str, object]) -> Polygon:
    outer = _point_list(face.get("outer", []))
    holes = [_point_list(ring) for ring in face.get("holes", [])]
    if len(outer) < 3:
        return Polygon()
    try:
        return Polygon(outer, holes)
    except ValueError:
        return Polygon()


def _rotated_rectangle_vertices(polygon: Polygon) -> List[Point]:
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or not isinstance(rectangle, Polygon):
        return []
    return _dedupe_points(_point_list(list(rectangle.exterior.coords)[:-1]))


def _rectangle_side_lengths(rectangle: Sequence[Point]) -> List[float]:
    return [
        math.hypot(rectangle[(index + 1) % len(rectangle)][0] - rectangle[index][0], rectangle[(index + 1) % len(rectangle)][1] - rectangle[index][1])
        for index in range(len(rectangle))
    ]


def _strip_face_candidates(
    faces: Sequence[Dict[str, object]],
    *,
    config: GlobalArcRegularizationConfig,
) -> List[Dict[str, object]]:
    if not config.enable_strip_face_smoothing:
        return []
    candidates: List[Dict[str, object]] = []
    for face in faces:
        if face.get("holes"):
            continue
        ring = _point_list(face.get("outer", []))
        if len(ring) < 5:
            continue
        polygon = _polygon_from_face_payload(face)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-6:
            continue
        rectangle = _rotated_rectangle_vertices(polygon)
        if len(rectangle) != 4:
            continue
        side_lengths = _rectangle_side_lengths(rectangle)
        long_side = max(side_lengths)
        short_side = min(side_lengths)
        if short_side <= 1e-6:
            continue
        aspect_ratio = float(long_side / short_side)
        if aspect_ratio < float(config.strip_min_aspect_ratio):
            continue
        if short_side > float(config.strip_max_width):
            continue
        if long_side < float(config.strip_min_length):
            continue
        image_area = 1.0
        bbox = face.get("bbox", [])
        if len(bbox) == 4:
            # Approximate local scale guard; the full image size is not stored per face here.
            bbox_area = max(1.0, float(abs((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))))
            if float(polygon.area) / bbox_area > float(config.strip_max_area_ratio) and aspect_ratio < 8.0:
                continue
            image_area = bbox_area
        point_updates = []
        total_move = 0.0
        moved_point_count = 0
        for point in ring:
            projected = _nearest_point_on_polyline(point, rectangle)
            distance = math.hypot(point[0] - projected[0], point[1] - projected[1])
            total_move += distance
            if distance > 1e-7:
                moved_point_count += 1
            point_updates.append({"from": [float(point[0]), float(point[1])], "to": [float(projected[0]), float(projected[1])]})
        if moved_point_count < int(config.min_vertex_reduction):
            continue
        max_move = max(
            math.hypot(float(item["from"][0]) - float(item["to"][0]), float(item["from"][1]) - float(item["to"][1]))
            for item in point_updates
        )
        candidates.append(
            {
                "face_id": int(face["id"]),
                "label": int(face["label"]),
                "aspect_ratio": float(aspect_ratio),
                "width": float(short_side),
                "length": float(long_side),
                "area": float(polygon.area),
                "bbox_area": float(image_area),
                "moved_point_count": int(moved_point_count),
                "mean_move": float(total_move / max(1, len(ring))),
                "max_move": float(max_move),
                "rectangle": [[float(x), float(y)] for x, y in rectangle],
                "point_updates": point_updates,
                "mode": "strip_face_rectangle_projection",
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["aspect_ratio"]),
            float(item["width"]),
            -int(item["moved_point_count"]),
            int(item["face_id"]),
        )
    )
    return candidates[: max(1, int(config.max_strip_face_candidates))]


def _load_source_graph(global_payload: Dict[str, object]) -> Dict[str, object]:
    source = global_payload.get("source_partition_graph")
    if not source:
        raise ValueError("global payload has no source_partition_graph; pass graph_data explicitly")
    return load_json(Path(str(source)))


def regularize_global_arc_payload(
    global_payload: Dict[str, object],
    graph_data: Dict[str, object] | None = None,
    *,
    config: GlobalArcRegularizationConfig | None = None,
) -> Dict[str, object]:
    config = config or GlobalArcRegularizationConfig()
    graph_data = graph_data or _load_source_graph(global_payload)

    output = deepcopy(global_payload)
    arcs = [dict(arc) for arc in output.get("arcs", [])]
    if not arcs:
        raise ValueError("global payload has no arcs")

    candidates = _arc_regularization_candidates(arcs, config=config)
    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    face_chain_accepted: List[Dict[str, object]] = []
    face_chain_rejected: List[Dict[str, object]] = []
    strip_face_accepted: List[Dict[str, object]] = []
    strip_face_rejected: List[Dict[str, object]] = []
    arc_index_by_id = {int(arc["id"]): index for index, arc in enumerate(arcs)}
    validation_config = GlobalApproxConfig(validity_eps=float(config.validity_eps))
    faces = _face_payloads_from_arcs(graph_data, arcs)
    validation = validate_global_approx_partition(graph_data, arcs, faces, config=validation_config)
    if not validation["is_valid"]:
        raise ValueError("input global payload cannot be validated against its source partition graph")

    for candidate in candidates:
        arc_id = int(candidate["arc_id"])
        arc_index = arc_index_by_id[arc_id]
        previous_arc = arcs[arc_index]
        next_arcs = [dict(arc) for arc in arcs]
        applied_arc = _apply_regularization_candidate(previous_arc, candidate)
        if applied_arc is None:
            rejected.append(
                {
                    "arc_id": int(arc_id),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                    "reason": "candidate endpoints no longer available",
                    "mode": str(candidate.get("mode", "")),
                }
            )
            continue
        next_arcs[arc_index] = applied_arc
        next_faces = _face_payloads_from_arcs(graph_data, next_arcs)
        next_validation = validate_global_approx_partition(graph_data, next_arcs, next_faces, config=validation_config)
        if next_validation["is_valid"]:
            arcs = next_arcs
            faces = next_faces
            validation = next_validation
            accepted.append(
                {
                    "arc_id": int(arc_id),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                    "mode": str(candidate.get("mode", "")),
                    "before_vertex_count": int(previous_arc.get("vertex_count", 0)),
                    "after_vertex_count": int(applied_arc["vertex_count"]),
                }
            )
        else:
            rejected.append(
                {
                    "arc_id": int(arc_id),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                    "mode": str(candidate.get("mode", "")),
                    "reason": "global validation failed",
                    "union_iou": float(next_validation["union_iou"]),
                    "overlap_area": float(next_validation["overlap_area"]),
                    "missing_adjacency": next_validation["missing_adjacency"],
                    "extra_adjacency": next_validation["extra_adjacency"],
                }
            )

    face_chain_candidates = _face_chain_candidates(faces, config=config)
    for candidate in face_chain_candidates:
        next_arcs, apply_info = _apply_face_chain_candidate(arcs, candidate)
        if next_arcs is None:
            face_chain_rejected.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "distance": float(candidate["distance"]),
                    "mode": str(candidate["mode"]),
                    "reason": str(apply_info["reason"]),
                    "changed_arc_ids": apply_info.get("changed_arc_ids", []),
                }
            )
            continue
        next_faces = _face_payloads_from_arcs(graph_data, next_arcs)
        next_validation = validate_global_approx_partition(graph_data, next_arcs, next_faces, config=validation_config)
        if next_validation["is_valid"]:
            arcs = next_arcs
            faces = next_faces
            validation = next_validation
            face_chain_accepted.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "distance": float(candidate["distance"]),
                    "length": float(candidate["length"]),
                    "mode": str(candidate["mode"]),
                    "changed_arc_ids": [int(value) for value in apply_info["changed_arc_ids"]],
                    "before_vertex_count": int(apply_info["before_vertex_count"]),
                    "after_vertex_count": int(apply_info["after_vertex_count"]),
                    "vertex_reduction": int(apply_info["vertex_reduction"]),
                }
            )
        else:
            face_chain_rejected.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "distance": float(candidate["distance"]),
                    "length": float(candidate["length"]),
                    "mode": str(candidate["mode"]),
                    "reason": "global validation failed",
                    "changed_arc_ids": [int(value) for value in apply_info["changed_arc_ids"]],
                    "union_iou": float(next_validation["union_iou"]),
                    "overlap_area": float(next_validation["overlap_area"]),
                    "missing_adjacency": next_validation["missing_adjacency"],
                    "extra_adjacency": next_validation["extra_adjacency"],
                }
            )

    strip_face_candidates = _strip_face_candidates(faces, config=config)
    for candidate in strip_face_candidates:
        next_arcs, apply_info = _apply_face_chain_candidate(arcs, candidate)
        if next_arcs is None:
            strip_face_rejected.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "aspect_ratio": float(candidate["aspect_ratio"]),
                    "width": float(candidate["width"]),
                    "length": float(candidate["length"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "mean_move": float(candidate["mean_move"]),
                    "max_move": float(candidate["max_move"]),
                    "mode": str(candidate["mode"]),
                    "reason": str(apply_info["reason"]),
                    "changed_arc_ids": apply_info.get("changed_arc_ids", []),
                }
            )
            continue
        next_faces = _face_payloads_from_arcs(graph_data, next_arcs)
        next_validation = validate_global_approx_partition(graph_data, next_arcs, next_faces, config=validation_config)
        if next_validation["is_valid"]:
            arcs = next_arcs
            faces = next_faces
            validation = next_validation
            strip_face_accepted.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "aspect_ratio": float(candidate["aspect_ratio"]),
                    "width": float(candidate["width"]),
                    "length": float(candidate["length"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "mean_move": float(candidate["mean_move"]),
                    "max_move": float(candidate["max_move"]),
                    "mode": str(candidate["mode"]),
                    "changed_arc_ids": [int(value) for value in apply_info["changed_arc_ids"]],
                    "before_vertex_count": int(apply_info["before_vertex_count"]),
                    "after_vertex_count": int(apply_info["after_vertex_count"]),
                    "vertex_reduction": int(apply_info["vertex_reduction"]),
                }
            )
        else:
            strip_face_rejected.append(
                {
                    "face_id": int(candidate["face_id"]),
                    "label": int(candidate["label"]),
                    "aspect_ratio": float(candidate["aspect_ratio"]),
                    "width": float(candidate["width"]),
                    "length": float(candidate["length"]),
                    "moved_point_count": int(candidate["moved_point_count"]),
                    "mean_move": float(candidate["mean_move"]),
                    "max_move": float(candidate["max_move"]),
                    "mode": str(candidate["mode"]),
                    "reason": "global validation failed",
                    "changed_arc_ids": [int(value) for value in apply_info["changed_arc_ids"]],
                    "union_iou": float(next_validation["union_iou"]),
                    "overlap_area": float(next_validation["overlap_area"]),
                    "missing_adjacency": next_validation["missing_adjacency"],
                    "extra_adjacency": next_validation["extra_adjacency"],
                }
            )

    output["format"] = f"{global_payload.get('format', 'global_owner_approx_partition_v1')}_arc_regularized"
    output["source_format"] = global_payload.get("format")
    output["arcs"] = arcs
    output["faces"] = faces
    output["validation"] = validation
    output["arc_regularization"] = {
        "policy": "rdp_arc_smoothing_v1",
        "config": asdict(config),
        "candidate_count": int(len(candidates) + len(face_chain_candidates) + len(strip_face_candidates)),
        "accepted_count": int(len(accepted) + len(face_chain_accepted) + len(strip_face_accepted)),
        "rejected_count": int(len(rejected) + len(face_chain_rejected) + len(strip_face_rejected)),
        "arc_candidate_count": int(len(candidates)),
        "arc_accepted_count": int(len(accepted)),
        "arc_rejected_count": int(len(rejected)),
        "face_chain_candidate_count": int(len(face_chain_candidates)),
        "face_chain_accepted_count": int(len(face_chain_accepted)),
        "face_chain_rejected_count": int(len(face_chain_rejected)),
        "strip_face_candidate_count": int(len(strip_face_candidates)),
        "strip_face_accepted_count": int(len(strip_face_accepted)),
        "strip_face_rejected_count": int(len(strip_face_rejected)),
        "accepted": accepted,
        "rejected": rejected[:64],
        "face_chain_accepted": face_chain_accepted,
        "face_chain_rejected": face_chain_rejected[:64],
        "strip_face_accepted": strip_face_accepted,
        "strip_face_rejected": strip_face_rejected[:64],
        "input_arc_vertex_count": int(sum(int(arc.get("vertex_count", 0)) for arc in global_payload.get("arcs", []))),
        "output_arc_vertex_count": int(sum(int(arc.get("vertex_count", 0)) for arc in arcs)),
        "vertex_reduction": int(
            sum(int(arc.get("vertex_count", 0)) for arc in global_payload.get("arcs", []))
            - sum(int(arc.get("vertex_count", 0)) for arc in arcs)
        ),
    }
    return output
