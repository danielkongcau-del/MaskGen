from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from partition_gen.manual_parse_graph_target_audit import load_json
from partition_gen.manual_parse_graph_visualization import polygon_world_rings


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    index = int(math.ceil(float(percentile) * len(sorted_values))) - 1
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _numeric_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "p90": _percentile(floats, 0.90),
        "p95": _percentile(floats, 0.95),
        "max": float(max(floats)),
    }


def _bbox_from_points(points: Sequence[Sequence[float]]) -> list[float] | None:
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _flatten_polygon_points(rings: Iterable[tuple[list[list[float]], list[list[list[float]]]]]) -> list[list[float]]:
    points: list[list[float]] = []
    for outer, holes in rings:
        points.extend(outer)
        for hole in holes:
            points.extend(hole)
    return points


def _bbox_area(bbox: Sequence[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_intersection(left: Sequence[float], right: Sequence[float]) -> list[float] | None:
    min_x = max(float(left[0]), float(right[0]))
    min_y = max(float(left[1]), float(right[1]))
    max_x = min(float(left[2]), float(right[2]))
    max_y = min(float(left[3]), float(right[3]))
    if max_x <= min_x or max_y <= min_y:
        return None
    return [float(min_x), float(min_y), float(max_x), float(max_y)]


def _bbox_union(values: Sequence[Sequence[float]]) -> list[float] | None:
    boxes = [bbox for bbox in values if bbox is not None]
    if not boxes:
        return None
    return [
        float(min(float(bbox[0]) for bbox in boxes)),
        float(min(float(bbox[1]) for bbox in boxes)),
        float(max(float(bbox[2]) for bbox in boxes)),
        float(max(float(bbox[3]) for bbox in boxes)),
    ]


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _point_in_bbox(point: tuple[float, float], bbox: Sequence[float]) -> bool:
    return float(bbox[0]) <= point[0] <= float(bbox[2]) and float(bbox[1]) <= point[1] <= float(bbox[3])


def _bbox_gap(left: Sequence[float], right: Sequence[float]) -> float:
    dx = max(float(left[0]) - float(right[2]), float(right[0]) - float(left[2]), 0.0)
    dy = max(float(left[1]) - float(right[3]), float(right[1]) - float(left[3]), 0.0)
    return float(math.hypot(dx, dy))


def _node_world_bboxes(target: dict) -> dict[str, dict]:
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    rows: dict[str, dict] = {}
    contains_children: dict[str, list[str]] = defaultdict(list)

    for relation in relations:
        if str(relation.get("type", "")) != "contains":
            continue
        parent = relation.get("parent")
        child = relation.get("child")
        if parent is not None and child is not None:
            contains_children[str(parent)].append(str(child))

    for node in nodes:
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        bbox = None
        if str(node.get("geometry_model", "none")) == "polygon_code":
            bbox = _bbox_from_points(_flatten_polygon_points(polygon_world_rings(node)))
        rows[node_id] = {
            "node_id": node_id,
            "role": str(node.get("role", "")),
            "label": int(node.get("label", 0)),
            "geometry_model": str(node.get("geometry_model", "none")),
            "bbox": None if bbox is None else [float(value) for value in bbox],
            "bbox_source": "geometry" if bbox is not None else None,
        }

    changed = True
    while changed:
        changed = False
        for parent, children in contains_children.items():
            if parent not in rows or rows[parent].get("bbox") is not None:
                continue
            child_boxes = [rows[child]["bbox"] for child in children if child in rows and rows[child].get("bbox") is not None]
            union = _bbox_union(child_boxes)
            if union is not None:
                rows[parent]["bbox"] = union
                rows[parent]["bbox_source"] = "contains_children"
                changed = True

    for row in rows.values():
        bbox = row.get("bbox")
        row["bbox_area"] = float(_bbox_area(bbox))
        row["bbox_center"] = None if bbox is None else [float(value) for value in _bbox_center(bbox)]
    return rows


def _relation_pairs(relation: dict) -> list[tuple[str, str, str]]:
    relation_type = str(relation.get("type", ""))
    if relation_type == "inserted_in":
        obj = relation.get("object")
        container = relation.get("container", relation.get("support"))
        return [(str(obj), str(container), "object_container")] if obj is not None and container is not None else []
    if relation_type == "contains":
        parent = relation.get("parent")
        child = relation.get("child")
        return [(str(child), str(parent), "child_parent")] if child is not None and parent is not None else []
    if relation_type == "divides":
        divider = relation.get("divider")
        target = relation.get("target", relation.get("support"))
        return [(str(divider), str(target), "divider_target")] if divider is not None and target is not None else []
    if relation_type == "adjacent_to":
        faces = [str(value) for value in relation.get("faces", []) or []]
        return [(left, right, "face_pair") for left, right in itertools.combinations(faces, 2)]
    return []


def _base_relation_row(
    *,
    source: str | None,
    target: dict,
    relation_index: int,
    relation: dict,
    left_id: str,
    right_id: str,
    endpoint_kind: str,
) -> dict:
    metadata = target.get("metadata", {}) or {}
    return {
        "source": source,
        "sample_index": metadata.get("sample_index"),
        "relation_index": int(relation_index),
        "relation_type": str(relation.get("type", "")),
        "relation_id": relation.get("id"),
        "endpoint_kind": endpoint_kind,
        "left_id": str(left_id),
        "right_id": str(right_id),
    }


def _evaluate_pair(
    row: dict,
    *,
    left: dict | None,
    right: dict | None,
    min_containment_ratio: float,
    max_adjacent_gap: float,
    max_adjacent_overlap_ratio: float,
    min_divider_target_intersection_ratio: float,
    max_divider_target_area_ratio: float,
) -> dict:
    relation_type = str(row["relation_type"])
    reasons: list[str] = []
    if left is None:
        reasons.append("missing_left_node")
    if right is None:
        reasons.append("missing_right_node")
    left_bbox = None if left is None else left.get("bbox")
    right_bbox = None if right is None else right.get("bbox")
    if left_bbox is None:
        reasons.append("missing_left_bbox")
    if right_bbox is None:
        reasons.append("missing_right_bbox")
    if reasons:
        return {
            **row,
            "evaluable": False,
            "passed": False,
            "failure_reasons": reasons,
            "severity": 1.0,
        }

    left_area = _bbox_area(left_bbox)
    right_area = _bbox_area(right_bbox)
    intersection = _bbox_intersection(left_bbox, right_bbox)
    intersection_area = _bbox_area(intersection)
    left_intersection_ratio = intersection_area / left_area if left_area > 0.0 else 0.0
    right_intersection_ratio = intersection_area / right_area if right_area > 0.0 else 0.0
    smaller_intersection_ratio = intersection_area / min(left_area, right_area) if min(left_area, right_area) > 0.0 else 0.0
    gap = _bbox_gap(left_bbox, right_bbox)
    left_center_in_right = _point_in_bbox(_bbox_center(left_bbox), right_bbox)
    right_center_in_left = _point_in_bbox(_bbox_center(right_bbox), left_bbox)

    passed = True
    severity = 0.0
    if relation_type in {"inserted_in", "contains"}:
        passed = bool(left_intersection_ratio >= float(min_containment_ratio) and left_center_in_right)
        if not left_center_in_right:
            reasons.append("subject_center_outside_container")
        if left_intersection_ratio < float(min_containment_ratio):
            reasons.append("low_subject_container_overlap")
        severity = max(0.0, 1.0 - float(left_intersection_ratio)) + (0.5 if not left_center_in_right else 0.0)
    elif relation_type == "adjacent_to":
        passed = bool(gap <= float(max_adjacent_gap) and smaller_intersection_ratio <= float(max_adjacent_overlap_ratio))
        if gap > float(max_adjacent_gap):
            reasons.append("adjacent_gap_too_large")
        if smaller_intersection_ratio > float(max_adjacent_overlap_ratio):
            reasons.append("adjacent_overlap_too_large")
        severity = float(smaller_intersection_ratio) + min(1.0, float(gap) / max(float(max_adjacent_gap), 1e-8))
    elif relation_type == "divides":
        area_ratio = left_area / right_area if right_area > 0.0 else math.inf
        passed = bool(
            (left_intersection_ratio >= float(min_divider_target_intersection_ratio) or left_center_in_right)
            and area_ratio <= float(max_divider_target_area_ratio)
        )
        if left_intersection_ratio < float(min_divider_target_intersection_ratio) and not left_center_in_right:
            reasons.append("divider_misses_target")
        if area_ratio > float(max_divider_target_area_ratio):
            reasons.append("divider_area_too_large")
        severity = max(0.0, 1.0 - float(left_intersection_ratio)) + max(0.0, area_ratio - float(max_divider_target_area_ratio))
    else:
        passed = True

    return {
        **row,
        "evaluable": True,
        "passed": bool(passed),
        "failure_reasons": [] if passed else reasons,
        "severity": float(severity),
        "left_role": None if left is None else left.get("role"),
        "right_role": None if right is None else right.get("role"),
        "left_bbox": [float(value) for value in left_bbox],
        "right_bbox": [float(value) for value in right_bbox],
        "left_bbox_source": None if left is None else left.get("bbox_source"),
        "right_bbox_source": None if right is None else right.get("bbox_source"),
        "left_bbox_area": float(left_area),
        "right_bbox_area": float(right_area),
        "intersection_area": float(intersection_area),
        "left_intersection_ratio": float(left_intersection_ratio),
        "right_intersection_ratio": float(right_intersection_ratio),
        "smaller_intersection_ratio": float(smaller_intersection_ratio),
        "bbox_gap": float(gap),
        "left_center_in_right": bool(left_center_in_right),
        "right_center_in_left": bool(right_center_in_left),
    }


def audit_manual_parse_graph_target_relation_spatial(
    target: dict,
    *,
    source: str | None = None,
    min_containment_ratio: float = 0.8,
    max_adjacent_gap: float = 4.0,
    max_adjacent_overlap_ratio: float = 0.1,
    min_divider_target_intersection_ratio: float = 0.3,
    max_divider_target_area_ratio: float = 1.25,
) -> dict:
    graph = target.get("parse_graph", {}) or {}
    relations = list(graph.get("relations", []) or [])
    node_rows = _node_world_bboxes(target)
    relation_rows: list[dict] = []
    relation_type_histogram = Counter()

    for relation_index, relation in enumerate(relations):
        relation_type = str(relation.get("type", ""))
        pairs = _relation_pairs(relation)
        if not pairs:
            relation_type_histogram[relation_type] += 1
            continue
        for left_id, right_id, endpoint_kind in pairs:
            relation_type_histogram[relation_type] += 1
            base_row = _base_relation_row(
                source=source,
                target=target,
                relation_index=int(relation_index),
                relation=relation,
                left_id=left_id,
                right_id=right_id,
                endpoint_kind=endpoint_kind,
            )
            relation_rows.append(
                _evaluate_pair(
                    base_row,
                    left=node_rows.get(left_id),
                    right=node_rows.get(right_id),
                    min_containment_ratio=float(min_containment_ratio),
                    max_adjacent_gap=float(max_adjacent_gap),
                    max_adjacent_overlap_ratio=float(max_adjacent_overlap_ratio),
                    min_divider_target_intersection_ratio=float(min_divider_target_intersection_ratio),
                    max_divider_target_area_ratio=float(max_divider_target_area_ratio),
                )
            )

    failure_rows = [row for row in relation_rows if not bool(row.get("passed", False))]
    evaluable_rows = [row for row in relation_rows if bool(row.get("evaluable", False))]
    return {
        "source": source,
        "sample_index": (target.get("metadata", {}) or {}).get("sample_index"),
        "relation_count": int(len(relations)),
        "relation_pair_count": int(len(relation_rows)),
        "evaluable_relation_pair_count": int(len(evaluable_rows)),
        "passed_relation_pair_count": int(sum(1 for row in relation_rows if row.get("passed"))),
        "failed_relation_pair_count": int(len(failure_rows)),
        "relation_type_histogram": dict(sorted(relation_type_histogram.items())),
        "relations": relation_rows,
    }


def audit_manual_parse_graph_targets_relation_spatial(
    paths: Sequence[Path],
    *,
    min_containment_ratio: float = 0.8,
    max_adjacent_gap: float = 4.0,
    max_adjacent_overlap_ratio: float = 0.1,
    min_divider_target_intersection_ratio: float = 0.3,
    max_divider_target_area_ratio: float = 1.25,
    top_k: int = 20,
) -> dict:
    rows: list[dict] = []
    load_errors: list[dict] = []
    all_relation_rows: list[dict] = []
    for path in paths:
        try:
            target = load_json(Path(path))
            row = audit_manual_parse_graph_target_relation_spatial(
                target,
                source=str(Path(path).as_posix()),
                min_containment_ratio=float(min_containment_ratio),
                max_adjacent_gap=float(max_adjacent_gap),
                max_adjacent_overlap_ratio=float(max_adjacent_overlap_ratio),
                min_divider_target_intersection_ratio=float(min_divider_target_intersection_ratio),
                max_divider_target_area_ratio=float(max_divider_target_area_ratio),
            )
            rows.append(row)
            all_relation_rows.extend(row["relations"])
        except Exception as exc:
            load_errors.append({"source": str(Path(path).as_posix()), "error": f"{type(exc).__name__}:{exc}"})

    relation_type_histogram = Counter()
    pass_by_type = Counter()
    fail_by_type = Counter()
    failure_reasons = Counter()
    role_pair_failures = Counter()
    for row in all_relation_rows:
        relation_type = str(row.get("relation_type", ""))
        relation_type_histogram[relation_type] += 1
        if row.get("passed"):
            pass_by_type[relation_type] += 1
        else:
            fail_by_type[relation_type] += 1
            failure_reasons.update(row.get("failure_reasons", []) or [])
            role_pair_failures[f"{row.get('left_role')}->{row.get('right_role')}"] += 1

    relation_type_metrics = {}
    for relation_type in sorted(relation_type_histogram):
        typed_rows = [row for row in all_relation_rows if str(row.get("relation_type", "")) == relation_type]
        typed_evaluable = [row for row in typed_rows if bool(row.get("evaluable", False))]
        relation_type_metrics[relation_type] = {
            "count": int(len(typed_rows)),
            "evaluable_count": int(len(typed_evaluable)),
            "passed_count": int(pass_by_type[relation_type]),
            "failed_count": int(fail_by_type[relation_type]),
            "pass_ratio": float(pass_by_type[relation_type] / len(typed_rows)) if typed_rows else 0.0,
            "left_intersection_ratio_stats": _numeric_stats(
                [float(row["left_intersection_ratio"]) for row in typed_evaluable]
            ),
            "smaller_intersection_ratio_stats": _numeric_stats(
                [float(row["smaller_intersection_ratio"]) for row in typed_evaluable]
            ),
            "bbox_gap_stats": _numeric_stats([float(row["bbox_gap"]) for row in typed_evaluable]),
            "severity_stats": _numeric_stats([float(row["severity"]) for row in typed_rows]),
        }

    failure_rows = [row for row in all_relation_rows if not bool(row.get("passed", False))]
    failure_rows.sort(key=lambda row: (-float(row.get("severity", 0.0)), str(row.get("source", "")), int(row.get("relation_index", 0))))

    return {
        "format": "maskgen_manual_parse_graph_relation_spatial_audit_v1",
        "input_path_count": int(len(paths)),
        "loaded_count": int(len(rows)),
        "load_error_count": int(len(load_errors)),
        "relation_pair_count": int(len(all_relation_rows)),
        "evaluable_relation_pair_count": int(sum(1 for row in all_relation_rows if row.get("evaluable"))),
        "passed_relation_pair_count": int(sum(1 for row in all_relation_rows if row.get("passed"))),
        "failed_relation_pair_count": int(len(failure_rows)),
        "pass_ratio": float(
            sum(1 for row in all_relation_rows if row.get("passed")) / len(all_relation_rows)
        )
        if all_relation_rows
        else 0.0,
        "thresholds": {
            "min_containment_ratio": float(min_containment_ratio),
            "max_adjacent_gap": float(max_adjacent_gap),
            "max_adjacent_overlap_ratio": float(max_adjacent_overlap_ratio),
            "min_divider_target_intersection_ratio": float(min_divider_target_intersection_ratio),
            "max_divider_target_area_ratio": float(max_divider_target_area_ratio),
        },
        "relation_type_histogram": dict(sorted(relation_type_histogram.items())),
        "relation_type_metrics": relation_type_metrics,
        "failure_reason_histogram": dict(failure_reasons.most_common()),
        "role_pair_failure_histogram": dict(role_pair_failures.most_common()),
        "top_failures": failure_rows[: int(top_k)],
        "load_errors": load_errors,
        "rows": rows,
    }
