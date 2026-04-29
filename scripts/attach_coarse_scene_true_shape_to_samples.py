from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_coarse_scene_ar import decode_coarse_scene_tokens_to_target  # noqa: E402
from partition_gen.manual_geometry_shape_fallback import (  # noqa: E402
    build_geometry_shape_fallback_library,
    geometry_target_from_fallback_shape,
    geometry_target_quality,
    select_fallback_geometry_shape,
)
from partition_gen.manual_layout_residual import geometry_local_bbox  # noqa: E402
from partition_gen.manual_layout_retrieval import write_jsonl  # noqa: E402
from partition_gen.manual_topology_placeholder_geometry import iter_jsonl  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach split true-shape fallbacks to coarse-scene sampled frames.")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL rows containing coarse scene tokens.")
    parser.add_argument("--library-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--min-true-shape-world-bbox-area", type=float, default=1.0)
    parser.add_argument("--min-true-shape-local-bbox-side", type=float, default=1e-6)
    parser.add_argument("--scale-fit-mode", type=str, default="cover", choices=["cover", "contain", "frame"])
    parser.add_argument("--disable-adjacent-frame-repair", action="store_true")
    parser.add_argument("--enable-legacy-adjacent-frame-repair", action="store_true")
    parser.add_argument("--adjacent-repair-iterations", type=int, default=5)
    parser.add_argument("--adjacent-repair-damping", type=float, default=0.8)
    parser.add_argument("--adjacent-repair-max-gap", type=float, default=4.0)
    parser.add_argument("--adjacent-repair-max-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "max": float(max(floats)),
    }


def _bbox_metrics(bbox: list[float]) -> dict:
    width = max(1e-6, float(bbox[2]) - float(bbox[0]))
    height = max(1e-6, float(bbox[3]) - float(bbox[1]))
    return {
        "min_x": float(bbox[0]),
        "min_y": float(bbox[1]),
        "max_x": float(bbox[2]),
        "max_y": float(bbox[3]),
        "width": float(width),
        "height": float(height),
        "center_x": float((float(bbox[0]) + float(bbox[2])) / 2.0),
        "center_y": float((float(bbox[1]) + float(bbox[3])) / 2.0),
    }


def _bbox_gap(left: list[float], right: list[float]) -> float:
    dx = max(float(right[0]) - float(left[2]), float(left[0]) - float(right[2]), 0.0)
    dy = max(float(right[1]) - float(left[3]), float(left[1]) - float(right[3]), 0.0)
    return float(math.hypot(dx, dy))


def _bbox_intersection(left: list[float], right: list[float]) -> list[float] | None:
    min_x = max(float(left[0]), float(right[0]))
    min_y = max(float(left[1]), float(right[1]))
    max_x = min(float(left[2]), float(right[2]))
    max_y = min(float(left[3]), float(right[3]))
    if max_x <= min_x or max_y <= min_y:
        return None
    return [float(min_x), float(min_y), float(max_x), float(max_y)]


def _bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_overlap_ratio(left: list[float], right: list[float]) -> float:
    inter = _bbox_intersection(left, right)
    return float(_bbox_area(inter) / max(1e-6, min(_bbox_area(left), _bbox_area(right))))


def _shift_bbox(bbox: list[float], dx: float, dy: float) -> list[float]:
    return [float(bbox[0]) + float(dx), float(bbox[1]) + float(dy), float(bbox[2]) + float(dx), float(bbox[3]) + float(dy)]


def _bbox_from_local_bbox(frame: dict, local_bbox: dict) -> list[float] | None:
    width = abs(float(local_bbox.get("width", 0.0)))
    height = abs(float(local_bbox.get("height", 0.0)))
    if width <= 1e-6 or height <= 1e-6:
        return None
    min_x = float(local_bbox.get("min_x", -width / 2.0))
    min_y = float(local_bbox.get("min_y", -height / 2.0))
    corners = [
        (min_x, min_y),
        (min_x + width, min_y),
        (min_x + width, min_y + height),
        (min_x, min_y + height),
    ]
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    origin_x, origin_y = float(origin[0]), float(origin[1])
    scale = max(1e-6, float(frame.get("scale", 1.0)))
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    world: list[tuple[float, float]] = []
    for local_x, local_y in corners:
        x = float(local_x) * scale
        y = float(local_y) * scale
        world.append((origin_x + x * cos_theta - y * sin_theta, origin_y + x * sin_theta + y * cos_theta))
    xs = [point[0] for point in world]
    ys = [point[1] for point in world]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _node_repair_bbox(node: dict) -> list[float] | None:
    local_bbox = node.get("true_shape_local_bbox")
    if isinstance(local_bbox, dict):
        bbox = _bbox_from_local_bbox(node.get("frame", {}) or {}, local_bbox)
        if bbox is not None:
            return bbox
    coarse_bbox = node.get("coarse_bbox")
    if isinstance(coarse_bbox, list) and len(coarse_bbox) == 4:
        return [float(value) for value in coarse_bbox]
    return None


def _shift_node_layout(node: dict, dx: float, dy: float) -> None:
    frame = copy.deepcopy(node.get("frame", {}) or {})
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    frame["origin"] = [float(origin[0]) + float(dx), float(origin[1]) + float(dy)]
    node["frame"] = frame
    coarse_bbox = node.get("coarse_bbox")
    if isinstance(coarse_bbox, list) and len(coarse_bbox) == 4:
        node["coarse_bbox"] = [
            float(coarse_bbox[0]) + float(dx),
            float(coarse_bbox[1]) + float(dy),
            float(coarse_bbox[2]) + float(dx),
            float(coarse_bbox[3]) + float(dy),
        ]


def _adjacent_side(child_bbox: list[float], anchor_bbox: list[float]) -> int:
    child = _bbox_metrics(child_bbox)
    anchor = _bbox_metrics(anchor_bbox)
    dx = child["center_x"] - anchor["center_x"]
    dy = child["center_y"] - anchor["center_y"]
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0.0 else 1
    return 2 if dy >= 0.0 else 3


def _adjacent_repair_children(relations: list[dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}

    def add(parent: object, child: object) -> None:
        if parent is None or child is None:
            return
        parent_id = str(parent)
        child_id = str(child)
        if parent_id == child_id:
            return
        children.setdefault(parent_id, []).append(child_id)

    for relation in relations:
        relation_type = str(relation.get("type", ""))
        if relation_type == "inserted_in":
            add(relation.get("container", relation.get("support")), relation.get("object"))
        elif relation_type == "contains":
            add(relation.get("parent"), relation.get("child"))
        elif relation_type == "divides":
            add(relation.get("target", relation.get("support")), relation.get("divider"))
        elif relation_type == "adjacent_to":
            faces = [str(value) for value in relation.get("faces", []) or []]
            if len(faces) >= 2:
                add(faces[0], faces[1])
    return {key: list(dict.fromkeys(values)) for key, values in children.items()}


def _adjacent_pairs(relations: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for relation in relations:
        if str(relation.get("type", "")) != "adjacent_to":
            continue
        faces = [str(value) for value in relation.get("faces", []) or []]
        if len(faces) >= 2:
            pairs.append((faces[0], faces[1]))
    return pairs


def _subtree_node_ids(root_id: str, children: dict[str, list[str]]) -> set[str]:
    output: set[str] = set()
    stack = [str(root_id)]
    while stack:
        node_id = stack.pop()
        if node_id in output:
            continue
        output.add(node_id)
        stack.extend(children.get(node_id, []))
    return output


def _shifted_bbox_map(bboxes: dict[str, list[float]], shifted_ids: set[str], dx: float, dy: float) -> dict[str, list[float]]:
    output = dict(bboxes)
    for node_id in shifted_ids:
        if node_id in output:
            output[node_id] = _shift_bbox(output[node_id], dx, dy)
    return output


def _adjacent_pair_score(
    left: list[float],
    right: list[float],
    *,
    max_gap: float,
    max_overlap_ratio: float,
) -> tuple[int, int, float]:
    gap_value = _bbox_gap(left, right)
    overlap = _bbox_overlap_ratio(left, right)
    overlap_fail = int(overlap > max_overlap_ratio)
    gap_fail = int(gap_value > max_gap)
    penalty = max(0.0, overlap - max_overlap_ratio) * 100.0 + max(0.0, gap_value - max_gap)
    return overlap_fail, gap_fail, float(penalty)


def _affected_constraint_score(
    bboxes: dict[str, list[float]],
    nodes: list[dict],
    adjacent_pairs: list[tuple[str, str]],
    affected_node_ids: set[str],
    *,
    max_gap: float,
    max_overlap_ratio: float,
) -> tuple[int, int, float]:
    overlap_failures = 0
    gap_failures = 0
    penalty = 0.0
    for left_id, right_id in adjacent_pairs:
        if left_id not in affected_node_ids and right_id not in affected_node_ids:
            continue
        left_bbox = bboxes.get(left_id)
        right_bbox = bboxes.get(right_id)
        if left_bbox is None or right_bbox is None:
            continue
        overlap_fail, gap_fail, pair_penalty = _adjacent_pair_score(
            left_bbox,
            right_bbox,
            max_gap=max_gap,
            max_overlap_ratio=max_overlap_ratio,
        )
        overlap_failures += overlap_fail
        gap_failures += gap_fail
        penalty += pair_penalty

    movable_support_ids = [
        str(node.get("id"))
        for node in nodes
        if str(node.get("id")) in affected_node_ids and str(node.get("role", "")) in {"support_region", "residual_region"}
    ]
    peer_support_ids = [
        str(node.get("id"))
        for node in nodes
        if str(node.get("id")) not in affected_node_ids and str(node.get("role", "")) in {"support_region", "residual_region"}
    ]
    for left_id in movable_support_ids:
        left_bbox = bboxes.get(left_id)
        if left_bbox is None:
            continue
        for right_id in peer_support_ids:
            right_bbox = bboxes.get(right_id)
            if right_bbox is None:
                continue
            overlap = _bbox_overlap_ratio(left_bbox, right_bbox)
            if overlap > max_overlap_ratio:
                overlap_failures += 1
                penalty += (overlap - max_overlap_ratio) * 100.0
    return int(overlap_failures), int(gap_failures), float(penalty)


def _adjacent_candidate_deltas(
    child_bbox: list[float],
    anchor_bbox: list[float],
    *,
    child_coarse: list[float],
    anchor_coarse: list[float],
    gap: float,
) -> list[tuple[float, float]]:
    child = _bbox_metrics(child_bbox)
    anchor = _bbox_metrics(anchor_bbox)
    side = _adjacent_side(child_coarse, anchor_coarse)

    def cross_centers(axis: str) -> list[float]:
        if axis == "y":
            span = min(child["height"], anchor["height"])
            low = anchor["min_y"] + span / 2.0
            high = anchor["max_y"] - span / 2.0
            decoded = child["center_y"]
        else:
            span = min(child["width"], anchor["width"])
            low = anchor["min_x"] + span / 2.0
            high = anchor["max_x"] - span / 2.0
            decoded = child["center_x"]
        if high < low:
            low = high = (low + high) / 2.0
        values = [
            max(low, min(high, decoded)),
            (low + high) / 2.0,
            low,
            high,
            low + (high - low) * 0.25,
            low + (high - low) * 0.75,
        ]
        return list(dict.fromkeys(float(value) for value in values))

    def target_center(candidate_side: int, cross_center: float) -> tuple[float, float]:
        if candidate_side == 0:
            return anchor["max_x"] + float(gap) + child["width"] / 2.0, cross_center
        if candidate_side == 1:
            return anchor["min_x"] - float(gap) - child["width"] / 2.0, cross_center
        if candidate_side == 2:
            return cross_center, anchor["max_y"] + float(gap) + child["height"] / 2.0
        return cross_center, anchor["min_y"] - float(gap) - child["height"] / 2.0

    deltas: list[tuple[float, float]] = []
    side_order = [side, *[candidate for candidate in (0, 1, 2, 3) if candidate != side]]
    for candidate_side in side_order:
        axis = "y" if candidate_side in {0, 1} else "x"
        for cross_center in cross_centers(axis):
            target_x, target_y = target_center(candidate_side, cross_center)
            deltas.append((float(target_x - child["center_x"]), float(target_y - child["center_y"])))
    return list(dict.fromkeys(deltas))


def _repair_adjacent_true_shape_frames(
    nodes: list[dict],
    relations: list[dict],
    *,
    gap: float = 0.0,
    max_gap: float = 4.0,
    max_iterations: int = 5,
    damping: float = 0.8,
    max_overlap_ratio: float = 0.1,
) -> dict:
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    children = _adjacent_repair_children(relations)
    shift_totals: dict[str, list[float]] = {}
    repair_count = 0
    rounds: list[dict] = []
    max_iterations = max(0, int(max_iterations))
    _legacy_damping = max(0.0, min(1.0, float(damping)))
    max_gap = max(0.0, float(max_gap))
    gap = max(0.0, float(gap))
    max_overlap_ratio = max(0.0, float(max_overlap_ratio))

    def shift_subtree(root_id: str, dx: float, dy: float) -> None:
        stack = [root_id]
        visited: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = node_by_id.get(node_id)
            if node is None:
                continue
            _shift_node_layout(node, dx, dy)
            total = shift_totals.setdefault(node_id, [0.0, 0.0])
            total[0] += float(dx)
            total[1] += float(dy)
            stack.extend(children.get(node_id, []))

    adjacent_pairs = _adjacent_pairs(relations)
    for iteration in range(max_iterations):
        bboxes = {node_id: bbox for node_id, node in node_by_id.items() if (bbox := _node_repair_bbox(node)) is not None}
        max_gap_before = 0.0
        target_pairs: list[tuple[float, str, str]] = []
        for anchor_id, child_id in adjacent_pairs:
            anchor_bbox = bboxes.get(anchor_id)
            child_bbox = bboxes.get(child_id)
            if anchor_bbox is None or child_bbox is None:
                continue
            gap_value = _bbox_gap(anchor_bbox, child_bbox)
            overlap = _bbox_overlap_ratio(anchor_bbox, child_bbox)
            max_gap_before = max(max_gap_before, float(gap_value))
            if gap_value > max_gap or overlap > max_overlap_ratio:
                target_pairs.append((max(gap_value - max_gap, 0.0) + max(overlap - max_overlap_ratio, 0.0) * 100.0, anchor_id, child_id))

        if not target_pairs:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "relation_repair_count": 0,
                    "applied_root_count": 0,
                    "max_gap_before": float(max_gap_before),
                }
            )
            break

        applied_root_count = 0
        relation_repair_count = 0
        for _severity, anchor_id, child_id in sorted(target_pairs, reverse=True):
            bboxes = {node_id: bbox for node_id, node in node_by_id.items() if (bbox := _node_repair_bbox(node)) is not None}
            anchor_node = node_by_id.get(anchor_id)
            child_node = node_by_id.get(child_id)
            if anchor_node is None or child_node is None:
                continue
            anchor_bbox = bboxes.get(anchor_id)
            child_bbox = bboxes.get(child_id)
            if anchor_bbox is None or child_bbox is None:
                continue
            current_pair_score = _adjacent_pair_score(anchor_bbox, child_bbox, max_gap=max_gap, max_overlap_ratio=max_overlap_ratio)
            if current_pair_score[0] == 0 and current_pair_score[1] == 0:
                continue
            anchor_coarse = anchor_node.get("coarse_bbox") if isinstance(anchor_node.get("coarse_bbox"), list) else anchor_bbox
            child_coarse = child_node.get("coarse_bbox") if isinstance(child_node.get("coarse_bbox"), list) else child_bbox
            subtree_ids = _subtree_node_ids(child_id, children)
            base_score = _affected_constraint_score(
                bboxes,
                nodes,
                adjacent_pairs,
                subtree_ids,
                max_gap=max_gap,
                max_overlap_ratio=max_overlap_ratio,
            )
            best_score = base_score
            best_delta: tuple[float, float] | None = None
            for dx, dy in _adjacent_candidate_deltas(
                child_bbox,
                anchor_bbox,
                child_coarse=[float(value) for value in child_coarse],
                anchor_coarse=[float(value) for value in anchor_coarse],
                gap=gap,
            ):
                if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
                    continue
                candidate_bboxes = _shifted_bbox_map(bboxes, subtree_ids, dx, dy)
                candidate_score = _affected_constraint_score(
                    candidate_bboxes,
                    nodes,
                    adjacent_pairs,
                    subtree_ids,
                    max_gap=max_gap,
                    max_overlap_ratio=max_overlap_ratio,
                )
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_delta = (float(dx), float(dy))
            if best_delta is None:
                continue
            dx, dy = best_delta
            shift_subtree(child_id, dx, dy)
            applied_root_count += 1
            relation_repair_count += 1
        repair_count += int(relation_repair_count)
        rounds.append(
            {
                "iteration": int(iteration),
                "relation_repair_count": int(relation_repair_count),
                "applied_root_count": int(applied_root_count),
                "max_gap_before": float(max_gap_before),
            }
        )
    return {
        "enabled": True,
        "repair_count": int(repair_count),
        "shifted_node_count": int(len(shift_totals)),
        "iteration_count": int(len(rounds)),
        "max_iterations": int(max_iterations),
        "damping": float(_legacy_damping),
        "max_gap": float(max_gap),
        "max_overlap_ratio": float(max_overlap_ratio),
        "target_gap": float(gap),
        "solver": "candidate_projection",
        "rounds": rounds,
        "shift_by_node": {key: [float(values[0]), float(values[1])] for key, values in shift_totals.items()},
    }


def _fit_frame_to_shape_bbox(frame: dict, coarse_bbox: list[float] | None, local_bbox: dict, *, mode: str) -> dict:
    if coarse_bbox is None or str(mode) == "frame":
        return copy.deepcopy(frame)
    metrics = _bbox_metrics(coarse_bbox)
    local_width = abs(float(local_bbox.get("width", 1.0)))
    local_height = abs(float(local_bbox.get("height", 1.0)))
    if local_width <= 1e-6 or local_height <= 1e-6:
        return copy.deepcopy(frame)
    scale_x = metrics["width"] / local_width
    scale_y = metrics["height"] / local_height
    scale = max(scale_x, scale_y) if str(mode) == "cover" else min(scale_x, scale_y)
    local_center_x = float(local_bbox.get("min_x", -local_width / 2.0)) + local_width / 2.0
    local_center_y = float(local_bbox.get("min_y", -local_height / 2.0)) + local_height / 2.0
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    offset_x = (local_center_x * cos_theta - local_center_y * sin_theta) * scale
    offset_y = (local_center_x * sin_theta + local_center_y * cos_theta) * scale
    output = copy.deepcopy(frame)
    output["origin"] = [float(metrics["center_x"] - offset_x), float(metrics["center_y"] - offset_y)]
    output["scale"] = float(max(1.0, scale))
    output["orientation"] = float(theta)
    return output


def main() -> None:
    args = parse_args()
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]
    shape_library, shape_summary = build_geometry_shape_fallback_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
        min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
    )
    manifest_rows: list[dict] = []
    shape_modes: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    attached_total = 0
    missing_total = 0
    quality_failure_total = 0
    scale_values: list[float] = []
    tokenizer_config = ParseGraphTokenizerConfig()

    for fallback_index, row in enumerate(sample_rows):
        sample_index = int(row.get("sample_index", fallback_index))
        try:
            target = decode_coarse_scene_tokens_to_target([str(token) for token in row.get("tokens", []) or []], config=tokenizer_config)
        except Exception as exc:
            error_histogram[type(exc).__name__] += 1
            continue
        layout_constraint_summary = copy.deepcopy(
            ((target.get("metadata", {}) or {}).get("coarse_scene_layout_constraints", {}) or {})
        )
        graph = target.get("parse_graph", {}) or {}
        output_nodes: list[dict] = []
        geometry_rows: list[dict] = []
        sample_attached = 0
        sample_missing = 0
        sample_quality_failure = 0
        sample_shape_modes: Counter[str] = Counter()
        sample_quality_reasons: Counter[str] = Counter()

        for node in graph.get("nodes", []) or []:
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref:
                geometry_row = {
                    "node_id": str(output_node.get("id", "")),
                    "role": str(output_node.get("role", "")),
                    "label": int(output_node.get("label", 0)),
                    "geometry_model": str(output_node.get("geometry_model", "none")),
                }
                try:
                    shape, shape_mode = select_fallback_geometry_shape(output_node, shape_library)
                    if shape is None:
                        raise RuntimeError("true shape library produced no candidate")
                    local_bbox = copy.deepcopy(shape.get("local_bbox") or {})
                    if not local_bbox:
                        probe = geometry_target_from_fallback_shape(
                            shape,
                            source_node_id=str(geometry_ref),
                            frame=output_node.get("frame", {}),
                        )
                        local_bbox = geometry_local_bbox(probe)
                    final_frame = _fit_frame_to_shape_bbox(
                        output_node.get("frame", {}),
                        output_node.get("coarse_bbox"),
                        local_bbox,
                        mode=str(args.scale_fit_mode),
                    )
                    generated = geometry_target_from_fallback_shape(shape, source_node_id=str(geometry_ref), frame=final_frame)
                    quality = geometry_target_quality(
                        generated,
                        final_frame,
                        canvas_size=target.get("size", [256, 256]),
                        min_world_bbox_area=float(args.min_true_shape_world_bbox_area),
                        min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
                    )
                    if not bool(quality.get("usable", False)):
                        quality_failure_total += 1
                        sample_quality_failure += 1
                        quality_reasons.update(quality.get("reasons", []) or [])
                        sample_quality_reasons.update(quality.get("reasons", []) or [])
                    shape_modes[shape_mode] += 1
                    sample_shape_modes[shape_mode] += 1
                    scale_values.append(float(final_frame.get("scale", 1.0)))
                    output_node["geometry_model"] = copy.deepcopy(generated.get("geometry_model", output_node.get("geometry_model")))
                    output_node["frame"] = copy.deepcopy(final_frame)
                    output_node["true_shape_local_bbox"] = local_bbox
                    output_node["local_bbox_quality"] = quality
                    output_node["geometry_fallback_mode"] = shape_mode
                    if "geometry" in generated:
                        output_node["geometry"] = copy.deepcopy(generated["geometry"])
                    if "atoms" in generated:
                        output_node["atoms"] = copy.deepcopy(generated["atoms"])
                    output_node["layout_frame_source"] = "coarse_scene"
                    output_node["layout_shape_attach_mode"] = "true_shape_fallback"
                    geometry_row.update(
                        {
                            "valid": True,
                            "final_frame": copy.deepcopy(final_frame),
                            "true_shape_local_bbox": local_bbox,
                            "local_bbox_quality": quality,
                            "shape_mode": shape_mode,
                        }
                    )
                    sample_attached += 1
                    attached_total += 1
                except Exception as exc:
                    sample_missing += 1
                    missing_total += 1
                    error_histogram[type(exc).__name__] += 1
                    geometry_row.update({"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]})
                    output_node["coarse_scene_true_shape_error"] = f"{type(exc).__name__}: {exc}"
                geometry_rows.append(geometry_row)
            output_nodes.append(output_node)

        relations = copy.deepcopy(list(graph.get("relations", []) or []))
        run_legacy_adjacent_repair = bool(args.enable_legacy_adjacent_frame_repair) and not bool(args.disable_adjacent_frame_repair)
        if not run_legacy_adjacent_repair:
            adjacent_frame_repair = {
                "enabled": False,
                "repair_count": 0,
                "shifted_node_count": 0,
                "iteration_count": 0,
                "max_iterations": int(args.adjacent_repair_iterations),
                "damping": float(args.adjacent_repair_damping),
                "max_gap": float(args.adjacent_repair_max_gap),
                "max_overlap_ratio": float(args.adjacent_repair_max_overlap_ratio),
                "target_gap": 0.0,
                "solver": "disabled",
                "reason": "coarse_layout_constraint_solver",
                "rounds": [],
                "shift_by_node": {},
            }
        else:
            adjacent_frame_repair = _repair_adjacent_true_shape_frames(
                output_nodes,
                relations,
                max_gap=float(args.adjacent_repair_max_gap),
                max_iterations=int(args.adjacent_repair_iterations),
                damping=float(args.adjacent_repair_damping),
                max_overlap_ratio=float(args.adjacent_repair_max_overlap_ratio),
            )
            shifts = adjacent_frame_repair.get("shift_by_node", {}) or {}
            if shifts:
                frame_by_id = {str(node.get("id")): copy.deepcopy(node.get("frame", {}) or {}) for node in output_nodes}
                for geometry_row in geometry_rows:
                    node_id = str(geometry_row.get("node_id", ""))
                    if node_id in shifts:
                        geometry_row["final_frame"] = copy.deepcopy(frame_by_id.get(node_id, geometry_row.get("final_frame", {})))
                        geometry_row["adjacent_frame_repair_shift"] = copy.deepcopy(shifts[node_id])

        attached_target = {
            "format": "maskgen_generator_target_v1",
            "target_type": "parse_graph",
            "size": copy.deepcopy(target.get("size", [256, 256])),
            "parse_graph": {
                "nodes": output_nodes,
                "relations": relations,
                "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
            },
            "metadata": {
                "coarse_scene_true_shape": True,
                "sample_index": int(sample_index),
                "checkpoint": row.get("checkpoint"),
                "geometry_valid_count": int(sample_attached),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "true_shape_modes": dict(sample_shape_modes),
                "true_shape_quality_failure_count": int(sample_quality_failure),
                "true_shape_quality_reasons": dict(sample_quality_reasons),
                "geometry_rows": geometry_rows,
                "scale_fit_mode": str(args.scale_fit_mode),
                "coarse_scene_layout_constraints": layout_constraint_summary,
                "adjacent_frame_repair": adjacent_frame_repair,
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, attached_target)
        manifest_rows.append(
            {
                "sample_index": int(sample_index),
                "output_path": str(output_path.as_posix()),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "true_shape_quality_failure_count": int(sample_quality_failure),
            }
        )
        if int(args.progress_every) > 0 and len(manifest_rows) % int(args.progress_every) == 0:
            print(f"coarse_scene_true_shape_attach {len(manifest_rows)}/{len(sample_rows)}", flush=True)

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_coarse_scene_true_shape_attach_summary_v1",
        "samples": str(args.samples.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(sample_rows)),
        "output_count": int(len(manifest_rows)),
        "shape_fallback_summary": shape_summary,
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "true_shape_modes": dict(shape_modes),
        "true_shape_quality_failure_count": int(quality_failure_total),
        "true_shape_quality_reasons": dict(quality_reasons),
        "final_scale_stats": _numeric_stats(scale_values),
        "error_histogram": dict(error_histogram),
        "scale_fit_mode": str(args.scale_fit_mode),
        "adjacent_frame_repair": {
            "enabled": bool(args.enable_legacy_adjacent_frame_repair) and not bool(args.disable_adjacent_frame_repair),
            "max_iterations": int(args.adjacent_repair_iterations),
            "damping": float(args.adjacent_repair_damping),
            "max_gap": float(args.adjacent_repair_max_gap),
            "max_overlap_ratio": float(args.adjacent_repair_max_overlap_ratio),
            "solver": "candidate_projection",
        },
        "coarse_scene_layout_constraints": {
            "enabled": True,
            "solver": "coarse_layout_constraint_v1",
        },
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached coarse-scene true-shape samples={summary['output_count']} "
        f"attached={attached_total} missing={missing_total} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
