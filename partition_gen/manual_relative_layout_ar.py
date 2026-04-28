from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import torch

from partition_gen.manual_geometry_constrained_sampling import _sample_from_logits
from partition_gen.manual_geometry_conditioning import (
    TOPOLOGY_CONTEXT_TOKEN,
    iter_jsonl,
    load_json,
    renderable_geometry_node_indices,
    _resolve_path,
)
from partition_gen.manual_layout_ar import _frame_errors
from partition_gen.manual_topology_placeholder_geometry import (
    GeometryPlaceholderLibrary,
    decode_topology_tokens_to_target,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_relations import divides_target, inserted_in_container
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    TokenReader,
    dequantize,
    int_token,
    q_token,
    tokens_to_ids,
)


REL_LAYOUT_CONDITION_TOKEN = "MANUAL_REL_LAYOUT_CONDITION_V1"
REL_LAYOUT_TARGET_TOKEN = "REL_LAYOUT_TARGET"
REL_LAYOUT_START_TOKEN = "MANUAL_REL_LAYOUT_V1"
EPS = 1.0e-6


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {str(target.get("source_node_id")): target for target in geometry_targets if target.get("source_node_id") is not None}


def _node_index_by_id(nodes: Sequence[dict]) -> Dict[str, int]:
    return {str(node.get("id")): index for index, node in enumerate(nodes) if "id" in node}


def _wrap_angle(value: float) -> float:
    wrapped = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def _frame_origin(frame: dict) -> list[float]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return [float(origin[0]), float(origin[1])]


def _has_geometry_frame(node_id: str, geometry_by_id: Dict[str, dict]) -> bool:
    target = geometry_by_id.get(str(node_id))
    return bool(target is not None and "frame" in target)


def _relation_maps(topology_target: dict) -> dict:
    relations = list((topology_target.get("parse_graph", {}) or {}).get("relations", []) or [])
    contains_parent_by_child: dict[str, str] = {}
    inserted_container_by_object: dict[str, str] = {}
    divides_target_by_divider: dict[str, str] = {}
    adjacent_degree: Counter[str] = Counter()
    for relation in relations:
        relation_type = str(relation.get("type", ""))
        if relation_type == "contains":
            child = relation.get("child")
            parent = relation.get("parent")
            if child is not None and parent is not None:
                contains_parent_by_child[str(child)] = str(parent)
        elif relation_type == "inserted_in":
            obj = relation.get("object")
            container = inserted_in_container(relation)
            if obj is not None and container is not None:
                inserted_container_by_object[str(obj)] = str(container)
        elif relation_type == "divides":
            divider = relation.get("divider")
            target = divides_target(relation)
            if divider is not None and target is not None:
                divides_target_by_divider[str(divider)] = str(target)
        elif relation_type == "adjacent_to":
            for face in relation.get("faces", []) or []:
                adjacent_degree[str(face)] += 1
    return {
        "contains_parent_by_child": contains_parent_by_child,
        "inserted_container_by_object": inserted_container_by_object,
        "divides_target_by_divider": divides_target_by_divider,
        "adjacent_degree": adjacent_degree,
    }


def resolve_frame_anchor(
    node_id: str,
    *,
    topology_target: dict,
    geometry_by_id: Dict[str, dict],
    visited: set[str] | None = None,
) -> dict:
    """Resolve a relation target/container id to a renderable node that has frame geometry."""

    graph = topology_target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    node_by_id = {str(node.get("id")): node for node in nodes if "id" in node}
    node_index_by_id = _node_index_by_id(nodes)
    maps = _relation_maps(topology_target)
    visited = set() if visited is None else set(visited)
    current = str(node_id)
    if current in visited or len(visited) > len(nodes):
        return {"anchor_mode": "global", "anchor_node_index": None, "reason": "cycle_or_depth"}
    visited.add(current)

    if current in node_index_by_id and _has_geometry_frame(current, geometry_by_id):
        return {
            "anchor_mode": "node",
            "anchor_node_index": int(node_index_by_id[current]),
            "anchor_node_id": current,
            "reason": "frame_node",
        }

    container = maps["inserted_container_by_object"].get(current)
    if container is not None:
        resolved = resolve_frame_anchor(container, topology_target=topology_target, geometry_by_id=geometry_by_id, visited=visited)
        resolved["reason"] = f"recursive_inserted_in:{resolved.get('reason')}"
        return resolved

    parent = maps["contains_parent_by_child"].get(current)
    if parent is not None:
        group_container = maps["inserted_container_by_object"].get(parent)
        if group_container is not None:
            resolved = resolve_frame_anchor(
                group_container,
                topology_target=topology_target,
                geometry_by_id=geometry_by_id,
                visited=visited,
            )
            resolved["reason"] = f"recursive_parent_container:{resolved.get('reason')}"
            return resolved

    node = node_by_id.get(current, {})
    if str(node.get("role", "")) == "insert_object_group":
        return {"anchor_mode": "global", "anchor_node_index": None, "reason": "group_without_container"}
    return {"anchor_mode": "global", "anchor_node_index": None, "reason": "missing_frame_anchor"}


def relative_layout_anchor_for_node(
    topology_target: dict,
    node_index: int,
    geometry_by_id: Dict[str, dict],
) -> dict:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    if node_index < 0 or node_index >= len(nodes):
        return {"anchor_mode": "global", "anchor_node_index": None, "reason": "invalid_node_index"}
    node = nodes[int(node_index)]
    node_id = str(node.get("id", ""))
    role = str(node.get("role", ""))
    maps = _relation_maps(topology_target)

    if role == "insert_object":
        parent = maps["contains_parent_by_child"].get(node_id)
        container = maps["inserted_container_by_object"].get(parent) if parent is not None else None
        if container is None:
            container = maps["inserted_container_by_object"].get(node_id)
        if container is not None:
            resolved = resolve_frame_anchor(container, topology_target=topology_target, geometry_by_id=geometry_by_id, visited={node_id})
            if resolved.get("anchor_mode") == "node" and int(resolved["anchor_node_index"]) != int(node_index):
                resolved["anchor_relation"] = "insert_container"
                return resolved
        return {"anchor_mode": "global", "anchor_node_index": None, "reason": "insert_missing_container"}

    if role == "divider_region":
        target = maps["divides_target_by_divider"].get(node_id)
        if target is not None:
            resolved = resolve_frame_anchor(target, topology_target=topology_target, geometry_by_id=geometry_by_id, visited={node_id})
            if resolved.get("anchor_mode") == "node" and int(resolved["anchor_node_index"]) != int(node_index):
                resolved["anchor_relation"] = "divides_target"
                return resolved
        return {"anchor_mode": "global", "anchor_node_index": None, "reason": "divider_missing_target"}

    return {
        "anchor_mode": "global",
        "anchor_node_index": None,
        "reason": "root_or_fallback",
        "adjacent_degree": int(maps["adjacent_degree"].get(node_id, 0)),
    }


def _abs_frame_tokens(frame: dict, *, config: ParseGraphTokenizerConfig) -> List[str]:
    origin = _frame_origin(frame)
    return [
        "FRAME_ABS",
        q_token(origin[0], low=config.position_min, high=config.position_max, bins=config.position_bins),
        q_token(origin[1], low=config.position_min, high=config.position_max, bins=config.position_bins),
        q_token(float(frame.get("scale", 1.0)), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
        q_token(float(frame.get("orientation", 0.0)), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    ]


def relative_frame_from_absolute(child_frame: dict, anchor_frame: dict) -> dict:
    child_origin = _frame_origin(child_frame)
    anchor_origin = _frame_origin(anchor_frame)
    anchor_scale = max(float(anchor_frame.get("scale", 1.0)), EPS)
    child_scale = max(float(child_frame.get("scale", 1.0)), EPS)
    return {
        "dx": (child_origin[0] - anchor_origin[0]) / anchor_scale,
        "dy": (child_origin[1] - anchor_origin[1]) / anchor_scale,
        "log_scale_ratio": math.log(child_scale / anchor_scale),
        "dtheta": _wrap_angle(float(child_frame.get("orientation", 0.0)) - float(anchor_frame.get("orientation", 0.0))),
    }


def absolute_frame_from_relative(relative_frame: dict, anchor_frame: dict) -> dict:
    anchor_origin = _frame_origin(anchor_frame)
    anchor_scale = max(float(anchor_frame.get("scale", 1.0)), EPS)
    return {
        "origin": [
            anchor_origin[0] + float(relative_frame.get("dx", 0.0)) * anchor_scale,
            anchor_origin[1] + float(relative_frame.get("dy", 0.0)) * anchor_scale,
        ],
        "scale": anchor_scale * math.exp(float(relative_frame.get("log_scale_ratio", 0.0))),
        "orientation": _wrap_angle(float(anchor_frame.get("orientation", 0.0)) + float(relative_frame.get("dtheta", 0.0))),
    }


def _rel_frame_tokens(relative_frame: dict, *, config: ParseGraphTokenizerConfig) -> List[str]:
    return [
        "FRAME_REL",
        q_token(
            float(relative_frame.get("dx", 0.0)),
            low=config.relative_offset_min,
            high=config.relative_offset_max,
            bins=config.coord_bins,
        ),
        q_token(
            float(relative_frame.get("dy", 0.0)),
            low=config.relative_offset_min,
            high=config.relative_offset_max,
            bins=config.coord_bins,
        ),
        q_token(
            float(relative_frame.get("log_scale_ratio", 0.0)),
            low=config.relative_log_scale_min,
            high=config.relative_log_scale_max,
            bins=config.scale_bins,
        ),
        q_token(float(relative_frame.get("dtheta", 0.0)), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    ]


def _clip_counts(relative_frame: dict, *, config: ParseGraphTokenizerConfig) -> Counter[str]:
    counts: Counter[str] = Counter()
    for key in ("dx", "dy"):
        value = float(relative_frame.get(key, 0.0))
        if value < config.relative_offset_min or value > config.relative_offset_max:
            counts[f"{key}_clipped"] += 1
    value = float(relative_frame.get("log_scale_ratio", 0.0))
    if value < config.relative_log_scale_min or value > config.relative_log_scale_max:
        counts["log_scale_ratio_clipped"] += 1
    return counts


def relative_layout_condition_prefix_tokens(
    topology_target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    topology_tokens = encode_topology_target(topology_target, config=config)
    if not topology_tokens or topology_tokens[0] != "<BOS>" or topology_tokens[-1] != "<EOS>":
        raise ValueError("Expected encode_topology_target to return <BOS> ... <EOS>")
    return [
        "<BOS>",
        REL_LAYOUT_CONDITION_TOKEN,
        TOPOLOGY_CONTEXT_TOKEN,
        *topology_tokens[1:-1],
        REL_LAYOUT_TARGET_TOKEN,
    ]


def _relative_layout_rows(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig,
) -> tuple[list[dict], dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    rows: list[dict] = []
    diagnostics = {
        "anchor_mode_histogram": Counter(),
        "anchor_reason_histogram": Counter(),
        "fallback_anchor_count": 0,
        "recursive_anchor_count": 0,
        "anchor_forward_ref_count": 0,
        "relative_clipping_counts": Counter(),
    }
    for index, node in enumerate(nodes):
        geometry_ref = node.get("geometry_ref")
        if not geometry_ref or str(geometry_ref) not in geometry_by_id:
            continue
        geometry_target = geometry_by_id[str(geometry_ref)]
        if "frame" not in geometry_target:
            continue
        anchor = relative_layout_anchor_for_node(topology_target, int(index), geometry_by_id)
        anchor_mode = str(anchor.get("anchor_mode", "global"))
        row = {"node_index": int(index), "anchor_mode": anchor_mode, "frame": copy.deepcopy(geometry_target["frame"])}
        diagnostics["anchor_mode_histogram"][anchor_mode] += 1
        reason = str(anchor.get("reason", "unknown"))
        diagnostics["anchor_reason_histogram"][reason] += 1
        if anchor_mode == "node":
            anchor_index = int(anchor["anchor_node_index"])
            anchor_node = nodes[anchor_index]
            anchor_ref = str(anchor_node.get("geometry_ref", anchor_node.get("id", "")))
            anchor_target = geometry_by_id.get(anchor_ref)
            if anchor_target is None or "frame" not in anchor_target:
                row["anchor_mode"] = "global"
                diagnostics["fallback_anchor_count"] += 1
            else:
                relative_frame = relative_frame_from_absolute(geometry_target["frame"], anchor_target["frame"])
                row.update(
                    {
                        "anchor_node_index": int(anchor_index),
                        "anchor_node_id": str(anchor_node.get("id")),
                        "relative_frame": relative_frame,
                        "anchor_relation": anchor.get("anchor_relation"),
                    }
                )
                diagnostics["relative_clipping_counts"].update(_clip_counts(relative_frame, config=config))
                if anchor_index > index:
                    diagnostics["anchor_forward_ref_count"] += 1
                if "recursive" in reason:
                    diagnostics["recursive_anchor_count"] += 1
        else:
            diagnostics["fallback_anchor_count"] += int(reason not in {"root_or_fallback"})
        rows.append(row)
    return rows, diagnostics


def encode_relative_layout_target(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    size = topology_target.get("size", [0, 0])
    layout_rows, _diagnostics = _relative_layout_rows(topology_target, geometry_targets, config=config)
    tokens: List[str] = ["<BOS>", REL_LAYOUT_START_TOKEN, "SIZE"]
    tokens.extend(int_token(int(value), config=config) for value in size[:2])
    tokens.extend(["NODE_BLOCK", int_token(len(layout_rows), config=config)])
    for row in layout_rows:
        tokens.extend(["NODE", int_token(int(row["node_index"]), config=config)])
        if row.get("anchor_mode") == "node" and row.get("anchor_node_index") is not None:
            tokens.extend(["ANCHOR_NODE", int_token(int(row["anchor_node_index"]), config=config)])
            tokens.extend(_rel_frame_tokens(row["relative_frame"], config=config))
        else:
            tokens.append("ANCHOR_GLOBAL")
            tokens.extend(_abs_frame_tokens(row["frame"], config=config))
        tokens.append("END_NODE")
    tokens.append("<EOS>")
    return tokens


def encode_conditioned_relative_layout_target(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    layout_tokens = encode_relative_layout_target(topology_target, geometry_targets, config=config)
    return [*relative_layout_condition_prefix_tokens(topology_target, config=config), *layout_tokens[1:]]


def relative_layout_start_index(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens):
        if str(token) == REL_LAYOUT_START_TOKEN:
            return int(index)
    raise ValueError(f"{REL_LAYOUT_START_TOKEN} not found in relative layout tokens")


def relative_layout_condition_prefix_from_tokens(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    return tokens[: relative_layout_start_index(tokens)]


def extract_relative_layout_tokens_from_conditioned(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    start = relative_layout_start_index(tokens)
    return ["<BOS>", *tokens[start:]]


def topology_target_from_relative_layout_conditioned_tokens(tokens: Sequence[str]) -> dict:
    tokens = [str(token) for token in tokens]
    if TOPOLOGY_CONTEXT_TOKEN not in tokens or REL_LAYOUT_TARGET_TOKEN not in tokens:
        raise ValueError("Conditioned relative layout tokens must contain TOPOLOGY_CONTEXT and REL_LAYOUT_TARGET")
    start = tokens.index(TOPOLOGY_CONTEXT_TOKEN) + 1
    end = tokens.index(REL_LAYOUT_TARGET_TOKEN)
    topology_tokens = ["<BOS>", *tokens[start:end], "<EOS>"]
    return decode_topology_tokens_to_target(topology_tokens)


def _decode_abs_frame(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME_ABS")
    return {
        "origin": [
            dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins),
            dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins),
        ],
        "scale": dequantize(reader.next_q(), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
        "orientation": dequantize(reader.next_q(), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    }


def _decode_rel_frame(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME_REL")
    return {
        "dx": dequantize(reader.next_q(), low=config.relative_offset_min, high=config.relative_offset_max, bins=config.coord_bins),
        "dy": dequantize(reader.next_q(), low=config.relative_offset_min, high=config.relative_offset_max, bins=config.coord_bins),
        "log_scale_ratio": dequantize(
            reader.next_q(),
            low=config.relative_log_scale_min,
            high=config.relative_log_scale_max,
            bins=config.scale_bins,
        ),
        "dtheta": dequantize(reader.next_q(), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    }


def decode_relative_layout_tokens_to_target(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    reader = TokenReader([str(token) for token in tokens])
    reader.expect("<BOS>")
    reader.expect(REL_LAYOUT_START_TOKEN)
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]
    reader.expect("NODE_BLOCK")
    node_count = reader.next_int()
    nodes: List[dict] = []
    for _index in range(int(node_count)):
        reader.expect("NODE")
        node_index = reader.next_int()
        anchor_token = reader.next()
        if anchor_token == "ANCHOR_GLOBAL":
            nodes.append({"node_index": int(node_index), "anchor_mode": "global", "frame": _decode_abs_frame(reader, config=config)})
        elif anchor_token == "ANCHOR_NODE":
            anchor_index = reader.next_int()
            nodes.append(
                {
                    "node_index": int(node_index),
                    "anchor_mode": "node",
                    "anchor_node_index": int(anchor_index),
                    "relative_frame": _decode_rel_frame(reader, config=config),
                }
            )
        else:
            raise ValueError(f"Expected ANCHOR_GLOBAL or ANCHOR_NODE, got {anchor_token}")
        reader.expect("END_NODE")
    reader.expect("<EOS>")
    if reader.index != len(reader.tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(reader.tokens) - reader.index}")
    return {
        "format": "maskgen_manual_relative_layout_target_v1",
        "target_type": "manual_parse_graph_relative_layout_v1",
        "size": size,
        "nodes": nodes,
    }


def resolve_relative_layout_frames(relative_layout_target: dict) -> tuple[dict[int, dict], list[str]]:
    rows = {int(item["node_index"]): copy.deepcopy(item) for item in relative_layout_target.get("nodes", []) or []}
    frames: dict[int, dict] = {}
    errors: list[str] = []
    unresolved = set(rows.keys())
    for _ in range(max(1, len(rows) + 1)):
        progressed = False
        for node_index in list(unresolved):
            row = rows[node_index]
            if row.get("anchor_mode") == "global":
                frames[node_index] = copy.deepcopy(row["frame"])
                unresolved.remove(node_index)
                progressed = True
            else:
                anchor_index = int(row.get("anchor_node_index", -1))
                if anchor_index in frames:
                    frames[node_index] = absolute_frame_from_relative(row["relative_frame"], frames[anchor_index])
                    unresolved.remove(node_index)
                    progressed = True
        if not unresolved:
            break
        if not progressed:
            break
    if unresolved:
        errors.append(f"unresolved_relative_frames:{sorted(unresolved)}")
    return frames, errors


def relative_layout_to_absolute_layout_target(relative_layout_target: dict) -> dict:
    frames, errors = resolve_relative_layout_frames(relative_layout_target)
    nodes = [{"node_index": int(index), "frame": copy.deepcopy(frame)} for index, frame in sorted(frames.items())]
    return {
        "format": "maskgen_manual_layout_target_v1",
        "target_type": "manual_parse_graph_layout_v1",
        "size": copy.deepcopy(relative_layout_target.get("size", [256, 256])),
        "nodes": nodes,
        "errors": errors,
    }


def validate_relative_layout_tokens(
    tokens: Sequence[str],
    *,
    topology_target: dict | None = None,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    errors: List[str] = []
    layout_target = None
    try:
        layout_target = decode_relative_layout_tokens_to_target(tokens, config=config)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if layout_target is not None:
        node_indices = [int(row["node_index"]) for row in layout_target.get("nodes", []) or []]
        duplicates = sorted(index for index, count in Counter(node_indices).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate_node_indices:{duplicates}")
        node_index_set = set(node_indices)
        for row in layout_target.get("nodes", []) or []:
            if row.get("anchor_mode") == "node":
                anchor_index = int(row.get("anchor_node_index", -1))
                if anchor_index not in node_index_set:
                    errors.append(f"invalid_anchor_index:{anchor_index}")
                if anchor_index == int(row["node_index"]):
                    errors.append(f"self_anchor_index:{anchor_index}")
        if topology_target is not None:
            topology_nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
            max_index = len(topology_nodes) - 1
            invalid = [index for index in node_indices if index < 0 or index > max_index]
            if invalid:
                errors.append(f"invalid_node_indices:{invalid}")
            expected = renderable_geometry_node_indices(topology_target)
            missing = sorted(set(expected) - set(node_indices))
            extra = sorted(set(node_indices) - set(expected))
            if missing:
                errors.append(f"missing_node_indices:{missing}")
            if extra:
                errors.append(f"extra_node_indices:{extra}")
        _frames, frame_errors = resolve_relative_layout_frames(layout_target)
        errors.extend(frame_errors)
    return {
        "format": "maskgen_manual_relative_layout_validation_v1",
        "valid": not errors,
        "errors": errors,
        "length": int(len(tokens)),
        "hit_eos": bool(tokens and str(tokens[-1]) == "<EOS>"),
        "target": layout_target,
    }


def build_relative_layout_sequence_rows(
    split_root: Path,
    *,
    config: ParseGraphTokenizerConfig,
    vocab: Dict[str, int],
    max_tokens: int | None = None,
    include_token_ids: bool = True,
) -> tuple[List[dict], dict]:
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    sequence_rows: List[dict] = []
    skipped_too_long = 0
    lengths: List[int] = []
    topology_lengths: List[int] = []
    layout_lengths: List[int] = []
    layout_node_counts: List[int] = []
    anchor_mode_histogram: Counter[str] = Counter()
    anchor_reason_histogram: Counter[str] = Counter()
    relative_clipping_counts: Counter[str] = Counter()
    fallback_anchor_count = 0
    recursive_anchor_count = 0
    anchor_forward_ref_count = 0

    for row in rows:
        topology_path = _resolve_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        tokens = encode_conditioned_relative_layout_target(topology_target, geometry_targets, config=config)
        if max_tokens is not None and len(tokens) > int(max_tokens):
            skipped_too_long += 1
            continue
        layout_tokens = encode_relative_layout_target(topology_target, geometry_targets, config=config)
        layout_target = decode_relative_layout_tokens_to_target(layout_tokens, config=config)
        layout_rows, diagnostics = _relative_layout_rows(topology_target, geometry_targets, config=config)
        anchor_mode_histogram.update(diagnostics["anchor_mode_histogram"])
        anchor_reason_histogram.update(diagnostics["anchor_reason_histogram"])
        relative_clipping_counts.update(diagnostics["relative_clipping_counts"])
        fallback_anchor_count += int(diagnostics["fallback_anchor_count"])
        recursive_anchor_count += int(diagnostics["recursive_anchor_count"])
        anchor_forward_ref_count += int(diagnostics["anchor_forward_ref_count"])
        sequence_row = {
            "format": "maskgen_tokenized_parse_graph_v1",
            "tokenizer": "manual_relative_layout_conditioned_v1",
            "source_topology": str(topology_path.as_posix()),
            "source_target": str(row.get("source_target", topology_path.as_posix())),
            "stem": row.get("stem"),
            "length": int(len(tokens)),
            "topology_length": int(len(encode_topology_target(topology_target, config=config))),
            "layout_length": int(len(layout_tokens)),
            "layout_node_count": int(len(layout_target["nodes"])),
            "loss_start_index": int(relative_layout_start_index(tokens)),
            "anchor_mode_histogram": dict(diagnostics["anchor_mode_histogram"]),
            "tokens": tokens,
        }
        if include_token_ids:
            sequence_row["ids"] = tokens_to_ids(tokens, vocab)
        sequence_rows.append(sequence_row)
        lengths.append(len(tokens))
        topology_lengths.append(int(sequence_row["topology_length"]))
        layout_lengths.append(len(layout_tokens))
        layout_node_counts.append(int(sequence_row["layout_node_count"]))

    summary = {
        "format": "maskgen_manual_relative_layout_tokenized_summary_v1",
        "split_root": str(split_root.as_posix()),
        "sample_count": int(len(rows)),
        "written_layout": int(len(sequence_rows)),
        "skipped_too_long": int(skipped_too_long),
        "conditioned_length_mean": float(mean(lengths)) if lengths else None,
        "conditioned_length_max": int(max(lengths)) if lengths else None,
        "topology_length_mean": float(mean(topology_lengths)) if topology_lengths else None,
        "topology_length_max": int(max(topology_lengths)) if topology_lengths else None,
        "layout_length_mean": float(mean(layout_lengths)) if layout_lengths else None,
        "layout_length_max": int(max(layout_lengths)) if layout_lengths else None,
        "layout_node_count_mean": float(mean(layout_node_counts)) if layout_node_counts else None,
        "layout_node_count_max": int(max(layout_node_counts)) if layout_node_counts else None,
        "anchor_mode_histogram": dict(anchor_mode_histogram),
        "anchor_reason_histogram": dict(anchor_reason_histogram),
        "fallback_anchor_count": int(fallback_anchor_count),
        "recursive_anchor_count": int(recursive_anchor_count),
        "anchor_forward_ref_count": int(anchor_forward_ref_count),
        "relative_clipping_counts": dict(relative_clipping_counts),
    }
    return sequence_rows, summary


def evaluate_relative_layout_sample_rows(rows: Sequence[dict], *, top_k_invalid: int = 20) -> dict:
    sample_count = len(rows)
    valid_count = 0
    hit_eos_count = 0
    invalid_samples: List[dict] = []
    node_counts: List[int] = []
    origin_errors: List[float] = []
    scale_errors: List[float] = []
    orientation_errors: List[float] = []
    role_errors = defaultdict(lambda: defaultdict(list))
    failure_histogram: Counter[str] = Counter()
    anchor_mode_histogram: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        hit_eos_count += int(bool(tokens and tokens[-1] == "<EOS>"))
        topology_target = row.get("topology_target")
        validation = validate_relative_layout_tokens(tokens, topology_target=topology_target)
        if not bool(validation["valid"]):
            reason = str(validation["errors"][0]) if validation["errors"] else "unknown"
            failure_histogram[reason] += 1
            if len(invalid_samples) < int(top_k_invalid):
                invalid_samples.append({"sample_index": row.get("sample_index", row_index), "errors": validation["errors"]})
            continue
        valid_count += 1
        relative_layout = validation["target"] or {}
        node_counts.append(len(relative_layout.get("nodes", []) or []))
        anchor_mode_histogram.update(str(item.get("anchor_mode", "unknown")) for item in relative_layout.get("nodes", []) or [])
        absolute_layout = relative_layout_to_absolute_layout_target(relative_layout)
        target_layout = row.get("target_layout")
        if target_layout is None:
            continue
        target_abs = relative_layout_to_absolute_layout_target(target_layout)
        target_by_index = {int(item["node_index"]): item["frame"] for item in target_abs.get("nodes", []) or []}
        topology_nodes = list((topology_target or {}).get("parse_graph", {}).get("nodes", []) or [])
        for item in absolute_layout.get("nodes", []) or []:
            node_index = int(item["node_index"])
            if node_index not in target_by_index:
                continue
            errors = _frame_errors(item["frame"], target_by_index[node_index])
            origin_errors.append(errors["origin"])
            scale_errors.append(errors["scale"])
            orientation_errors.append(errors["orientation"])
            role = str(topology_nodes[node_index].get("role", "unknown")) if 0 <= node_index < len(topology_nodes) else "unknown"
            role_errors[role]["origin"].append(errors["origin"])
            role_errors[role]["scale"].append(errors["scale"])
            role_errors[role]["orientation"].append(errors["orientation"])

    def avg(values: Sequence[float]) -> float | None:
        return float(mean(values)) if values else None

    return {
        "format": "maskgen_manual_relative_layout_sample_eval_v1",
        "sample_count": int(sample_count),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "hit_eos_count": int(hit_eos_count),
        "layout_node_count_mean": float(mean(node_counts)) if node_counts else None,
        "origin_mae": avg(origin_errors),
        "scale_mae": avg(scale_errors),
        "orientation_mae": avg(orientation_errors),
        "anchor_mode_histogram": dict(anchor_mode_histogram),
        "role_metrics": {
            role: {
                "count": int(len(values["origin"])),
                "origin_mae": avg(values["origin"]),
                "scale_mae": avg(values["scale"]),
                "orientation_mae": avg(values["orientation"]),
            }
            for role, values in sorted(role_errors.items())
        },
        "failure_reason_histogram": dict(failure_histogram.most_common()),
        "invalid_samples": invalid_samples,
    }


class RelativeLayoutGrammarState:
    def __init__(self, topology_target: dict, *, config: ParseGraphTokenizerConfig | None = None) -> None:
        self.topology_target = topology_target
        self.config = config or ParseGraphTokenizerConfig()
        self.expected_node_indices = renderable_geometry_node_indices(topology_target)
        nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
        geometry_stub_by_id = {
            str(node.get("geometry_ref")): {"frame": {}}
            for node in nodes
            if node.get("geometry_ref")
        }
        self.anchor_by_node_index = {
            int(index): relative_layout_anchor_for_node(topology_target, int(index), geometry_stub_by_id)
            for index in self.expected_node_indices
        }
        self.phase = "manual"
        self.node_count = len(self.expected_node_indices)
        self.node_position = 0
        self.done = False
        self.errors: List[str] = []

    def _current_anchor(self) -> dict:
        if self.node_position >= len(self.expected_node_indices):
            return {"anchor_mode": "global"}
        return self.anchor_by_node_index.get(int(self.expected_node_indices[self.node_position]), {"anchor_mode": "global"})

    def allowed_token_strings(self) -> List[str]:
        if self.done:
            return []
        if self.phase == "manual":
            return [REL_LAYOUT_START_TOKEN]
        if self.phase == "size":
            return ["SIZE"]
        if self.phase == "size_w":
            return [int_token(int(self.config.position_max), config=self.config)]
        if self.phase == "size_h":
            return [int_token(int(self.config.position_max), config=self.config)]
        if self.phase == "node_block":
            return ["NODE_BLOCK"]
        if self.phase == "node_count":
            return [int_token(self.node_count, config=self.config)]
        if self.phase == "node_start":
            return ["NODE"]
        if self.phase == "node_index":
            return [int_token(self.expected_node_indices[self.node_position], config=self.config)]
        if self.phase == "anchor":
            return ["ANCHOR_NODE"] if self._current_anchor().get("anchor_mode") == "node" else ["ANCHOR_GLOBAL"]
        if self.phase == "anchor_index":
            return [int_token(int(self._current_anchor()["anchor_node_index"]), config=self.config)]
        if self.phase == "frame_kind":
            return ["FRAME_REL"] if self._current_anchor().get("anchor_mode") == "node" else ["FRAME_ABS"]
        if self.phase in {"origin_x", "origin_y"}:
            return [f"Q_{index}" for index in range(int(self.config.position_bins))]
        if self.phase == "abs_scale":
            return [f"Q_{index}" for index in range(int(self.config.scale_bins))]
        if self.phase == "orientation":
            return [f"Q_{index}" for index in range(int(self.config.angle_bins))]
        if self.phase in {"rel_dx", "rel_dy"}:
            return [f"Q_{index}" for index in range(int(self.config.coord_bins))]
        if self.phase == "rel_log_scale":
            return [f"Q_{index}" for index in range(int(self.config.scale_bins))]
        if self.phase == "end_node":
            return ["END_NODE"]
        if self.phase == "eos":
            return ["<EOS>"]
        return []

    def step(self, token: str) -> bool:
        token = str(token)
        if token not in set(self.allowed_token_strings()):
            self.errors.append(f"illegal_{token}_in_phase_{self.phase}")
            self.done = True
            return False
        if self.phase == "manual":
            self.phase = "size"
        elif self.phase == "size":
            self.phase = "size_w"
        elif self.phase == "size_w":
            self.phase = "size_h"
        elif self.phase == "size_h":
            self.phase = "node_block"
        elif self.phase == "node_block":
            self.phase = "node_count"
        elif self.phase == "node_count":
            self.phase = "node_start" if self.node_count > 0 else "eos"
        elif self.phase == "node_start":
            self.phase = "node_index"
        elif self.phase == "node_index":
            self.phase = "anchor"
        elif self.phase == "anchor":
            self.phase = "anchor_index" if token == "ANCHOR_NODE" else "frame_kind"
        elif self.phase == "anchor_index":
            self.phase = "frame_kind"
        elif self.phase == "frame_kind":
            self.phase = "rel_dx" if token == "FRAME_REL" else "origin_x"
        elif self.phase == "origin_x":
            self.phase = "origin_y"
        elif self.phase == "origin_y":
            self.phase = "abs_scale"
        elif self.phase == "abs_scale":
            self.phase = "orientation"
        elif self.phase == "rel_dx":
            self.phase = "rel_dy"
        elif self.phase == "rel_dy":
            self.phase = "rel_log_scale"
        elif self.phase == "rel_log_scale":
            self.phase = "orientation"
        elif self.phase == "orientation":
            self.phase = "end_node"
        elif self.phase == "end_node":
            self.node_position += 1
            self.phase = "node_start" if self.node_position < self.node_count else "eos"
        elif self.phase == "eos":
            self.done = True
        return True

    def diagnostics(self) -> dict:
        return {
            "phase": self.phase,
            "done": bool(self.done),
            "errors": list(self.errors),
            "node_count": int(self.node_count),
            "node_position": int(self.node_position),
        }


@torch.no_grad()
def sample_relative_layout_constrained(
    model,
    vocab: Dict[str, int],
    *,
    topology_target: dict,
    prefix_tokens: Sequence[str] | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: int | None = None,
    config: ParseGraphTokenizerConfig | None = None,
    device: torch.device | str | None = None,
    use_cache: bool = True,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    device = torch.device(device) if device is not None else next(model.parameters()).device
    prefix = list(prefix_tokens or relative_layout_condition_prefix_tokens(topology_target, config=config))
    missing = [token for token in prefix if token not in vocab]
    if missing:
        raise ValueError(f"Relative layout prefix contains tokens not in vocab: {missing}")
    state = RelativeLayoutGrammarState(topology_target, config=config)
    ids = [int(vocab[token]) for token in prefix]
    tokens = [str(token) for token in prefix]
    stopped_reason = "max_new_tokens"
    block_size = int(getattr(model.config, "block_size", max_new_tokens + len(ids)))
    use_kv_cache = bool(
        use_cache and getattr(model, "supports_kv_cache", False) and int(max_new_tokens) + len(ids) <= int(block_size)
    )
    past_kv = None

    for _step in range(int(max_new_tokens)):
        allowed_tokens = state.allowed_token_strings()
        allowed_ids = [int(vocab[token]) for token in allowed_tokens if token in vocab]
        if not allowed_ids:
            state.errors.append(f"empty_allowed_set_phase_{state.phase}")
            stopped_reason = "empty_allowed_set"
            break
        if use_kv_cache:
            if past_kv is None:
                input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
                outputs = model(input_ids, use_cache=True)
            else:
                input_ids = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                outputs = model(input_ids, past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
        else:
            input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
            outputs = model(input_ids)
        logits = outputs["logits"][0, -1, :]
        next_id = _sample_from_logits(logits, allowed_ids=allowed_ids, temperature=temperature, top_k=top_k)
        next_token = inverse_vocab.get(int(next_id), "<UNK>")
        ids.append(int(next_id))
        tokens.append(next_token)
        state.step(next_token)
        if next_token == "<EOS>" or state.done:
            stopped_reason = "eos" if next_token == "<EOS>" else "done"
            break
    return {
        "ids": ids,
        "tokens": tokens,
        "layout_tokens": extract_relative_layout_tokens_from_conditioned(tokens),
        "length": int(len(ids)),
        "hit_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "stopped_reason": stopped_reason,
        "constraint_diagnostics": state.diagnostics(),
    }


def sample_model_conditioned_relative_layout_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    source_rows: Sequence[dict],
    progress_every: int = 0,
    progress_label: str = "relative_layout_sample",
) -> List[dict]:
    if not source_rows:
        raise ValueError("source_rows must contain relative layout token rows")
    rows: List[dict] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for sample_index in range(int(num_samples)):
            source_row = source_rows[sample_index % len(source_rows)]
            source_tokens = [str(token) for token in source_row.get("tokens", []) or []]
            topology_target = topology_target_from_relative_layout_conditioned_tokens(source_tokens)
            target_layout = decode_relative_layout_tokens_to_target(extract_relative_layout_tokens_from_conditioned(source_tokens))
            sample = sample_relative_layout_constrained(
                model,
                vocab,
                topology_target=topology_target,
                prefix_tokens=relative_layout_condition_prefix_from_tokens(source_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=top_k,
                device=device,
            )
            rows.append(
                {
                    "format": "maskgen_manual_relative_layout_sample_v1",
                    "sample_index": int(sample_index),
                    "sampling_mode": "relative_layout_constrained",
                    "prefix_tokens": relative_layout_condition_prefix_from_tokens(source_tokens),
                    "length": int(len(sample["layout_tokens"])),
                    "conditioned_length": int(len(sample["tokens"])),
                    "hit_eos": bool(sample["hit_eos"]),
                    "stopped_reason": sample.get("stopped_reason"),
                    "tokens": sample["layout_tokens"],
                    "conditioned_tokens": sample["tokens"],
                    "ids": sample["ids"],
                    "topology_target": topology_target,
                    "target_layout": target_layout,
                    "constraint_diagnostics": sample.get("constraint_diagnostics"),
                }
            )
            if int(progress_every) > 0 and (sample_index + 1) % int(progress_every) == 0:
                print(f"{progress_label} {sample_index + 1}/{num_samples}")
    finally:
        if was_training:
            model.train()
    return rows


def attach_relative_layout_frames_to_topology(
    topology_target: dict,
    relative_layout_target: dict,
    *,
    shape_library: GeometryPlaceholderLibrary | None = None,
    geometry_by_node_id: Dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    absolute_layout = relative_layout_to_absolute_layout_target(relative_layout_target)
    frame_by_index = {int(item["node_index"]): copy.deepcopy(item["frame"]) for item in absolute_layout.get("nodes", []) or []}
    graph = topology_target.get("parse_graph", {}) or {}
    output_nodes: List[dict] = []
    attached = 0
    missing = 0
    attach_modes: Counter[str] = Counter()
    for index, node in enumerate(graph.get("nodes", []) or []):
        output = copy.deepcopy(node)
        geometry_ref = output.pop("geometry_ref", None)
        if geometry_ref:
            frame = frame_by_index.get(int(index))
            shape_target = None
            mode = "missing"
            if geometry_by_node_id is not None:
                shape_target = geometry_by_node_id.get(str(geometry_ref))
                mode = "true_shape" if shape_target is not None else "missing"
            elif shape_library is not None:
                shape_target, mode = shape_library.choose(
                    role=str(output.get("role", "")),
                    label=int(output.get("label", 0)),
                    geometry_model=str(output.get("geometry_model", "polygon_code")),
                )
            if frame is not None and shape_target is not None:
                attached += 1
                attach_modes[mode] += 1
                output["geometry_model"] = copy.deepcopy(shape_target.get("geometry_model", output.get("geometry_model")))
                output["frame"] = copy.deepcopy(frame)
                if "geometry" in shape_target:
                    output["geometry"] = copy.deepcopy(shape_target["geometry"])
                if "atoms" in shape_target:
                    output["atoms"] = copy.deepcopy(shape_target["atoms"])
                output["layout_frame_source"] = "relative_layout_ar"
                output["layout_shape_attach_mode"] = mode
            else:
                missing += 1
                attach_modes["missing"] += 1
        output_nodes.append(output)
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": copy.deepcopy(topology_target.get("size", [256, 256])),
        "parse_graph": {
            "nodes": output_nodes,
            "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
            "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
        },
        "metadata": {
            "relative_layout_ar_attached": True,
            "attached_geometry_count": int(attached),
            "missing_geometry_count": int(missing),
            "attach_modes": dict(attach_modes),
            "relative_layout_errors": list(absolute_layout.get("errors", []) or []),
        },
    }
    return target, copy.deepcopy(target["metadata"])


def attach_relative_layout_to_topology_sample_rows(
    rows: Sequence[dict],
    *,
    model,
    vocab: Dict[str, int],
    tokenizer_config: ParseGraphTokenizerConfig,
    device,
    shape_library: GeometryPlaceholderLibrary,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_k: int | None = 50,
    include_invalid: bool = False,
) -> list[dict]:
    targets: list[dict] = []
    for fallback_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not include_invalid:
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample = sample_relative_layout_constrained(
            model,
            vocab,
            topology_target=topology_target,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=top_k,
            config=tokenizer_config,
            device=device,
        )
        try:
            layout_target = decode_relative_layout_tokens_to_target(sample["layout_tokens"], config=tokenizer_config)
            layout_valid = True
        except Exception:
            layout_target = {"nodes": []}
            layout_valid = False
        target, _diagnostics = attach_relative_layout_frames_to_topology(
            topology_target,
            layout_target,
            shape_library=shape_library,
        )
        target["metadata"].update(
            {
                "sample_index": int(row.get("sample_index", fallback_index)),
                "semantic_valid": bool(validation["semantic_valid"]),
                "relative_layout_valid": bool(layout_valid),
                "relative_layout_hit_eos": bool(sample.get("hit_eos", False)),
            }
        )
        targets.append(target)
    return targets


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
