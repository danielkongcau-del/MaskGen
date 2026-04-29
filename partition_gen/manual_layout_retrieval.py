from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from partition_gen.manual_geometry_conditioning import (
    _resolve_path,
    iter_jsonl,
    load_json,
    renderable_geometry_node_indices,
)


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _node_key(node: dict, *, level: str) -> tuple:
    role = str(node.get("role", ""))
    label = int(node.get("label", 0))
    geometry_model = str(node.get("geometry_model", "polygon_code"))
    if level == "exact":
        return role, label, geometry_model
    if level == "role_label":
        return role, label
    if level == "role":
        return (role,)
    return ()


def _histogram_l1(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(int(left.get(key, 0)) - int(right.get(key, 0))) for key in keys))


def _sequence_distance(left: Sequence[str], right: Sequence[str]) -> float:
    shared = min(len(left), len(right))
    mismatches = sum(1 for index in range(shared) if str(left[index]) != str(right[index]))
    return float(mismatches + abs(len(left) - len(right)))


def _frame_origin(frame: dict) -> tuple[float, float]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return float(origin[0]), float(origin[1])


def _median_frame(frames: Sequence[dict]) -> dict | None:
    if not frames:
        return None
    origins = [_frame_origin(frame) for frame in frames]
    return {
        "origin": [
            float(median([origin[0] for origin in origins])),
            float(median([origin[1] for origin in origins])),
        ],
        "scale": float(median([float(frame.get("scale", 1.0)) for frame in frames])),
        "orientation": float(median([float(frame.get("orientation", 0.0)) for frame in frames])),
    }


def _numeric_stats(values: Sequence[float]) -> dict:
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


def topology_signature(topology_target: dict) -> dict:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    renderable_indices = renderable_geometry_node_indices(topology_target)
    renderable_nodes = [nodes[index] for index in renderable_indices]
    relations = list((topology_target.get("parse_graph", {}) or {}).get("relations", []) or [])

    role_histogram = Counter(str(node.get("role", "")) for node in renderable_nodes)
    label_histogram = Counter(str(node.get("label", 0)) for node in renderable_nodes)
    role_label_histogram = Counter(f"{node.get('role', '')}:{node.get('label', 0)}" for node in renderable_nodes)
    geometry_model_histogram = Counter(str(node.get("geometry_model", "polygon_code")) for node in renderable_nodes)
    relation_type_histogram = Counter(str(relation.get("type", "")) for relation in relations)

    return {
        "renderable_count": int(len(renderable_nodes)),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "role_histogram": dict(role_histogram),
        "label_histogram": dict(label_histogram),
        "role_label_histogram": dict(role_label_histogram),
        "geometry_model_histogram": dict(geometry_model_histogram),
        "relation_type_histogram": dict(relation_type_histogram),
        "role_sequence": [str(node.get("role", "")) for node in renderable_nodes],
        "role_label_sequence": [f"{node.get('role', '')}:{node.get('label', 0)}" for node in renderable_nodes],
    }


def topology_signature_distance(query: dict, candidate: dict) -> float:
    return float(
        2.0 * abs(int(query["renderable_count"]) - int(candidate["renderable_count"]))
        + 0.5 * abs(int(query["node_count"]) - int(candidate["node_count"]))
        + 0.5 * abs(int(query["relation_count"]) - int(candidate["relation_count"]))
        + 3.0 * _histogram_l1(query["role_histogram"], candidate["role_histogram"])
        + 2.0 * _histogram_l1(query["role_label_histogram"], candidate["role_label_histogram"])
        + 1.0 * _histogram_l1(query["label_histogram"], candidate["label_histogram"])
        + 1.0 * _histogram_l1(query["geometry_model_histogram"], candidate["geometry_model_histogram"])
        + 1.5 * _histogram_l1(query["relation_type_histogram"], candidate["relation_type_histogram"])
        + 0.5 * _sequence_distance(query["role_sequence"], candidate["role_sequence"])
        + 0.5 * _sequence_distance(query["role_label_sequence"], candidate["role_label_sequence"])
    )


def layout_rows_from_split_targets(topology_target: dict, geometry_targets: Sequence[dict]) -> list[dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    rows: list[dict] = []
    for node_index in renderable_geometry_node_indices(topology_target):
        node = nodes[int(node_index)]
        geometry_ref = node.get("geometry_ref")
        geometry_target = geometry_by_id.get(str(geometry_ref))
        if geometry_target is None or "frame" not in geometry_target:
            continue
        rows.append(
            {
                "node_index": int(node_index),
                "node_id": str(node.get("id", "")),
                "geometry_ref": str(geometry_ref),
                "role": str(node.get("role", "")),
                "label": int(node.get("label", 0)),
                "geometry_model": str(node.get("geometry_model", "polygon_code")),
                "frame": copy.deepcopy(geometry_target["frame"]),
            }
        )
    return rows


def geometry_condition_target_from_topology_node(node: dict, *, frame: dict, source_node_id: str | None = None) -> dict:
    node_id = str(source_node_id if source_node_id is not None else node.get("geometry_ref", node.get("id", "")))
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": node_id,
        "role": str(node.get("role", "")),
        "label": int(node.get("label", 0)),
        "geometry_model": str(node.get("geometry_model", "polygon_code")),
        "frame": copy.deepcopy(frame),
    }


def load_split_row(row: dict, *, split_root: Path, manifest_parent: Path) -> tuple[Path, dict, list[dict]]:
    topology_path = _resolve_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_parent)
    topology_target = load_json(topology_path)
    geometry_targets = [
        load_json(_resolve_path(value, split_root=split_root, manifest_parent=manifest_parent))
        for value in row.get("geometry_paths", []) or []
    ]
    return topology_path, topology_target, geometry_targets


def build_layout_retrieval_library(split_root: Path, *, max_samples: int | None = None) -> tuple[list[dict], dict]:
    split_root = Path(split_root)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]

    entries: list[dict] = []
    skipped_missing_layout = 0
    for index, row in enumerate(rows):
        topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        layout_rows = layout_rows_from_split_targets(topology_target, geometry_targets)
        if not layout_rows:
            skipped_missing_layout += 1
            continue
        entries.append(
            {
                "library_index": int(index),
                "stem": row.get("stem"),
                "topology_path": str(topology_path.as_posix()),
                "topology_target": topology_target,
                "signature": topology_signature(topology_target),
                "layout_rows": layout_rows,
            }
        )
    summary = {
        "format": "maskgen_layout_retrieval_library_summary_v1",
        "split_root": str(split_root.as_posix()),
        "input_count": int(len(rows)),
        "entry_count": int(len(entries)),
        "skipped_missing_layout": int(skipped_missing_layout),
    }
    return entries, summary


def build_layout_retrieval_fallbacks(library_entries: Sequence[dict]) -> dict:
    frames_by_exact = defaultdict(list)
    frames_by_role_label = defaultdict(list)
    frames_by_role = defaultdict(list)
    all_frames: list[dict] = []
    for entry in library_entries:
        for row in entry.get("layout_rows", []) or []:
            frame = row.get("frame")
            if frame is None:
                continue
            frames_by_exact[_node_key(row, level="exact")].append(frame)
            frames_by_role_label[_node_key(row, level="role_label")].append(frame)
            frames_by_role[_node_key(row, level="role")].append(frame)
            all_frames.append(frame)
    return {
        "exact": {key: _median_frame(frames) for key, frames in frames_by_exact.items()},
        "role_label": {key: _median_frame(frames) for key, frames in frames_by_role_label.items()},
        "role": {key: _median_frame(frames) for key, frames in frames_by_role.items()},
        "global": _median_frame(all_frames) or {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
    }


def retrieve_layout_entry(
    topology_target: dict,
    library_entries: Sequence[dict],
    *,
    exclude_stem: object | None = None,
) -> tuple[dict, float]:
    if not library_entries:
        raise ValueError("layout retrieval library is empty")
    query_signature = topology_signature(topology_target)
    best_entry = None
    best_score = math.inf
    for entry in library_entries:
        if exclude_stem is not None and entry.get("stem") == exclude_stem:
            continue
        score = topology_signature_distance(query_signature, entry["signature"])
        if score < best_score:
            best_score = score
            best_entry = entry
    if best_entry is None:
        best_entry = library_entries[0]
        best_score = topology_signature_distance(query_signature, best_entry["signature"])
    return best_entry, float(best_score)


def map_retrieved_layout_frames(
    topology_target: dict,
    retrieved_entry: dict,
    *,
    fallback_frames: dict | None = None,
) -> tuple[dict[int, dict], dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    retrieved_rows = list(retrieved_entry.get("layout_rows", []) or [])
    fallback_frames = fallback_frames or build_layout_retrieval_fallbacks([retrieved_entry])
    used: set[int] = set()
    mapping_modes: Counter[str] = Counter()
    frame_by_node_index: dict[int, dict] = {}
    node_mapping_modes: dict[int, str] = {}

    def choose_row(node: dict, *, level: str) -> tuple[int, dict] | None:
        key = _node_key(node, level=level)
        for row_index, row in enumerate(retrieved_rows):
            if row_index in used:
                continue
            if _node_key(row, level=level) == key:
                return row_index, row
        return None

    def choose_fallback(node: dict) -> tuple[str, dict]:
        for level in ("exact", "role_label", "role"):
            frame = fallback_frames.get(level, {}).get(_node_key(node, level=level))
            if frame is not None:
                return f"fallback_{level}_median", frame
        return "fallback_global_median", fallback_frames["global"]

    for node_index in renderable_geometry_node_indices(topology_target):
        node = nodes[int(node_index)]
        selected = None
        selected_mode = ""
        for level in ("exact", "role_label", "role"):
            selected = choose_row(node, level=level)
            if selected is not None:
                selected_mode = f"retrieved_{level}_order"
                break
        if selected is not None:
            row_index, row = selected
            used.add(row_index)
            frame = row["frame"]
        else:
            selected_mode, frame = choose_fallback(node)
        frame_by_node_index[int(node_index)] = copy.deepcopy(frame)
        mapping_modes[selected_mode] += 1
        node_mapping_modes[int(node_index)] = str(selected_mode)

    diagnostics = {
        "mapping_mode_histogram": dict(mapping_modes),
        "node_mapping_modes": {str(key): value for key, value in sorted(node_mapping_modes.items())},
        "mapped_frame_count": int(len(frame_by_node_index)),
        "retrieved_frame_count": int(len(retrieved_rows)),
        "unused_retrieved_frame_count": int(len(retrieved_rows) - len(used)),
    }
    return frame_by_node_index, diagnostics


def attach_retrieved_layout_to_topology(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    retrieved_entry: dict,
    fallback_frames: dict | None = None,
    retrieval_score: float | None = None,
) -> tuple[dict, dict]:
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    frame_by_index, mapping_diagnostics = map_retrieved_layout_frames(
        topology_target,
        retrieved_entry,
        fallback_frames=fallback_frames,
    )
    graph = topology_target.get("parse_graph", {}) or {}
    output_nodes: list[dict] = []
    attached = 0
    missing = 0
    attach_modes: Counter[str] = Counter()

    for index, node in enumerate(graph.get("nodes", []) or []):
        output = copy.deepcopy(node)
        geometry_ref = output.pop("geometry_ref", None)
        if geometry_ref:
            geometry_target = geometry_by_id.get(str(geometry_ref))
            frame = frame_by_index.get(int(index))
            if geometry_target is not None and frame is not None:
                attached += 1
                attach_modes["retrieved_frame_true_shape"] += 1
                output["geometry_model"] = copy.deepcopy(geometry_target.get("geometry_model", output.get("geometry_model")))
                output["frame"] = copy.deepcopy(frame)
                if "geometry" in geometry_target:
                    output["geometry"] = copy.deepcopy(geometry_target["geometry"])
                if "atoms" in geometry_target:
                    output["atoms"] = copy.deepcopy(geometry_target["atoms"])
                output["layout_frame_source"] = "retrieved_layout"
                output["layout_shape_attach_mode"] = "true_shape"
            else:
                missing += 1
                attach_modes["missing"] += 1
        output_nodes.append(output)

    metadata = {
        "layout_retrieval_attached": True,
        "attached_geometry_count": int(attached),
        "missing_geometry_count": int(missing),
        "attach_modes": dict(attach_modes),
        "retrieved_stem": retrieved_entry.get("stem"),
        "retrieved_library_index": int(retrieved_entry.get("library_index", -1)),
        "retrieved_topology_path": retrieved_entry.get("topology_path"),
        "retrieval_score": None if retrieval_score is None else float(retrieval_score),
        "mapping_diagnostics": mapping_diagnostics,
    }
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": copy.deepcopy(topology_target.get("size", [256, 256])),
        "parse_graph": {
            "nodes": output_nodes,
            "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
            "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
        },
        "metadata": metadata,
    }
    return target, copy.deepcopy(metadata)


def attach_retrieved_layout_to_split_targets(
    split_root: Path,
    *,
    library_entries: Sequence[dict],
    fallback_frames: dict,
    max_samples: int | None = None,
    exclude_same_stem: bool = False,
) -> list[dict]:
    split_root = Path(split_root)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    targets: list[dict] = []
    for index, row in enumerate(rows):
        topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        retrieved_entry, retrieval_score = retrieve_layout_entry(
            topology_target,
            library_entries,
            exclude_stem=row.get("stem") if bool(exclude_same_stem) else None,
        )
        target, diagnostics = attach_retrieved_layout_to_topology(
            topology_target,
            geometry_targets,
            retrieved_entry=retrieved_entry,
            fallback_frames=fallback_frames,
            retrieval_score=retrieval_score,
        )
        target["metadata"].update(
            {
                "sample_index": int(index),
                "source_topology": str(topology_path.as_posix()),
                "query_stem": row.get("stem"),
                "retrieval_diagnostics": diagnostics,
            }
        )
        targets.append(target)
    return targets


def summarize_retrieved_layout_targets(targets: Sequence[dict]) -> dict:
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    attached = 0
    missing = 0
    for target in targets:
        metadata = target.get("metadata", {}) or {}
        attached += int(metadata.get("attached_geometry_count", 0))
        missing += int(metadata.get("missing_geometry_count", 0))
        attach_modes.update(metadata.get("attach_modes", {}) or {})
        mapping_modes.update((metadata.get("mapping_diagnostics", {}) or {}).get("mapping_mode_histogram", {}) or {})
        score = metadata.get("retrieval_score")
        if score is not None:
            retrieval_scores.append(float(score))
    return {
        "attached_geometry_count": int(attached),
        "missing_geometry_count": int(missing),
        "attach_modes": dict(attach_modes),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
