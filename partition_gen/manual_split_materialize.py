from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
from typing import Iterable, Sequence


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_split_path(value: object, *, split_root: Path, manifest_parent: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    for base in (manifest_parent, split_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def attach_true_geometry_to_topology(topology_target: dict, geometry_targets: Sequence[dict]) -> tuple[dict, dict]:
    geometry_by_id = geometry_targets_by_source_node_id(geometry_targets)
    graph = topology_target.get("parse_graph", {}) or {}
    output_nodes: list[dict] = []
    attached = 0
    missing = 0
    attach_modes: Counter[str] = Counter()

    for node in graph.get("nodes", []) or []:
        output_node = copy.deepcopy(node)
        geometry_ref = output_node.pop("geometry_ref", None)
        if geometry_ref:
            geometry_target = geometry_by_id.get(str(geometry_ref))
            if geometry_target is None:
                missing += 1
                attach_modes["missing"] += 1
            else:
                attached += 1
                attach_modes["true_split_geometry"] += 1
                output_node["geometry_model"] = copy.deepcopy(
                    geometry_target.get("geometry_model", output_node.get("geometry_model"))
                )
                if "frame" in geometry_target:
                    output_node["frame"] = copy.deepcopy(geometry_target["frame"])
                if "geometry" in geometry_target:
                    output_node["geometry"] = copy.deepcopy(geometry_target["geometry"])
                if "atoms" in geometry_target:
                    output_node["atoms"] = copy.deepcopy(geometry_target["atoms"])
                output_node["split_geometry_source_node_id"] = str(geometry_ref)
        output_nodes.append(output_node)

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
            "materialized_from_manual_split": True,
            "attached_geometry_count": int(attached),
            "missing_geometry_count": int(missing),
            "attach_modes": dict(attach_modes),
        },
    }
    return target, copy.deepcopy(target["metadata"])


def materialize_manual_split_targets(split_root: Path, *, max_samples: int | None = None) -> list[dict]:
    split_root = Path(split_root)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]

    targets: list[dict] = []
    for row_index, row in enumerate(rows):
        topology_path = resolve_split_path(
            row["topology_path"],
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(
                resolve_split_path(
                    value,
                    split_root=split_root,
                    manifest_parent=manifest_path.parent,
                )
            )
            for value in row.get("geometry_paths", []) or []
        ]
        target, diagnostics = attach_true_geometry_to_topology(topology_target, geometry_targets)
        target["metadata"].update(
            {
                "sample_index": int(row_index),
                "source_topology": str(topology_path.as_posix()),
                "source_geometry_count": int(len(geometry_targets)),
                "materialize_diagnostics": diagnostics,
            }
        )
        targets.append(target)
    return targets
