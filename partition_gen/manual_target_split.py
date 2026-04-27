from __future__ import annotations

import copy
from typing import Dict, List, Sequence


def _is_renderable_geometry_node(node: dict) -> bool:
    if not bool(node.get("renderable", True)):
        return False
    if bool(node.get("is_reference_only", False)):
        return False
    if str(node.get("role")) == "insert_object_group":
        return False
    if str(node.get("geometry_model", "none")) == "none":
        return False
    return True


def _strip_topology_node(node: dict, *, has_geometry_ref: bool) -> dict:
    output = copy.deepcopy(node)
    output.pop("frame", None)
    output.pop("geometry", None)
    output.pop("atoms", None)
    if has_geometry_ref:
        output["geometry_ref"] = str(node.get("id"))
    elif not bool(output.get("renderable", True)) or bool(output.get("is_reference_only", False)):
        output["geometry_model"] = "none"
    return output


def _build_geometry_target(node: dict, *, target: dict, source_target: str | None) -> dict | None:
    geometry_model = str(node.get("geometry_model", "none"))
    if geometry_model == "polygon_code":
        payload = {
            "format": "maskgen_generator_target_v1",
            "target_type": "manual_parse_graph_geometry_v1",
            "source_node_id": str(node.get("id")),
            "role": node.get("role"),
            "label": node.get("label"),
            "geometry_model": geometry_model,
            "frame": copy.deepcopy(node.get("frame", {})),
            "geometry": copy.deepcopy(node.get("geometry", {})),
            "metadata": {
                "source_target": source_target,
                "node_id": str(node.get("id")),
                "split_profile": "topology_geometry_v1",
            },
        }
    elif geometry_model == "convex_atoms":
        payload = {
            "format": "maskgen_generator_target_v1",
            "target_type": "manual_parse_graph_geometry_v1",
            "source_node_id": str(node.get("id")),
            "role": node.get("role"),
            "label": node.get("label"),
            "geometry_model": geometry_model,
            "frame": copy.deepcopy(node.get("frame", {})),
            "atoms": copy.deepcopy(node.get("atoms", [])),
            "metadata": {
                "source_target": source_target,
                "node_id": str(node.get("id")),
                "split_profile": "topology_geometry_v1",
            },
        }
    else:
        return None
    return payload


def build_topology_geometry_split_targets(
    target: dict,
    source_target: str | None = None,
) -> tuple[dict, List[dict], Dict[str, object]]:
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    residuals = list(graph.get("residuals", []) or [])

    topology_nodes: List[dict] = []
    geometry_targets: List[dict] = []
    missing_geometry_count = 0
    polygon_geometry_target_count = 0
    convex_atom_geometry_target_count = 0

    for node in nodes:
        has_geometry_target = _is_renderable_geometry_node(node)
        geometry_target = None
        if has_geometry_target:
            geometry_target = _build_geometry_target(node, target=target, source_target=source_target)
            if geometry_target is None:
                missing_geometry_count += 1
                has_geometry_target = False
            else:
                geometry_targets.append(geometry_target)
                if geometry_target["geometry_model"] == "polygon_code":
                    polygon_geometry_target_count += 1
                elif geometry_target["geometry_model"] == "convex_atoms":
                    convex_atom_geometry_target_count += 1
        topology_nodes.append(_strip_topology_node(node, has_geometry_ref=has_geometry_target))

    topology_target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": copy.deepcopy(target.get("size", [0, 0])),
        "parse_graph": {
            "nodes": topology_nodes,
            "relations": copy.deepcopy(relations),
            "residuals": copy.deepcopy(residuals),
        },
        "metadata": {
            "source_target": source_target,
            "split_profile": "topology_geometry_v1",
            "geometry_target_count": int(len(geometry_targets)),
        },
    }
    diagnostics = {
        "topology_node_count": int(len(topology_nodes)),
        "topology_relation_count": int(len(relations)),
        "renderable_node_count": int(sum(1 for node in nodes if bool(node.get("renderable", True)))),
        "non_renderable_node_count": int(sum(1 for node in nodes if not bool(node.get("renderable", True)))),
        "geometry_target_count": int(len(geometry_targets)),
        "polygon_geometry_target_count": int(polygon_geometry_target_count),
        "convex_atom_geometry_target_count": int(convex_atom_geometry_target_count),
        "missing_geometry_count": int(missing_geometry_count),
        "reference_only_count": int(sum(1 for node in nodes if bool(node.get("is_reference_only", False)))),
    }
    return topology_target, geometry_targets, diagnostics


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _relation_contains_pairs(relations: Sequence[dict]) -> set[tuple[str, str]]:
    return {
        (str(relation.get("parent")), str(relation.get("child")))
        for relation in relations
        if str(relation.get("type")) == "contains"
        and relation.get("parent") is not None
        and relation.get("child") is not None
    }


def merge_topology_geometry_targets(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    require_all_geometry: bool = True,
) -> dict:
    """Rebuild a full manual parse graph target from split topology and geometry targets."""

    graph = topology_target.get("parse_graph", {}) or {}
    geometry_by_ref = _geometry_targets_by_source_node_id(geometry_targets)
    nodes: List[dict] = []
    missing_geometry_refs: List[str] = []
    consumed_geometry_refs: set[str] = set()

    for node in graph.get("nodes", []) or []:
        merged_node = copy.deepcopy(node)
        geometry_ref = merged_node.pop("geometry_ref", None)
        if geometry_ref:
            ref = str(geometry_ref)
            geometry_target = geometry_by_ref.get(ref)
            if geometry_target is None:
                missing_geometry_refs.append(ref)
            else:
                consumed_geometry_refs.add(ref)
                merged_node["geometry_model"] = copy.deepcopy(geometry_target.get("geometry_model", merged_node.get("geometry_model")))
                if "frame" in geometry_target:
                    merged_node["frame"] = copy.deepcopy(geometry_target["frame"])
                if "geometry" in geometry_target:
                    merged_node["geometry"] = copy.deepcopy(geometry_target["geometry"])
                if "atoms" in geometry_target:
                    merged_node["atoms"] = copy.deepcopy(geometry_target["atoms"])
        nodes.append(merged_node)

    if require_all_geometry and missing_geometry_refs:
        raise ValueError(f"Missing geometry targets for refs: {', '.join(missing_geometry_refs)}")

    relations = copy.deepcopy(list(graph.get("relations", []) or []))
    contains_pairs = _relation_contains_pairs(relations)
    for node in nodes:
        if str(node.get("role")) != "insert_object_group":
            continue
        parent = str(node.get("id"))
        for child in node.get("children", []) or []:
            pair = (parent, str(child))
            if pair not in contains_pairs:
                relations.append({"type": "contains", "parent": parent, "child": str(child)})
                contains_pairs.add(pair)

    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": copy.deepcopy(topology_target.get("size", [0, 0])),
        "parse_graph": {
            "nodes": nodes,
            "relations": relations,
            "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
        },
        "metadata": {
            "merged_from_topology_geometry_split": True,
            "split_profile": "topology_geometry_v1",
            "topology_target_type": topology_target.get("target_type"),
            "geometry_target_count": int(len(geometry_targets)),
            "attached_geometry_count": int(len(consumed_geometry_refs)),
            "missing_geometry_ref_ids": missing_geometry_refs,
            "extra_geometry_target_ids": sorted(ref for ref in geometry_by_ref if ref not in consumed_geometry_refs),
        },
    }
