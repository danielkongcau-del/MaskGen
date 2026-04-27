from __future__ import annotations

import copy
from typing import Dict, List


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
