from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from partition_gen.parse_graph_relations import relation_refs


def _node_ids(topology_target: dict) -> set[str]:
    return {str(node.get("id")) for node in topology_target.get("parse_graph", {}).get("nodes", []) if node.get("id") is not None}


def _geometry_targets_by_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _topology_geometry_refs(topology_target: dict) -> List[str]:
    refs: List[str] = []
    for node in topology_target.get("parse_graph", {}).get("nodes", []) or []:
        if node.get("geometry_ref"):
            refs.append(str(node["geometry_ref"]))
    return refs


def validate_topology_geometry_split(
    topology_target: dict,
    geometry_targets: Sequence[dict],
) -> Dict[str, object]:
    nodes = list(topology_target.get("parse_graph", {}).get("nodes", []) or [])
    relations = list(topology_target.get("parse_graph", {}).get("relations", []) or [])
    node_ids = _node_ids(topology_target)
    geometry_by_node = _geometry_targets_by_node_id(geometry_targets)
    geometry_refs = _topology_geometry_refs(topology_target)

    missing_geometry_ref_ids = sorted(ref for ref in geometry_refs if ref not in geometry_by_node)
    extra_geometry_target_ids = sorted(node_id for node_id in geometry_by_node if node_id not in geometry_refs)
    duplicate_geometry_target_ids = sorted(
        {
            str(target.get("source_node_id"))
            for target in geometry_targets
            if sum(1 for other in geometry_targets if other.get("source_node_id") == target.get("source_node_id")) > 1
        }
    )

    topology_nodes_with_geometry_payload = [
        str(node.get("id"))
        for node in nodes
        if "frame" in node or "geometry" in node or "atoms" in node
    ]

    invalid_relation_refs = []
    for relation in relations:
        for key, value in relation_refs(relation):
            if str(value) not in node_ids:
                invalid_relation_refs.append({"relation_type": relation.get("type"), "key": key, "value": str(value)})

    renderable_geometry_ref_count = sum(
        1
        for node in nodes
        if bool(node.get("renderable", True))
        and not bool(node.get("is_reference_only", False))
        and str(node.get("geometry_model", "none")) != "none"
        and node.get("geometry_ref")
    )

    is_valid = not (
        missing_geometry_ref_ids
        or extra_geometry_target_ids
        or duplicate_geometry_target_ids
        or topology_nodes_with_geometry_payload
        or invalid_relation_refs
    )
    return {
        "is_valid": bool(is_valid),
        "topology_node_count": int(len(nodes)),
        "topology_relation_count": int(len(relations)),
        "geometry_target_count": int(len(geometry_targets)),
        "geometry_ref_count": int(len(geometry_refs)),
        "renderable_geometry_ref_count": int(renderable_geometry_ref_count),
        "missing_geometry_ref_ids": missing_geometry_ref_ids,
        "extra_geometry_target_ids": extra_geometry_target_ids,
        "duplicate_geometry_target_ids": duplicate_geometry_target_ids,
        "topology_nodes_with_geometry_payload": topology_nodes_with_geometry_payload,
        "invalid_relation_refs": invalid_relation_refs,
    }


def summarize_split_validation(rows: Iterable[dict]) -> Dict[str, object]:
    rows = list(rows)
    return {
        "sample_count": int(len(rows)),
        "valid_count": int(sum(1 for row in rows if bool(row.get("split_valid")))),
        "invalid_count": int(sum(1 for row in rows if not bool(row.get("split_valid")))),
        "missing_geometry_ref_count": int(sum(int(row.get("missing_geometry_ref_count", 0)) for row in rows)),
        "invalid_relation_ref_count": int(sum(int(row.get("invalid_relation_ref_count", 0)) for row in rows)),
        "topology_nodes_with_geometry_payload_count": int(
            sum(int(row.get("topology_nodes_with_geometry_payload_count", 0)) for row in rows)
        ),
    }
