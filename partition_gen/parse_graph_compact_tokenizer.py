from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from partition_gen.parse_graph_relations import divides_target, inserted_in_container, relation_refs
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    _encode_frame,
    _encode_manual_convex_atoms,
    _encode_manual_polygon_geometry,
    _manual_geometry_token,
    _manual_relation_token,
    _manual_role_token,
    encode_generator_target,
    int_token,
)


def _node_index_by_id(nodes: Sequence[dict]) -> Dict[str, int]:
    return {str(node["id"]): index for index, node in enumerate(nodes) if "id" in node}


def _index(value: object, node_index_by_id: Dict[str, int]) -> int | None:
    if value is None:
        return None
    return node_index_by_id.get(str(value))


def _append_geometry(tokens: List[str], node: dict, *, config: ParseGraphTokenizerConfig) -> None:
    geometry_model = str(node.get("geometry_model", "none"))
    if geometry_model == "polygon_code":
        _encode_frame(tokens, node.get("frame", {}), config=config)
        _encode_manual_polygon_geometry(tokens, node.get("geometry", {}), config=config)
    elif geometry_model == "convex_atoms":
        _encode_frame(tokens, node.get("frame", {}), config=config)
        _encode_manual_convex_atoms(tokens, node.get("atoms", []), config=config)


def _append_compact_node(
    tokens: List[str],
    node: dict,
    *,
    node_index_by_id: Dict[str, int],
    config: ParseGraphTokenizerConfig,
    include_geometry: bool,
    topology_mode: bool = False,
) -> None:
    tokens.extend(
        [
            "NODE",
            _manual_role_token(str(node.get("role", ""))),
            int_token(int(node.get("label", 0)), config=config),
        ]
    )
    if topology_mode:
        tokens.append(int_token(1 if bool(node.get("renderable", True)) else 0, config=config))
    tokens.extend(
        [
            int_token(1 if bool(node.get("is_reference_only", False)) else 0, config=config),
            _manual_geometry_token(str(node.get("geometry_model", "none"))),
        ]
    )
    if topology_mode:
        tokens.append(int_token(1 if node.get("geometry_ref") else 0, config=config))
    if str(node.get("role")) == "insert_object_group":
        child_indices = [
            node_index_by_id[str(child_id)]
            for child_id in node.get("children", []) or []
            if str(child_id) in node_index_by_id
        ]
        tokens.extend(["CHILDREN", int_token(len(child_indices), config=config)])
        tokens.extend(int_token(index, config=config) for index in child_indices)
    if include_geometry:
        _append_geometry(tokens, node, config=config)
    tokens.append("END_NODE")


def _relation_pair_indices(relation: dict, node_index_by_id: Dict[str, int]) -> Tuple[int, int] | None:
    relation_type = str(relation.get("type"))
    if relation_type == "inserted_in":
        left = _index(relation.get("object"), node_index_by_id)
        right = _index(inserted_in_container(relation), node_index_by_id)
    elif relation_type == "divides":
        left = _index(relation.get("divider"), node_index_by_id)
        right = _index(divides_target(relation), node_index_by_id)
    elif relation_type == "adjacent_to":
        faces = list(relation.get("faces", []) or [])
        left = _index(faces[0], node_index_by_id) if len(faces) >= 1 else None
        right = _index(faces[1], node_index_by_id) if len(faces) >= 2 else None
    else:
        return None
    if left is None or right is None:
        return None
    return int(left), int(right)


def _other_relation_indices(relation: dict, node_index_by_id: Dict[str, int]) -> List[int]:
    indices: List[int] = []
    for _key, value in relation_refs(relation):
        index = _index(value, node_index_by_id)
        if index is not None:
            indices.append(int(index))
    return indices


def _relation_block_tokens_and_diagnostics(
    relations: Sequence[dict],
    *,
    node_index_by_id: Dict[str, int],
    config: ParseGraphTokenizerConfig,
) -> tuple[List[str], Dict[str, int]]:
    inserted_pairs: List[Tuple[int, int]] = []
    divide_pairs: List[Tuple[int, int]] = []
    adjacent_pairs: List[Tuple[int, int]] = []
    other_relations: List[tuple[str, List[int]]] = []
    contains_count = 0

    for relation in relations:
        relation_type = str(relation.get("type"))
        if relation_type == "contains":
            contains_count += 1
            continue
        pair = _relation_pair_indices(relation, node_index_by_id)
        if relation_type == "inserted_in" and pair is not None:
            inserted_pairs.append(pair)
        elif relation_type == "divides" and pair is not None:
            divide_pairs.append(pair)
        elif relation_type == "adjacent_to" and pair is not None:
            adjacent_pairs.append(pair)
        else:
            other_relations.append((relation_type, _other_relation_indices(relation, node_index_by_id)))

    tokens: List[str] = []
    for block_token, pairs in (
        ("REL_BLOCK_INSERTED_IN", inserted_pairs),
        ("REL_BLOCK_DIVIDES", divide_pairs),
        ("REL_BLOCK_ADJACENT_TO", adjacent_pairs),
    ):
        tokens.extend([block_token, int_token(len(pairs), config=config)])
        for left, right in pairs:
            tokens.extend([int_token(left, config=config), int_token(right, config=config)])
        tokens.append("END_BLOCK")

    tokens.extend(["REL_BLOCK_OTHER", int_token(len(other_relations), config=config)])
    for relation_type, indices in other_relations:
        tokens.extend([_manual_relation_token(relation_type), int_token(len(indices), config=config)])
        tokens.extend(int_token(index, config=config) for index in indices)
    tokens.append("END_BLOCK")

    return tokens, {
        "contains_relation_count": int(contains_count),
        "skipped_contains_relation_count": int(contains_count),
        "inserted_in_count": int(len(inserted_pairs)),
        "divides_count": int(len(divide_pairs)),
        "adjacent_to_count": int(len(adjacent_pairs)),
        "other_relation_count": int(len(other_relations)),
        "relation_token_count_compact": int(len(tokens)),
    }


def encode_generator_target_compact(
    target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    size = target.get("size", [0, 0])
    node_index_by_id = _node_index_by_id(nodes)

    tokens: List[str] = ["<BOS>", "MANUAL_PARSE_GRAPH_COMPACT_V1", "SIZE"]
    tokens.extend(int_token(int(value), config=config) for value in size[:2])
    tokens.extend(["NODE_BLOCK", int_token(len(nodes), config=config)])
    for node in nodes:
        _append_compact_node(
            tokens,
            node,
            node_index_by_id=node_index_by_id,
            config=config,
            include_geometry=True,
        )
    relation_tokens, _diagnostics = _relation_block_tokens_and_diagnostics(
        relations,
        node_index_by_id=node_index_by_id,
        config=config,
    )
    tokens.extend(relation_tokens)
    tokens.extend(["RESIDUALS", int_token(len(graph.get("residuals", [])), config=config), "<EOS>"])
    return tokens


def compact_tokenizer_diagnostics(
    target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> Dict[str, object]:
    config = config or ParseGraphTokenizerConfig()
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    node_index_by_id = _node_index_by_id(nodes)
    compact_tokens = encode_generator_target_compact(target, config=config)
    old_tokens = encode_generator_target(target, config=config)
    relation_tokens, relation_diag = _relation_block_tokens_and_diagnostics(
        relations,
        node_index_by_id=node_index_by_id,
        config=config,
    )
    return {
        "old_total_tokens": int(len(old_tokens)),
        "compact_total_tokens": int(len(compact_tokens)),
        "token_reduction": int(len(old_tokens) - len(compact_tokens)),
        "token_reduction_ratio": float((len(old_tokens) - len(compact_tokens)) / len(old_tokens)) if old_tokens else 0.0,
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        **relation_diag,
        "relation_token_count_compact": int(len(relation_tokens)),
    }


def encode_topology_target(
    topology_target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    graph = topology_target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    size = topology_target.get("size", [0, 0])
    node_index_by_id = _node_index_by_id(nodes)

    tokens: List[str] = ["<BOS>", "MANUAL_TOPOLOGY_V1", "SIZE"]
    tokens.extend(int_token(int(value), config=config) for value in size[:2])
    tokens.extend(["NODE_BLOCK", int_token(len(nodes), config=config)])
    for node in nodes:
        _append_compact_node(
            tokens,
            node,
            node_index_by_id=node_index_by_id,
            config=config,
            include_geometry=False,
            topology_mode=True,
        )
    relation_tokens, _diagnostics = _relation_block_tokens_and_diagnostics(
        relations,
        node_index_by_id=node_index_by_id,
        config=config,
    )
    tokens.extend(relation_tokens)
    tokens.extend(["RESIDUALS", int_token(len(graph.get("residuals", [])), config=config), "<EOS>"])
    return tokens


def encode_geometry_target(
    geometry_target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    tokens: List[str] = [
        "<BOS>",
        "MANUAL_GEOMETRY_V1",
        "GEOMETRY_BLOCK",
        _manual_role_token(str(geometry_target.get("role", ""))),
        "LABEL",
        int_token(int(geometry_target.get("label", 0)), config=config),
        _manual_geometry_token(str(geometry_target.get("geometry_model", "none"))),
    ]
    _append_geometry(tokens, geometry_target, config=config)
    tokens.append("<EOS>")
    return tokens


def geometry_token_lengths(
    geometry_targets: Iterable[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[int]:
    config = config or ParseGraphTokenizerConfig()
    return [len(encode_geometry_target(target, config=config)) for target in geometry_targets]
