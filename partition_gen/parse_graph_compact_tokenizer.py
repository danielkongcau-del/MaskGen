from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from partition_gen.parse_graph_relations import divides_target, inserted_in_container, relation_refs
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    TokenReader,
    _encode_frame,
    _encode_manual_convex_atoms,
    _encode_manual_polygon_geometry,
    _manual_geometry_token,
    _manual_relation_token,
    _manual_role_token,
    encode_generator_target,
    int_token,
)


_ROLE_TOKEN_TO_MANUAL_ROLE = {
    "ROLE_SUPPORT": "support_region",
    "ROLE_DIVIDER": "divider_region",
    "ROLE_INSERT": "insert_object",
    "ROLE_INSERT_GROUP": "insert_object_group",
    "ROLE_RESIDUAL": "residual_region",
    "ROLE_UNKNOWN": "unknown",
}

_MANUAL_ROLE_TO_ID_PREFIX = {
    "support_region": "support",
    "divider_region": "divider",
    "insert_object": "insert",
    "insert_object_group": "insert_group",
    "residual_region": "residual",
    "unknown": "node",
}

_GEOMETRY_TOKEN_TO_MODEL = {
    "GEOM_NONE": "none",
    "GEOM_POLYGON_CODE": "polygon_code",
    "GEOM_CONVEX_ATOMS": "convex_atoms",
    "GEOM_UNKNOWN": "unknown",
}

_RELATION_TOKEN_TO_TYPE = {
    "REL_INSERTED_IN": "inserted_in",
    "REL_CONTAINS": "contains",
    "REL_DIVIDES": "divides",
    "REL_ADJACENT_TO": "adjacent_to",
    "REL_UNKNOWN": "unknown",
}


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


def _generated_node_id(role: str, role_counts: Dict[str, int]) -> str:
    prefix = _MANUAL_ROLE_TO_ID_PREFIX.get(str(role), "node")
    index = int(role_counts.get(prefix, 0))
    role_counts[prefix] = index + 1
    return f"{prefix}_{index}"


def _node_id_for_index(node_ids: Sequence[str], index: int) -> str:
    if index < 0 or index >= len(node_ids):
        raise ValueError(f"Node index {index} is out of range for {len(node_ids)} decoded nodes")
    return str(node_ids[int(index)])


def _decode_topology_pair_block(reader: TokenReader, block_name: str, node_ids: Sequence[str]) -> List[tuple[str, str]]:
    reader.expect(block_name)
    pair_count = reader.next_int()
    pairs: List[tuple[str, str]] = []
    for _pair_index in range(int(pair_count)):
        left = _node_id_for_index(node_ids, reader.next_int())
        right = _node_id_for_index(node_ids, reader.next_int())
        pairs.append((left, right))
    reader.expect("END_BLOCK")
    return pairs


def decode_topology_tokens_to_target(tokens: Sequence[str]) -> dict:
    """Decode a `MANUAL_TOPOLOGY_V1` token sequence into a topology target JSON.

    The topology tokenizer stores only a binary geometry-reference flag. During
    decoding, a non-zero flag becomes `geometry_ref=<decoded node id>`, which is
    the convention expected by the split geometry targets and merge helper.
    """

    reader = TokenReader(tokens)
    reader.expect("<BOS>")
    reader.expect("MANUAL_TOPOLOGY_V1")
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]
    reader.expect("NODE_BLOCK")
    node_count = reader.next_int()

    nodes: List[dict] = []
    node_ids: List[str] = []
    role_counts: Dict[str, int] = {}
    child_indices_by_node_id: Dict[str, List[int]] = {}

    for _node_index in range(int(node_count)):
        reader.expect("NODE")
        role_token = reader.next()
        role = _ROLE_TOKEN_TO_MANUAL_ROLE.get(role_token, "unknown")
        node_id = _generated_node_id(role, role_counts)
        label = reader.next_int()
        renderable = bool(reader.next_int())
        is_reference_only = bool(reader.next_int())
        geometry_token = reader.next()
        geometry_model = _GEOMETRY_TOKEN_TO_MODEL.get(geometry_token, "unknown")
        geometry_ref_flag = reader.next_int()
        node = {
            "id": node_id,
            "role": role,
            "label": int(label),
            "renderable": renderable,
            "is_reference_only": is_reference_only,
            "geometry_model": geometry_model,
        }
        if geometry_ref_flag:
            node["geometry_ref"] = node_id
        if reader.index < len(reader.tokens) and reader.tokens[reader.index] == "CHILDREN":
            reader.expect("CHILDREN")
            child_count = reader.next_int()
            child_indices = [reader.next_int() for _child_index in range(int(child_count))]
            child_indices_by_node_id[node_id] = [int(index) for index in child_indices]
        reader.expect("END_NODE")
        nodes.append(node)
        node_ids.append(node_id)

    for node in nodes:
        child_indices = child_indices_by_node_id.get(str(node["id"]))
        if child_indices is not None:
            node["children"] = [_node_id_for_index(node_ids, int(index)) for index in child_indices]
            node["count"] = int(len(node["children"]))

    relations: List[dict] = []
    for node in nodes:
        if str(node.get("role")) == "insert_object_group":
            for child_id in node.get("children", []) or []:
                relations.append({"type": "contains", "parent": node["id"], "child": str(child_id)})

    for object_id, container_id in _decode_topology_pair_block(reader, "REL_BLOCK_INSERTED_IN", node_ids):
        relations.append(
            {
                "type": "inserted_in",
                "object": object_id,
                "container": container_id,
                "support": container_id,
            }
        )
    for divider_id, target_id in _decode_topology_pair_block(reader, "REL_BLOCK_DIVIDES", node_ids):
        relations.append(
            {
                "type": "divides",
                "divider": divider_id,
                "target": target_id,
                "support": target_id,
            }
        )
    for left_id, right_id in _decode_topology_pair_block(reader, "REL_BLOCK_ADJACENT_TO", node_ids):
        relations.append({"type": "adjacent_to", "faces": [left_id, right_id]})

    reader.expect("REL_BLOCK_OTHER")
    other_count = reader.next_int()
    for _relation_index in range(int(other_count)):
        relation_token = reader.next()
        ref_count = reader.next_int()
        refs = [_node_id_for_index(node_ids, reader.next_int()) for _ref_index in range(int(ref_count))]
        relations.append(
            {
                "type": _RELATION_TOKEN_TO_TYPE.get(relation_token, "unknown"),
                "refs": refs,
            }
        )
    reader.expect("END_BLOCK")
    reader.expect("RESIDUALS")
    residual_count = reader.next_int()
    residuals = [{"reason": "tokenized_residual_placeholder"} for _ in range(int(residual_count))]
    reader.expect("<EOS>")
    if reader.index != len(reader.tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(reader.tokens) - reader.index}")

    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": size,
        "parse_graph": {
            "nodes": nodes,
            "relations": relations,
            "residuals": residuals,
        },
        "metadata": {
            "decoded_from_tokens": True,
            "tokenizer": "manual_topology_v1",
        },
    }


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
