from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target, encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, int_token, tokens_to_ids


CONDITION_TOKEN = "MANUAL_GEOMETRY_CONDITION_V1"
TOPOLOGY_CONTEXT_TOKEN = "TOPOLOGY_CONTEXT"
TARGET_NODE_TOKEN = "TARGET_NODE"
GEOMETRY_TARGET_TOKEN = "GEOMETRY_TARGET"
GEOMETRY_START_TOKEN = "MANUAL_GEOMETRY_V1"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve_path(value: object, *, split_root: Path, manifest_parent: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    for base in (manifest_parent, split_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def topology_node_index_by_id(topology_target: dict) -> Dict[str, int]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    return {str(node.get("id")): index for index, node in enumerate(nodes) if "id" in node}


def renderable_geometry_node_indices(topology_target: dict) -> List[int]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    return [
        int(index)
        for index, node in enumerate(nodes)
        if bool(node.get("geometry_ref"))
        and bool(node.get("renderable", True))
        and not bool(node.get("is_reference_only", False))
    ]


def geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {str(target.get("source_node_id")): target for target in geometry_targets}


def topology_condition_prefix_tokens(
    topology_target: dict,
    *,
    target_node_index: int,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    """Encode full topology context plus the target node index, without geometry output tokens."""

    config = config or ParseGraphTokenizerConfig()
    topology_tokens = encode_topology_target(topology_target, config=config)
    if not topology_tokens or topology_tokens[0] != "<BOS>" or topology_tokens[-1] != "<EOS>":
        raise ValueError("Expected encode_topology_target to return <BOS> ... <EOS>")
    return [
        "<BOS>",
        CONDITION_TOKEN,
        TOPOLOGY_CONTEXT_TOKEN,
        *topology_tokens[1:-1],
        TARGET_NODE_TOKEN,
        int_token(int(target_node_index), config=config),
        GEOMETRY_TARGET_TOKEN,
    ]


def conditioned_geometry_prefix_tokens(
    topology_target: dict,
    *,
    target_node_index: int,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    """Build the forced prefix used at inference for a topology-conditioned geometry sample."""

    config = config or ParseGraphTokenizerConfig()
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    node = nodes[int(target_node_index)]
    geometry_head = encode_geometry_target(
        {
            "role": str(node.get("role", "")),
            "label": int(node.get("label", 0)),
            "geometry_model": str(node.get("geometry_model", "polygon_code")),
        },
        config=config,
    )
    return [
        *topology_condition_prefix_tokens(
            topology_target,
            target_node_index=int(target_node_index),
            config=config,
        ),
        *geometry_head[1:7],
    ]


def encode_conditioned_geometry_target(
    topology_target: dict,
    geometry_target: dict,
    *,
    target_node_index: int,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    geometry_tokens = encode_geometry_target(geometry_target, config=config)
    return [
        *topology_condition_prefix_tokens(
            topology_target,
            target_node_index=int(target_node_index),
            config=config,
        ),
        *geometry_tokens[1:],
    ]


def geometry_start_index(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens):
        if str(token) == GEOMETRY_START_TOKEN:
            return int(index)
    raise ValueError(f"{GEOMETRY_START_TOKEN} not found in conditioned geometry tokens")


def conditioned_geometry_prefix_from_tokens(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    start = geometry_start_index(tokens)
    if len(tokens) < start + 6:
        raise ValueError("Conditioned geometry sequence is too short for a forced geometry prefix")
    expected = [GEOMETRY_START_TOKEN, "GEOMETRY_BLOCK"]
    if tokens[start : start + 2] != expected:
        raise ValueError(f"Expected geometry start {expected}, got {tokens[start:start + 2]}")
    if tokens[start + 3] != "LABEL":
        raise ValueError(f"Expected LABEL at conditioned geometry prefix, got {tokens[start + 3]}")
    return tokens[: start + 6]


def extract_geometry_tokens_from_conditioned(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    start = geometry_start_index(tokens)
    return ["<BOS>", *tokens[start:]]


def build_conditioned_geometry_sequence_rows(
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
    skipped_missing_geometry = 0
    skipped_too_long = 0
    lengths: List[int] = []
    topology_lengths: List[int] = []
    geometry_lengths: List[int] = []

    for row in rows:
        topology_path = _resolve_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        geometry_by_id = geometry_targets_by_source_node_id(geometry_targets)
        node_index_by_id = topology_node_index_by_id(topology_target)
        topology_token_count = len(encode_topology_target(topology_target, config=config))

        for source_node_id, geometry_target in geometry_by_id.items():
            if source_node_id not in node_index_by_id:
                skipped_missing_geometry += 1
                continue
            target_node_index = int(node_index_by_id[source_node_id])
            tokens = encode_conditioned_geometry_target(
                topology_target,
                geometry_target,
                target_node_index=target_node_index,
                config=config,
            )
            if max_tokens is not None and len(tokens) > int(max_tokens):
                skipped_too_long += 1
                continue
            geometry_token_count = len(encode_geometry_target(geometry_target, config=config))
            sequence_row = {
                "format": "maskgen_tokenized_parse_graph_v1",
                "tokenizer": "manual_geometry_conditioned_v1",
                "source_topology": str(topology_path.as_posix()),
                "source_target": str(row.get("source_target", topology_path.as_posix())),
                "stem": row.get("stem"),
                "source_node_id": str(source_node_id),
                "target_node_index": int(target_node_index),
                "length": int(len(tokens)),
                "topology_length": int(topology_token_count),
                "geometry_length": int(geometry_token_count),
                "loss_start_index": int(geometry_start_index(tokens)),
                "tokens": tokens,
            }
            if include_token_ids:
                sequence_row["ids"] = tokens_to_ids(tokens, vocab)
            sequence_rows.append(sequence_row)
            lengths.append(len(tokens))
            topology_lengths.append(topology_token_count)
            geometry_lengths.append(geometry_token_count)

    summary = {
        "format": "maskgen_manual_conditioned_geometry_tokenized_summary_v1",
        "split_root": str(split_root.as_posix()),
        "sample_count": int(len(rows)),
        "written_conditioned_geometry": int(len(sequence_rows)),
        "skipped_missing_geometry": int(skipped_missing_geometry),
        "skipped_too_long": int(skipped_too_long),
        "conditioned_length_mean": float(mean(lengths)) if lengths else None,
        "conditioned_length_max": int(max(lengths)) if lengths else None,
        "topology_length_mean": float(mean(topology_lengths)) if topology_lengths else None,
        "topology_length_max": int(max(topology_lengths)) if topology_lengths else None,
        "geometry_length_mean": float(mean(geometry_lengths)) if geometry_lengths else None,
        "geometry_length_max": int(max(geometry_lengths)) if geometry_lengths else None,
    }
    return sequence_rows, summary
