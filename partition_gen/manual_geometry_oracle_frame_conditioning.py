from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from partition_gen.manual_geometry_conditioning import (
    GEOMETRY_START_TOKEN,
    build_conditioned_geometry_sequence_rows,
    extract_geometry_tokens_from_conditioned,
    geometry_targets_by_source_node_id,
    iter_jsonl,
    load_json,
    topology_condition_prefix_tokens,
    topology_node_index_by_id,
    _resolve_path,
)
from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target, encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, tokens_to_ids


SHAPE_START_TOKENS = ("POLYS", "ATOMS")


def geometry_shape_start_index(tokens: Sequence[str]) -> int:
    tokens = [str(token) for token in tokens]
    try:
        geometry_start = tokens.index(GEOMETRY_START_TOKEN)
    except ValueError as exc:
        raise ValueError(f"{GEOMETRY_START_TOKEN} not found in geometry tokens") from exc
    for index in range(int(geometry_start), len(tokens)):
        if tokens[index] in SHAPE_START_TOKENS:
            return int(index)
    raise ValueError("No POLYS/ATOMS shape start token found in geometry tokens")


def oracle_frame_conditioned_geometry_prefix_tokens(
    topology_target: dict,
    geometry_target: dict,
    *,
    target_node_index: int,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    geometry_tokens = encode_geometry_target(geometry_target, config=config)
    shape_start = geometry_shape_start_index(geometry_tokens)
    return [
        *topology_condition_prefix_tokens(
            topology_target,
            target_node_index=int(target_node_index),
            config=config,
        ),
        *geometry_tokens[1:shape_start],
    ]


def encode_oracle_frame_conditioned_geometry_target(
    topology_target: dict,
    geometry_target: dict,
    *,
    target_node_index: int,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    geometry_tokens = encode_geometry_target(geometry_target, config=config)
    shape_start = geometry_shape_start_index(geometry_tokens)
    return [
        *oracle_frame_conditioned_geometry_prefix_tokens(
            topology_target,
            geometry_target,
            target_node_index=int(target_node_index),
            config=config,
        ),
        *geometry_tokens[shape_start:],
    ]


def oracle_frame_geometry_prefix_from_tokens(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    shape_start = geometry_shape_start_index(tokens)
    return tokens[:shape_start]


def extract_geometry_tokens_from_oracle_frame_conditioned(tokens: Sequence[str]) -> List[str]:
    return extract_geometry_tokens_from_conditioned(tokens)


def build_oracle_frame_geometry_sequence_rows(
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
    prefix_lengths: List[int] = []

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
            tokens = encode_oracle_frame_conditioned_geometry_target(
                topology_target,
                geometry_target,
                target_node_index=target_node_index,
                config=config,
            )
            if max_tokens is not None and len(tokens) > int(max_tokens):
                skipped_too_long += 1
                continue
            geometry_tokens = encode_geometry_target(geometry_target, config=config)
            loss_start_index = int(geometry_shape_start_index(tokens))
            sequence_row = {
                "format": "maskgen_tokenized_parse_graph_v1",
                "tokenizer": "manual_oracle_frame_geometry_conditioned_v1",
                "source_topology": str(topology_path.as_posix()),
                "source_target": str(row.get("source_target", topology_path.as_posix())),
                "stem": row.get("stem"),
                "source_node_id": str(source_node_id),
                "target_node_index": int(target_node_index),
                "length": int(len(tokens)),
                "topology_length": int(topology_token_count),
                "geometry_length": int(len(geometry_tokens)),
                "oracle_frame_prefix_length": int(loss_start_index),
                "loss_start_index": loss_start_index,
                "tokens": tokens,
            }
            if include_token_ids:
                sequence_row["ids"] = tokens_to_ids(tokens, vocab)
            sequence_rows.append(sequence_row)
            lengths.append(len(tokens))
            topology_lengths.append(topology_token_count)
            geometry_lengths.append(len(geometry_tokens))
            prefix_lengths.append(loss_start_index)

    summary = {
        "format": "maskgen_manual_oracle_frame_geometry_tokenized_summary_v1",
        "split_root": str(split_root.as_posix()),
        "sample_count": int(len(rows)),
        "written_oracle_frame_geometry": int(len(sequence_rows)),
        "skipped_missing_geometry": int(skipped_missing_geometry),
        "skipped_too_long": int(skipped_too_long),
        "conditioned_length_mean": float(mean(lengths)) if lengths else None,
        "conditioned_length_max": int(max(lengths)) if lengths else None,
        "topology_length_mean": float(mean(topology_lengths)) if topology_lengths else None,
        "topology_length_max": int(max(topology_lengths)) if topology_lengths else None,
        "geometry_length_mean": float(mean(geometry_lengths)) if geometry_lengths else None,
        "geometry_length_max": int(max(geometry_lengths)) if geometry_lengths else None,
        "oracle_frame_prefix_length_mean": float(mean(prefix_lengths)) if prefix_lengths else None,
        "oracle_frame_prefix_length_max": int(max(prefix_lengths)) if prefix_lengths else None,
    }
    return sequence_rows, summary


def build_legacy_conditioned_geometry_sequence_rows(*args, **kwargs):
    return build_conditioned_geometry_sequence_rows(*args, **kwargs)
