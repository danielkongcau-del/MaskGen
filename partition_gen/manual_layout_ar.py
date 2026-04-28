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
    topology_node_index_by_id,
    _resolve_path,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target
from partition_gen.manual_topology_placeholder_geometry import (
    GeometryPlaceholderLibrary,
    decode_topology_tokens_to_target,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    TokenReader,
    dequantize,
    int_token,
    q_token,
    token_int,
    tokens_to_ids,
)


LAYOUT_CONDITION_TOKEN = "MANUAL_LAYOUT_CONDITION_V1"
LAYOUT_TARGET_TOKEN = "LAYOUT_TARGET"
LAYOUT_START_TOKEN = "MANUAL_LAYOUT_V1"


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {str(target.get("source_node_id")): target for target in geometry_targets}


def _node_index_by_id(nodes: Sequence[dict]) -> Dict[str, int]:
    return {str(node.get("id")): index for index, node in enumerate(nodes) if "id" in node}


def _frame_tokens(frame: dict, *, config: ParseGraphTokenizerConfig) -> List[str]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return [
        "FRAME",
        q_token(float(origin[0]), low=config.position_min, high=config.position_max, bins=config.position_bins),
        q_token(float(origin[1]), low=config.position_min, high=config.position_max, bins=config.position_bins),
        q_token(float(frame.get("scale", 1.0)), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
        q_token(
            float(frame.get("orientation", 0.0)),
            low=config.angle_min,
            high=config.angle_max,
            bins=config.angle_bins,
        ),
    ]


def layout_condition_prefix_tokens(
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
        LAYOUT_CONDITION_TOKEN,
        TOPOLOGY_CONTEXT_TOKEN,
        *topology_tokens[1:-1],
        LAYOUT_TARGET_TOKEN,
    ]


def encode_layout_target(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    size = topology_target.get("size", [0, 0])
    layout_rows: List[tuple[int, dict]] = []
    for index, node in enumerate(nodes):
        geometry_ref = node.get("geometry_ref")
        if not geometry_ref or str(geometry_ref) not in geometry_by_id:
            continue
        geometry_target = geometry_by_id[str(geometry_ref)]
        if "frame" not in geometry_target:
            continue
        layout_rows.append((int(index), geometry_target["frame"]))

    tokens: List[str] = ["<BOS>", LAYOUT_START_TOKEN, "SIZE"]
    tokens.extend(int_token(int(value), config=config) for value in size[:2])
    tokens.extend(["NODE_BLOCK", int_token(len(layout_rows), config=config)])
    for node_index, frame in layout_rows:
        tokens.extend(["NODE", int_token(int(node_index), config=config)])
        tokens.extend(_frame_tokens(frame, config=config))
        tokens.append("END_NODE")
    tokens.append("<EOS>")
    return tokens


def encode_conditioned_layout_target(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    layout_tokens = encode_layout_target(topology_target, geometry_targets, config=config)
    return [*layout_condition_prefix_tokens(topology_target, config=config), *layout_tokens[1:]]


def layout_start_index(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens):
        if str(token) == LAYOUT_START_TOKEN:
            return int(index)
    raise ValueError(f"{LAYOUT_START_TOKEN} not found in layout tokens")


def layout_condition_prefix_from_tokens(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    start = layout_start_index(tokens)
    return tokens[:start]


def extract_layout_tokens_from_conditioned(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    start = layout_start_index(tokens)
    return ["<BOS>", *tokens[start:]]


def topology_target_from_layout_conditioned_tokens(tokens: Sequence[str]) -> dict:
    tokens = [str(token) for token in tokens]
    if TOPOLOGY_CONTEXT_TOKEN not in tokens or LAYOUT_TARGET_TOKEN not in tokens:
        raise ValueError("Conditioned layout tokens must contain TOPOLOGY_CONTEXT and LAYOUT_TARGET")
    start = tokens.index(TOPOLOGY_CONTEXT_TOKEN) + 1
    end = tokens.index(LAYOUT_TARGET_TOKEN)
    topology_tokens = ["<BOS>", *tokens[start:end], "<EOS>"]
    return decode_topology_tokens_to_target(topology_tokens)


def _decode_layout_frame(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME")
    return {
        "origin": [
            dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins),
            dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins),
        ],
        "scale": dequantize(reader.next_q(), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
        "orientation": dequantize(reader.next_q(), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    }


def decode_layout_tokens_to_target(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    reader = TokenReader([str(token) for token in tokens])
    reader.expect("<BOS>")
    reader.expect(LAYOUT_START_TOKEN)
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]
    reader.expect("NODE_BLOCK")
    node_count = reader.next_int()
    nodes: List[dict] = []
    for _index in range(int(node_count)):
        reader.expect("NODE")
        node_index = reader.next_int()
        frame = _decode_layout_frame(reader, config=config)
        reader.expect("END_NODE")
        nodes.append({"node_index": int(node_index), "frame": frame})
    reader.expect("<EOS>")
    if reader.index != len(reader.tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(reader.tokens) - reader.index}")
    return {
        "format": "maskgen_manual_layout_target_v1",
        "target_type": "manual_parse_graph_layout_v1",
        "size": size,
        "nodes": nodes,
    }


def validate_layout_tokens(
    tokens: Sequence[str],
    *,
    topology_target: dict | None = None,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    errors: List[str] = []
    layout_target = None
    try:
        layout_target = decode_layout_tokens_to_target(tokens, config=config)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if layout_target is not None:
        node_indices = [int(row["node_index"]) for row in layout_target.get("nodes", []) or []]
        duplicates = sorted(index for index, count in Counter(node_indices).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate_node_indices:{duplicates}")
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
    return {
        "format": "maskgen_manual_layout_validation_v1",
        "valid": not errors,
        "errors": errors,
        "length": int(len(tokens)),
        "hit_eos": bool(tokens and str(tokens[-1]) == "<EOS>"),
        "target": layout_target,
    }


def build_layout_sequence_rows(
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

    for row in rows:
        topology_path = _resolve_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        tokens = encode_conditioned_layout_target(topology_target, geometry_targets, config=config)
        if max_tokens is not None and len(tokens) > int(max_tokens):
            skipped_too_long += 1
            continue
        layout_tokens = encode_layout_target(topology_target, geometry_targets, config=config)
        layout_target = decode_layout_tokens_to_target(layout_tokens, config=config)
        sequence_row = {
            "format": "maskgen_tokenized_parse_graph_v1",
            "tokenizer": "manual_layout_conditioned_v1",
            "source_topology": str(topology_path.as_posix()),
            "source_target": str(row.get("source_target", topology_path.as_posix())),
            "stem": row.get("stem"),
            "length": int(len(tokens)),
            "topology_length": int(len(encode_topology_target(topology_target, config=config))),
            "layout_length": int(len(layout_tokens)),
            "layout_node_count": int(len(layout_target["nodes"])),
            "loss_start_index": int(layout_start_index(tokens)),
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
        "format": "maskgen_manual_layout_tokenized_summary_v1",
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
    }
    return sequence_rows, summary


def _frame_errors(pred_frame: dict, target_frame: dict) -> dict:
    pred_origin = pred_frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    target_origin = target_frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    angle_diff = abs(float(pred_frame.get("orientation", 0.0)) - float(target_frame.get("orientation", 0.0)))
    return {
        "origin": float(math.dist(pred_origin, target_origin)),
        "scale": float(abs(float(pred_frame.get("scale", 0.0)) - float(target_frame.get("scale", 0.0)))),
        "orientation": float(min(angle_diff, 2.0 * math.pi - angle_diff)),
    }


def evaluate_layout_sample_rows(rows: Sequence[dict], *, top_k_invalid: int = 20) -> dict:
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

    for row_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        hit_eos_count += int(bool(tokens and tokens[-1] == "<EOS>"))
        topology_target = row.get("topology_target")
        validation = validate_layout_tokens(tokens, topology_target=topology_target)
        if not bool(validation["valid"]):
            reason = str(validation["errors"][0]) if validation["errors"] else "unknown"
            failure_histogram[reason] += 1
            if len(invalid_samples) < int(top_k_invalid):
                invalid_samples.append({"sample_index": row.get("sample_index", row_index), "errors": validation["errors"]})
            continue
        valid_count += 1
        layout_target = validation["target"] or {}
        node_counts.append(len(layout_target.get("nodes", []) or []))
        target_layout = row.get("target_layout")
        if target_layout is None:
            continue
        target_by_index = {int(item["node_index"]): item["frame"] for item in target_layout.get("nodes", []) or []}
        topology_nodes = list((topology_target or {}).get("parse_graph", {}).get("nodes", []) or [])
        for item in layout_target.get("nodes", []) or []:
            node_index = int(item["node_index"])
            if node_index not in target_by_index:
                continue
            errors = _frame_errors(item["frame"], target_by_index[node_index])
            origin_errors.append(errors["origin"])
            scale_errors.append(errors["scale"])
            orientation_errors.append(errors["orientation"])
            role = (
                str(topology_nodes[node_index].get("role", "unknown"))
                if 0 <= node_index < len(topology_nodes)
                else "unknown"
            )
            role_errors[role]["origin"].append(errors["origin"])
            role_errors[role]["scale"].append(errors["scale"])
            role_errors[role]["orientation"].append(errors["orientation"])

    def avg(values: Sequence[float]) -> float | None:
        return float(mean(values)) if values else None

    return {
        "format": "maskgen_manual_layout_sample_eval_v1",
        "sample_count": int(sample_count),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "hit_eos_count": int(hit_eos_count),
        "layout_node_count_mean": float(mean(node_counts)) if node_counts else None,
        "origin_mae": avg(origin_errors),
        "scale_mae": avg(scale_errors),
        "orientation_mae": avg(orientation_errors),
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


class LayoutGrammarState:
    def __init__(self, expected_node_indices: Sequence[int], *, config: ParseGraphTokenizerConfig | None = None) -> None:
        self.expected_node_indices = [int(value) for value in expected_node_indices]
        self.config = config or ParseGraphTokenizerConfig()
        self.phase = "manual"
        self.node_count = len(self.expected_node_indices)
        self.node_position = 0
        self.done = False
        self.errors: List[str] = []

    def allowed_token_strings(self) -> List[str]:
        if self.done:
            return []
        if self.phase == "manual":
            return [LAYOUT_START_TOKEN]
        if self.phase == "size":
            return ["SIZE"]
        if self.phase == "size_w":
            return [f"I_{self.config.position_max:.0f}".replace(".0", "")]
        if self.phase == "size_h":
            return [f"I_{self.config.position_max:.0f}".replace(".0", "")]
        if self.phase == "node_block":
            return ["NODE_BLOCK"]
        if self.phase == "node_count":
            return [int_token(self.node_count, config=self.config)]
        if self.phase == "node_start":
            return ["NODE"]
        if self.phase == "node_index":
            return [int_token(self.expected_node_indices[self.node_position], config=self.config)]
        if self.phase == "frame":
            return ["FRAME"]
        if self.phase in {"origin_x", "origin_y"}:
            return [f"Q_{index}" for index in range(int(self.config.position_bins))]
        if self.phase == "scale":
            return [f"Q_{index}" for index in range(int(self.config.scale_bins))]
        if self.phase == "orientation":
            return [f"Q_{index}" for index in range(int(self.config.angle_bins))]
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
        transitions = {
            "manual": "size",
            "size": "size_w",
            "size_w": "size_h",
            "size_h": "node_block",
            "node_block": "node_count",
            "node_count": "node_start" if self.node_count > 0 else "eos",
            "node_start": "node_index",
            "node_index": "frame",
            "frame": "origin_x",
            "origin_x": "origin_y",
            "origin_y": "scale",
            "scale": "orientation",
            "orientation": "end_node",
        }
        if self.phase in transitions:
            self.phase = transitions[self.phase]
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
def sample_layout_constrained(
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
    prefix = list(prefix_tokens or layout_condition_prefix_tokens(topology_target, config=config))
    missing = [token for token in prefix if token not in vocab]
    if missing:
        raise ValueError(f"Layout prefix contains tokens not in vocab: {missing}")
    expected_node_indices = renderable_geometry_node_indices(topology_target)
    state = LayoutGrammarState(expected_node_indices, config=config)
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
        "layout_tokens": extract_layout_tokens_from_conditioned(tokens),
        "length": int(len(ids)),
        "hit_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "stopped_reason": stopped_reason,
        "constraint_diagnostics": state.diagnostics(),
    }


def sample_model_conditioned_layout_rows(
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
    progress_label: str = "layout_sample",
) -> List[dict]:
    if not source_rows:
        raise ValueError("source_rows must contain layout token rows")
    rows: List[dict] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for sample_index in range(int(num_samples)):
            source_row = source_rows[sample_index % len(source_rows)]
            source_tokens = [str(token) for token in source_row.get("tokens", []) or []]
            topology_target = topology_target_from_layout_conditioned_tokens(source_tokens)
            target_layout = decode_layout_tokens_to_target(extract_layout_tokens_from_conditioned(source_tokens))
            sample = sample_layout_constrained(
                model,
                vocab,
                topology_target=topology_target,
                prefix_tokens=layout_condition_prefix_from_tokens(source_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=top_k,
                device=device,
            )
            rows.append(
                {
                    "format": "maskgen_manual_layout_sample_v1",
                    "sample_index": int(sample_index),
                    "sampling_mode": "layout_constrained",
                    "prefix_tokens": layout_condition_prefix_from_tokens(source_tokens),
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


def attach_layout_frames_to_topology(
    topology_target: dict,
    layout_target: dict,
    *,
    shape_library: GeometryPlaceholderLibrary | None = None,
    geometry_by_node_id: Dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    frame_by_index = {int(item["node_index"]): copy.deepcopy(item["frame"]) for item in layout_target.get("nodes", []) or []}
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
                output["layout_frame_source"] = "layout_ar"
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
            "layout_ar_attached": True,
            "attached_geometry_count": int(attached),
            "missing_geometry_count": int(missing),
            "attach_modes": dict(attach_modes),
        },
    }
    return target, copy.deepcopy(target["metadata"])


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
