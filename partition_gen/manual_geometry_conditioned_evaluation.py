from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from partition_gen.manual_geometry_conditioning import (
    conditioned_geometry_prefix_from_tokens,
    extract_geometry_tokens_from_conditioned,
)
from partition_gen.manual_geometry_oracle_frame_conditioning import (
    extract_geometry_tokens_from_oracle_frame_conditioned,
    oracle_frame_geometry_prefix_from_tokens,
)
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig, sample_geometry_constrained


def _conditioned_prefix_rows_from_source_rows(source_rows: Sequence[dict], num_samples: int) -> List[List[str]]:
    prefixes: List[List[str]] = []
    if not source_rows:
        return prefixes
    for index in range(int(num_samples)):
        row = source_rows[index % len(source_rows)]
        prefixes.append(conditioned_geometry_prefix_from_tokens([str(token) for token in row.get("tokens", []) or []]))
    return prefixes


def _oracle_frame_prefix_rows_from_source_rows(source_rows: Sequence[dict], num_samples: int) -> List[List[str]]:
    prefixes: List[List[str]] = []
    if not source_rows:
        return prefixes
    for index in range(int(num_samples)):
        row = source_rows[index % len(source_rows)]
        prefixes.append(oracle_frame_geometry_prefix_from_tokens([str(token) for token in row.get("tokens", []) or []]))
    return prefixes


def sample_model_conditioned_geometry_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    source_rows: Sequence[dict],
    constraint_config: GeometryConstrainedSamplerConfig | None = None,
    progress_every: int = 0,
    progress_label: str = "conditioned_geometry_sample",
) -> List[dict]:
    device = torch.device(device)
    prefixes = _conditioned_prefix_rows_from_source_rows(source_rows, int(num_samples))
    if not prefixes:
        raise ValueError("source_rows must contain conditioned geometry token rows")
    was_training = bool(model.training)
    model.eval()
    rows: List[dict] = []
    try:
        with torch.no_grad():
            for sample_index, prefix in enumerate(prefixes[: int(num_samples)]):
                sample = sample_geometry_constrained(
                    model,
                    vocab,
                    prefix_tokens=prefix,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=top_k,
                    constraint_config=constraint_config or GeometryConstrainedSamplerConfig(),
                    device=device,
                )
                conditioned_tokens = [str(token) for token in sample["tokens"]]
                geometry_tokens = extract_geometry_tokens_from_conditioned(conditioned_tokens)
                rows.append(
                    {
                        "format": "maskgen_manual_conditioned_geometry_sample_v1",
                        "sample_index": int(sample_index),
                        "sampling_mode": "conditioned_constrained",
                        "prefix_tokens": list(prefix),
                        "length": int(len(geometry_tokens)),
                        "conditioned_length": int(len(conditioned_tokens)),
                        "hit_eos": bool(sample["hit_eos"]),
                        "stopped_reason": sample.get("stopped_reason"),
                        "tokens": geometry_tokens,
                        "conditioned_tokens": conditioned_tokens,
                        "ids": sample["ids"],
                        "constraint_diagnostics": sample.get("constraint_diagnostics"),
                    }
                )
                if int(progress_every) > 0 and (sample_index + 1) % int(progress_every) == 0:
                    print(f"{progress_label} {sample_index + 1}/{num_samples}")
    finally:
        if was_training:
            model.train()
    return rows


def sample_model_oracle_frame_geometry_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    source_rows: Sequence[dict],
    constraint_config: GeometryConstrainedSamplerConfig | None = None,
    progress_every: int = 0,
    progress_label: str = "oracle_frame_geometry_sample",
) -> List[dict]:
    device = torch.device(device)
    prefixes = _oracle_frame_prefix_rows_from_source_rows(source_rows, int(num_samples))
    if not prefixes:
        raise ValueError("source_rows must contain oracle-frame geometry token rows")
    was_training = bool(model.training)
    model.eval()
    rows: List[dict] = []
    try:
        with torch.no_grad():
            for sample_index, prefix in enumerate(prefixes[: int(num_samples)]):
                sample = sample_geometry_constrained(
                    model,
                    vocab,
                    prefix_tokens=prefix,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=top_k,
                    constraint_config=constraint_config or GeometryConstrainedSamplerConfig(),
                    device=device,
                )
                conditioned_tokens = [str(token) for token in sample["tokens"]]
                geometry_tokens = extract_geometry_tokens_from_oracle_frame_conditioned(conditioned_tokens)
                rows.append(
                    {
                        "format": "maskgen_manual_oracle_frame_geometry_sample_v1",
                        "sample_index": int(sample_index),
                        "sampling_mode": "oracle_frame_constrained",
                        "prefix_tokens": list(prefix),
                        "length": int(len(geometry_tokens)),
                        "conditioned_length": int(len(conditioned_tokens)),
                        "hit_eos": bool(sample["hit_eos"]),
                        "stopped_reason": sample.get("stopped_reason"),
                        "tokens": geometry_tokens,
                        "conditioned_tokens": conditioned_tokens,
                        "ids": sample["ids"],
                        "constraint_diagnostics": sample.get("constraint_diagnostics"),
                    }
                )
                if int(progress_every) > 0 and (sample_index + 1) % int(progress_every) == 0:
                    print(f"{progress_label} {sample_index + 1}/{num_samples}")
    finally:
        if was_training:
            model.train()
    return rows
