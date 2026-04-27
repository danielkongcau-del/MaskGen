from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

import torch

from partition_gen.manual_geometry_constrained_sampling import (
    GeometryConstrainedSamplerConfig,
    sample_geometry_constrained,
)
from partition_gen.manual_geometry_sample_validation import (
    geometry_prefix_from_tokens,
    geometry_prefix_tokens,
    parse_geometry_structure,
    validate_geometry_tokens,
)
from partition_gen.parse_graph_tokenizer import ids_to_tokens


def _percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * float(q)))
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _numeric_stats(values: Sequence[int]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    int_values = [int(value) for value in values]
    return {
        "count": int(len(int_values)),
        "mean": float(mean(int_values)),
        "min": int(min(int_values)),
        "median": float(median(int_values)),
        "p90": _percentile(int_values, 0.90),
        "p95": _percentile(int_values, 0.95),
        "p99": _percentile(int_values, 0.99),
        "max": int(max(int_values)),
    }


def evaluate_geometry_sample_rows(
    rows: Sequence[dict],
    *,
    top_k_invalid: int = 20,
) -> Dict[str, object]:
    sample_count = int(len(rows))
    valid_count = 0
    hit_eos_count = 0
    failure_histogram: Counter[str] = Counter()
    invalid_samples: List[dict] = []
    lengths: List[int] = []
    valid_lengths: List[int] = []
    polygon_counts: List[int] = []
    hole_counts: List[int] = []
    atom_counts: List[int] = []
    point_totals: List[int] = []
    point_counts: List[int] = []
    roles: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    geometry_models: Counter[str] = Counter()
    parse_failures: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        lengths.append(int(row.get("length", len(tokens))))
        hit_eos_count += int("<EOS>" in tokens)
        validation = validate_geometry_tokens(tokens)
        if not bool(validation["valid"]):
            reason = str(validation["errors"][0]) if validation["errors"] else "unknown"
            failure_histogram[reason] += 1
            if len(invalid_samples) < int(top_k_invalid):
                invalid_samples.append(
                    {
                        "sample_index": row.get("sample_index", row_index),
                        "length": int(len(tokens)),
                        "errors": list(validation["errors"]),
                    }
                )
            continue
        valid_count += 1
        try:
            parsed = parse_geometry_structure(tokens)
        except Exception as exc:
            parse_failures[f"{type(exc).__name__}: {exc}"] += 1
            continue
        valid_lengths.append(int(parsed["length"]))
        polygon_counts.append(int(parsed["polygon_count"]))
        hole_counts.append(int(parsed["hole_count"]))
        atom_counts.append(int(parsed["atom_count"]))
        point_totals.append(int(parsed["point_total"]))
        point_counts.extend(int(value) for value in parsed["point_counts"])
        roles[str(parsed["role"])] += 1
        labels[str(parsed["label"])] += 1
        geometry_models[str(parsed["geometry_model"])] += 1

    return {
        "format": "maskgen_manual_geometry_sample_eval_v1",
        "sample_count": sample_count,
        "hit_eos_count": int(hit_eos_count),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "valid_structure_count": int(len(valid_lengths)),
        "lengths": _numeric_stats(lengths),
        "valid_lengths": _numeric_stats(valid_lengths),
        "polygon_counts": _numeric_stats(polygon_counts),
        "hole_counts": _numeric_stats(hole_counts),
        "atom_counts": _numeric_stats(atom_counts),
        "point_totals": _numeric_stats(point_totals),
        "point_counts": _numeric_stats(point_counts),
        "role_histogram": dict(sorted(roles.items())),
        "label_histogram": dict(sorted(labels.items(), key=lambda item: int(item[0]))),
        "geometry_model_histogram": dict(sorted(geometry_models.items())),
        "failure_reason_histogram": dict(failure_histogram.most_common()),
        "parse_failure_histogram": dict(parse_failures.most_common()),
        "invalid_samples": invalid_samples,
    }


def _prefix_rows_from_source_rows(source_rows: Sequence[dict], num_samples: int) -> List[List[str]]:
    prefixes: List[List[str]] = []
    if not source_rows:
        return prefixes
    for index in range(int(num_samples)):
        row = source_rows[index % len(source_rows)]
        prefixes.append(geometry_prefix_from_tokens([str(token) for token in row.get("tokens", []) or []]))
    return prefixes


def sample_model_geometry_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    source_rows: Sequence[dict] | None = None,
    prefix_role: str = "support_region",
    prefix_label: int = 0,
    prefix_geometry_model: str = "polygon_code",
    constraint_config: GeometryConstrainedSamplerConfig | None = None,
    progress_every: int = 0,
    progress_label: str = "geometry_sample",
) -> List[dict]:
    device = torch.device(device)
    inverse_mode = "constrained" if constraint_config is not None else "unconstrained"
    bos_id = int(vocab["<BOS>"])
    eos_id = int(vocab["<EOS>"])
    prefixes = _prefix_rows_from_source_rows(source_rows or [], int(num_samples))
    if not prefixes:
        prefixes = [
            geometry_prefix_tokens(role=prefix_role, label=int(prefix_label), geometry_model=prefix_geometry_model)
            for _ in range(int(num_samples))
        ]
    was_training = bool(model.training)
    model.eval()
    rows: List[dict] = []
    try:
        with torch.no_grad():
            for sample_index, prefix in enumerate(prefixes[: int(num_samples)]):
                if constraint_config is not None:
                    sample = sample_geometry_constrained(
                        model,
                        vocab,
                        prefix_tokens=prefix,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=top_k,
                        constraint_config=constraint_config,
                        device=device,
                    )
                    tokens = [str(token) for token in sample["tokens"]]
                    ids = [int(value) for value in sample["ids"]]
                    hit_eos = bool(sample["hit_eos"])
                    stopped_reason = sample.get("stopped_reason")
                    diagnostics = sample.get("constraint_diagnostics")
                else:
                    prefix_ids = [int(vocab[token]) for token in prefix if token in vocab]
                    if not prefix_ids:
                        prefix_ids = [bos_id]
                    generated = model.generate(
                        torch.tensor([prefix_ids], dtype=torch.long, device=device),
                        max_new_tokens=int(max_new_tokens),
                        eos_id=eos_id,
                        temperature=float(temperature),
                        top_k=top_k,
                    )[0].tolist()
                    ids = [int(value) for value in generated]
                    tokens = ids_to_tokens(ids, vocab)
                    hit_eos = bool(eos_id in ids)
                    stopped_reason = "eos" if hit_eos else "max_new_tokens"
                    diagnostics = None
                rows.append(
                    {
                        "format": "maskgen_manual_geometry_sample_v1",
                        "sample_index": int(sample_index),
                        "sampling_mode": inverse_mode,
                        "prefix_tokens": list(prefix),
                        "length": int(len(tokens)),
                        "hit_eos": bool(hit_eos),
                        "stopped_reason": stopped_reason,
                        "tokens": tokens,
                        "ids": ids,
                        "constraint_diagnostics": diagnostics,
                    }
                )
                if int(progress_every) > 0 and (sample_index + 1) % int(progress_every) == 0:
                    print(f"{progress_label} {sample_index + 1}/{num_samples}")
    finally:
        if was_training:
            model.train()
    return rows


def write_geometry_sample_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def write_geometry_sample_evaluation_markdown(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Geometry AR Evaluation",
        "",
        f"- samples: {summary.get('sample_count')}",
        f"- valid: {summary.get('valid_count')} ({summary.get('valid_rate')})",
        f"- hit_eos: {summary.get('hit_eos_count')}",
        "",
        "| metric | mean | p95 | max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("valid_lengths", "polygon_counts", "hole_counts", "atom_counts", "point_totals", "point_counts"):
        stats = summary.get(key, {}) or {}
        lines.append(f"| {key} | {stats.get('mean')} | {stats.get('p95')} | {stats.get('max')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
