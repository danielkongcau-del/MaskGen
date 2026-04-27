from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_tokenizer import ids_to_tokens


def _int_value(token: str) -> int:
    if not str(token).startswith("I_"):
        raise ValueError(f"Expected integer token, got {token}")
    return int(str(token).split("_", 1)[1])


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


def parse_topology_structure(tokens: Sequence[str]) -> Dict[str, object]:
    """Parse a validated manual topology token sequence into coarse structural stats."""

    tokens = [str(token) for token in tokens]
    index = 0

    def expect(expected: str) -> None:
        nonlocal index
        actual = tokens[index] if index < len(tokens) else None
        if actual != expected:
            raise ValueError(f"Expected {expected} at {index}, got {actual}")
        index += 1

    def next_int() -> int:
        nonlocal index
        if index >= len(tokens):
            raise ValueError(f"Expected int at {index}, got end")
        value = _int_value(tokens[index])
        index += 1
        return int(value)

    expect("<BOS>")
    expect("MANUAL_TOPOLOGY_V1")
    expect("SIZE")
    size = [next_int(), next_int()]
    expect("NODE_BLOCK")
    node_count = next_int()

    roles: Counter[str] = Counter()
    labels: Counter[int] = Counter()
    role_labels: Counter[str] = Counter()
    renderable: Counter[int] = Counter()
    reference_only: Counter[int] = Counter()
    geometry_models: Counter[str] = Counter()
    geometry_refs: Counter[int] = Counter()
    insert_group_child_counts: List[int] = []
    insert_group_child_total = 0

    for _node_index in range(int(node_count)):
        expect("NODE")
        role = tokens[index]
        index += 1
        label = next_int()
        node_renderable = next_int()
        node_reference_only = next_int()
        geometry_model = tokens[index]
        index += 1
        geometry_ref = next_int()
        child_count = None
        if index < len(tokens) and tokens[index] == "CHILDREN":
            index += 1
            child_count = next_int()
            index += int(child_count)
        expect("END_NODE")

        roles[str(role)] += 1
        labels[int(label)] += 1
        role_labels[f"{role}|{label}"] += 1
        renderable[int(node_renderable)] += 1
        reference_only[int(node_reference_only)] += 1
        geometry_models[str(geometry_model)] += 1
        geometry_refs[int(geometry_ref)] += 1
        if child_count is not None:
            insert_group_child_counts.append(int(child_count))
            insert_group_child_total += int(child_count)

    relation_counts: Dict[str, int] = {}
    for block_name in ("REL_BLOCK_INSERTED_IN", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO"):
        expect(block_name)
        pair_count = next_int()
        index += 2 * int(pair_count)
        expect("END_BLOCK")
        relation_counts[block_name] = int(pair_count)

    expect("REL_BLOCK_OTHER")
    other_count = next_int()
    for _relation_index in range(int(other_count)):
        index += 1
        ref_count = next_int()
        index += int(ref_count)
    expect("END_BLOCK")
    relation_counts["REL_BLOCK_OTHER"] = int(other_count)

    expect("RESIDUALS")
    residual_count = next_int()
    expect("<EOS>")
    if index != len(tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(tokens) - index}")

    relation_counts["RESIDUALS"] = int(residual_count)
    return {
        "length": int(len(tokens)),
        "size": size,
        "node_count": int(node_count),
        "roles": dict(roles),
        "labels": {str(key): int(value) for key, value in labels.items()},
        "role_labels": dict(role_labels),
        "renderable": {str(key): int(value) for key, value in renderable.items()},
        "reference_only": {str(key): int(value) for key, value in reference_only.items()},
        "geometry_models": dict(geometry_models),
        "geometry_refs": {str(key): int(value) for key, value in geometry_refs.items()},
        "insert_group_count": int(len(insert_group_child_counts)),
        "insert_group_child_total": int(insert_group_child_total),
        "insert_group_child_counts": insert_group_child_counts,
        "relation_counts": relation_counts,
    }


def evaluate_topology_sample_rows(
    rows: Sequence[dict],
    *,
    top_k_invalid: int = 20,
) -> Dict[str, object]:
    sample_count = int(len(rows))
    validation_results: List[dict] = []
    failure_histogram: Counter[str] = Counter()
    invalid_samples: List[dict] = []
    lengths: List[int] = []
    valid_lengths: List[int] = []
    node_counts: List[int] = []
    insert_group_counts: List[int] = []
    insert_group_child_totals: List[int] = []
    insert_group_child_counts: List[int] = []
    relation_totals: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    role_labels: Counter[str] = Counter()
    renderable: Counter[str] = Counter()
    reference_only: Counter[str] = Counter()
    geometry_models: Counter[str] = Counter()
    geometry_refs: Counter[str] = Counter()
    parse_failures: Counter[str] = Counter()
    hit_eos_count = 0

    for row_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", [])]
        lengths.append(int(row.get("length", len(tokens))))
        hit_eos_count += int("<EOS>" in tokens)
        validation = validate_topology_tokens(tokens)
        validation_results.append(validation)
        if not validation["valid"]:
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
        try:
            parsed = parse_topology_structure(tokens)
        except Exception as error:  # noqa: BLE001 - diagnostic summary records parser failures.
            parse_failures[f"{type(error).__name__}: {error}"] += 1
            continue

        valid_lengths.append(int(parsed["length"]))
        node_counts.append(int(parsed["node_count"]))
        insert_group_counts.append(int(parsed["insert_group_count"]))
        insert_group_child_totals.append(int(parsed["insert_group_child_total"]))
        insert_group_child_counts.extend(int(value) for value in parsed["insert_group_child_counts"])
        relation_totals.update({str(key): int(value) for key, value in parsed["relation_counts"].items()})
        roles.update({str(key): int(value) for key, value in parsed["roles"].items()})
        labels.update({str(key): int(value) for key, value in parsed["labels"].items()})
        role_labels.update({str(key): int(value) for key, value in parsed["role_labels"].items()})
        renderable.update({str(key): int(value) for key, value in parsed["renderable"].items()})
        reference_only.update({str(key): int(value) for key, value in parsed["reference_only"].items()})
        geometry_models.update({str(key): int(value) for key, value in parsed["geometry_models"].items()})
        geometry_refs.update({str(key): int(value) for key, value in parsed["geometry_refs"].items()})

    valid_count = sum(1 for result in validation_results if result["valid"])
    valid_structure_count = len(node_counts)
    denominator = max(1, valid_structure_count)
    return {
        "format": "maskgen_manual_topology_sample_eval_v1",
        "sample_count": sample_count,
        "hit_eos_count": int(hit_eos_count),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "valid_structure_count": int(valid_structure_count),
        "lengths": _numeric_stats(lengths),
        "valid_lengths": _numeric_stats(valid_lengths),
        "node_counts": _numeric_stats(node_counts),
        "insert_group_counts": _numeric_stats(insert_group_counts),
        "insert_group_child_totals": _numeric_stats(insert_group_child_totals),
        "insert_group_child_counts": _numeric_stats(insert_group_child_counts),
        "role_histogram": dict(sorted(roles.items())),
        "label_histogram": dict(sorted(labels.items(), key=lambda item: int(item[0]))),
        "role_label_histogram": dict(sorted(role_labels.items())),
        "renderable_histogram": dict(sorted(renderable.items())),
        "reference_only_histogram": dict(sorted(reference_only.items())),
        "geometry_model_histogram": dict(sorted(geometry_models.items())),
        "geometry_ref_histogram": dict(sorted(geometry_refs.items())),
        "relation_totals": dict(sorted(relation_totals.items())),
        "relation_mean_per_valid_sample": {
            key: float(value / denominator) for key, value in sorted(relation_totals.items())
        },
        "failure_reason_histogram": dict(failure_histogram.most_common()),
        "parse_failure_histogram": dict(parse_failures.most_common()),
        "invalid_samples": invalid_samples,
    }


def sample_model_topology_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    constraint_config=None,
    progress_every: int = 0,
    progress_label: str = "topology_sample",
) -> List[dict]:
    import torch

    from partition_gen.manual_topology_constrained_sampling import sample_topology_constrained

    device = torch.device(device)
    inverse_mode = "constrained" if constraint_config is not None else "unconstrained"
    bos_id = int(vocab["<BOS>"])
    eos_id = int(vocab["<EOS>"])
    was_training = bool(model.training)
    model.eval()
    rows: List[dict] = []
    try:
        with torch.no_grad():
            for sample_index in range(int(num_samples)):
                if constraint_config is not None:
                    sample = sample_topology_constrained(
                        model,
                        vocab,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=top_k,
                        constraint_config=constraint_config,
                        device=device,
                    )
                    row = {
                        "format": "maskgen_manual_topology_ar_sample_v1",
                        "sample_index": int(sample_index),
                        "sampling_mode": inverse_mode,
                        "length": int(sample["length"]),
                        "hit_eos": bool(sample["hit_eos"]),
                        "ids": [int(value) for value in sample["ids"]],
                        "tokens": list(sample["tokens"]),
                        "constraint_diagnostics": sample["constraint_diagnostics"],
                    }
                else:
                    start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                    generated = model.generate(
                        start,
                        max_new_tokens=int(max_new_tokens),
                        eos_id=eos_id,
                        temperature=float(temperature),
                        top_k=top_k,
                    )[0].detach().cpu().tolist()
                    row = {
                        "format": "maskgen_manual_topology_ar_sample_v1",
                        "sample_index": int(sample_index),
                        "sampling_mode": inverse_mode,
                        "length": int(len(generated)),
                        "hit_eos": bool(eos_id in generated[1:]),
                        "ids": [int(value) for value in generated],
                        "tokens": ids_to_tokens(generated, vocab),
                    }
                rows.append(row)
                if int(progress_every) > 0 and (
                    (sample_index + 1) % int(progress_every) == 0 or sample_index + 1 == int(num_samples)
                ):
                    print(f"{progress_label} {sample_index + 1}/{int(num_samples)}", flush=True)
    finally:
        if was_training:
            model.train()
    return rows


def evaluate_model_topology_samples(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    constraint_config=None,
    top_k_invalid: int = 20,
    progress_every: int = 0,
    progress_label: str = "topology_eval_sample",
) -> Dict[str, object]:
    rows = sample_model_topology_rows(
        model,
        vocab,
        num_samples=int(num_samples),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=top_k,
        device=device,
        constraint_config=constraint_config,
        progress_every=int(progress_every),
        progress_label=progress_label,
    )
    summary = evaluate_topology_sample_rows(rows, top_k_invalid=int(top_k_invalid))
    summary["sampling_mode"] = "constrained" if constraint_config is not None else "unconstrained"
    summary["sampling_config"] = {
        "num_samples": int(num_samples),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": None if top_k is None else int(top_k),
        "constraint_config": None if constraint_config is None else constraint_config.__dict__,
    }
    return summary


def write_topology_sample_evaluation_markdown(summary: dict, output_path: Path) -> None:
    lengths = summary.get("lengths", {})
    valid_lengths = summary.get("valid_lengths", {})
    node_counts = summary.get("node_counts", {})
    relation_mean = summary.get("relation_mean_per_valid_sample", {})
    lines = [
        "# Manual Topology Sample Evaluation",
        "",
        f"- samples: {summary['sample_count']}",
        f"- valid: {summary['valid_count']}",
        f"- valid_rate: {summary['valid_rate']:.4f}",
        f"- hit_eos: {summary['hit_eos_count']}",
        f"- length mean / p95 / max: {lengths.get('mean')} / {lengths.get('p95')} / {lengths.get('max')}",
        f"- valid length mean / p95 / max: {valid_lengths.get('mean')} / {valid_lengths.get('p95')} / {valid_lengths.get('max')}",
        f"- node count mean / p95 / max: {node_counts.get('mean')} / {node_counts.get('p95')} / {node_counts.get('max')}",
        "",
        "## Relation Mean Per Valid Sample",
        "",
        "| relation | mean |",
        "| --- | ---: |",
    ]
    for relation, value in sorted(relation_mean.items()):
        lines.append(f"| `{relation}` | {float(value):.4f} |")
    lines.extend(["", "## Role Histogram", "", "| role | count |", "| --- | ---: |"])
    for role, count in sorted((summary.get("role_histogram") or {}).items()):
        lines.append(f"| `{role}` | {count} |")
    lines.extend(["", "## Label Histogram", "", "| label | count |", "| --- | ---: |"])
    for label, count in sorted((summary.get("label_histogram") or {}).items(), key=lambda item: int(item[0])):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## Failure Reasons", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in (summary.get("failure_reason_histogram") or {}).items():
        lines.append(f"| `{reason}` | {count} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_topology_sample_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
