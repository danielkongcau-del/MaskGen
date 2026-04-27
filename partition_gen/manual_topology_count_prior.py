from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from partition_gen.manual_topology_evaluation import parse_topology_structure
from partition_gen.manual_topology_sample_validation import validate_topology_tokens


COUNT_PRIOR_KEYS = ("node_count", "child_count", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO")


def _counter_to_dict(counter: Counter[int]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _normalize_histogram(histogram: Mapping[object, object]) -> Dict[int, float]:
    counts = {int(key): float(value) for key, value in histogram.items()}
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {}
    return {int(key): float(value / total) for key, value in sorted(counts.items())}


def topology_count_prior_from_rows(rows: Sequence[dict], *, source: str | None = None) -> Dict[str, object]:
    histograms = {key: Counter() for key in COUNT_PRIOR_KEYS}
    valid_count = 0
    semantic_valid_count = 0

    for row in rows:
        tokens = [str(token) for token in row.get("tokens", [])]
        validation = validate_topology_tokens(tokens)
        if not validation["valid"]:
            continue
        valid_count += 1
        if not validation.get("semantic_valid", False):
            continue
        semantic_valid_count += 1
        parsed = parse_topology_structure(tokens)
        histograms["node_count"][int(parsed["node_count"])] += 1
        for child_count in parsed["insert_group_child_counts"]:
            histograms["child_count"][int(child_count)] += 1
        relation_counts = parsed["relation_counts"]
        histograms["REL_BLOCK_DIVIDES"][int(relation_counts.get("REL_BLOCK_DIVIDES", 0))] += 1
        histograms["REL_BLOCK_ADJACENT_TO"][int(relation_counts.get("REL_BLOCK_ADJACENT_TO", 0))] += 1

    return {
        "format": "maskgen_manual_topology_count_prior_v1",
        "source": source,
        "sample_count": int(len(rows)),
        "valid_count": int(valid_count),
        "semantic_valid_count": int(semantic_valid_count),
        "histograms": {key: _counter_to_dict(counter) for key, counter in histograms.items()},
        "priors": {key: _normalize_histogram(counter) for key, counter in histograms.items()},
    }


def topology_count_prior_from_token_root(token_root: Path) -> Dict[str, object]:
    token_root = Path(token_root)
    path = token_root / "topology_sequences.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return topology_count_prior_from_rows(rows, source=str(token_root.as_posix()))


def load_topology_count_prior(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    histograms = payload.get("histograms", {})
    if "priors" not in payload:
        payload["priors"] = {key: _normalize_histogram(histogram) for key, histogram in histograms.items()}
    else:
        payload["priors"] = {
            str(key): {int(inner_key): float(value) for inner_key, value in dict(inner).items()}
            for key, inner in dict(payload["priors"]).items()
        }
    return payload


def write_topology_count_prior(payload: Mapping[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
