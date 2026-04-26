from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

from partition_gen.operation_label_pair_prior import REL_DIVIDES, REL_INSERTED_IN, REL_PARALLEL


SUPPORTED_RELATIONS = {REL_DIVIDES, REL_INSERTED_IN, REL_PARALLEL}


def pair_key(left_label: int, right_label: int) -> str:
    a, b = sorted((int(left_label), int(right_label)))
    return f"{a}:{b}"


def load_role_spec(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_role_spec(payload)
    return payload


def validate_role_spec(payload: Dict[str, object]) -> None:
    if payload.get("format") != "maskgen_role_spec_v1":
        raise ValueError("role spec format must be maskgen_role_spec_v1")
    for index, rule in enumerate(payload.get("relations", [])):
        if "subject_label" not in rule or "object_label" not in rule:
            raise ValueError(f"role spec relation {index} must include subject_label and object_label")
        relation = str(rule.get("relation", ""))
        if relation not in SUPPORTED_RELATIONS:
            raise ValueError(f"unsupported role spec relation {relation!r}; expected one of {sorted(SUPPORTED_RELATIONS)}")
        if int(rule["subject_label"]) == int(rule["object_label"]):
            raise ValueError(f"role spec relation {index} uses the same subject and object label")


def _rule_selected(rule: Dict[str, object]) -> Dict[str, object]:
    subject_label = int(rule["subject_label"])
    object_label = int(rule["object_label"])
    relation_type = str(rule["relation"])
    if relation_type == REL_PARALLEL:
        relation = "PARALLEL"
        subject_label = min(subject_label, object_label)
        object_label = max(int(rule["subject_label"]), int(rule["object_label"]))
    else:
        relation = f"{subject_label}_{relation_type}_{object_label}"
    hard = bool(rule.get("hard", True))
    return {
        "relation_type": relation_type,
        "relation": relation,
        "subject_label": int(subject_label),
        "object_label": int(object_label),
        "score": float(rule.get("score", 1.0)),
        "confidence": float(rule.get("confidence", 1.0 if hard else 0.5)),
        "source": "explicit_role_spec",
        "hard": hard,
        "rule_id": rule.get("id"),
        "notes": rule.get("notes"),
    }


def _empty_pair(labels: Tuple[int, int]) -> Dict[str, object]:
    a_label, b_label = sorted((int(labels[0]), int(labels[1])))
    return {
        "key": pair_key(a_label, b_label),
        "labels": [a_label, b_label],
        "selected": {},
        "scores": {},
        "direction_stats": {},
    }


def apply_role_spec_to_label_pair_priors(
    label_pair_prior_payload: Dict[str, object],
    role_spec_payload: Dict[str, object] | None,
    *,
    require_explicit: bool = False,
) -> Dict[str, object]:
    if not role_spec_payload:
        return label_pair_prior_payload
    validate_role_spec(role_spec_payload)
    output = deepcopy(label_pair_prior_payload)
    pairs_by_key = {pair.get("key") or pair_key(pair["labels"][0], pair["labels"][1]): pair for pair in output.get("pairs", [])}
    explicit_keys = []
    for rule in role_spec_payload.get("relations", []):
        key = pair_key(int(rule["subject_label"]), int(rule["object_label"]))
        pair = pairs_by_key.setdefault(key, _empty_pair((int(rule["subject_label"]), int(rule["object_label"]))))
        pair["selected"] = _rule_selected(rule)
        pair["role_spec_rule"] = rule
        explicit_keys.append(key)
    if require_explicit:
        pairs_by_key = {key: pair for key, pair in pairs_by_key.items() if key in set(explicit_keys)}
    output["pairs"] = [pairs_by_key[key] for key in sorted(pairs_by_key)]
    histogram: Dict[str, int] = {}
    source_histogram: Dict[str, int] = {}
    for pair in output["pairs"]:
        selected = pair.get("selected", {})
        relation_type = str(selected.get("relation_type", "UNDECIDED"))
        source = str(selected.get("source", "auto_label_pair_prior"))
        histogram[relation_type] = int(histogram.get(relation_type, 0)) + 1
        source_histogram[source] = int(source_histogram.get(source, 0)) + 1
    output["relation_histogram"] = dict(sorted(histogram.items()))
    output["source_histogram"] = dict(sorted(source_histogram.items()))
    output["role_spec"] = {
        "format": role_spec_payload.get("format"),
        "name": role_spec_payload.get("name"),
        "relation_count": int(len(role_spec_payload.get("relations", []))),
        "explicit_pair_keys": sorted(set(explicit_keys)),
        "require_explicit": bool(require_explicit),
        "defaults": role_spec_payload.get("defaults", {}),
    }
    return output
