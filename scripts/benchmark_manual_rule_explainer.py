from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_rule_explainer import (  # noqa: E402
    ManualRuleExplainerConfig,
    build_manual_rule_explanation_payload,
)
from partition_gen.operation_role_spec import load_role_spec, pair_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark manual role-spec parse graph explanations.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--role-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-all-faces-of-support-labels", action="store_true")
    parser.add_argument("--include-all-faces-of-divider-labels", action="store_true")
    parser.add_argument("--include-soft-rules", action="store_true")
    parser.add_argument("--disable-support-component-split", action="store_true")
    parser.add_argument("--disable-divider-component-split", action="store_true")
    parser.add_argument("--min-shared-length", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_evidence_paths(root: Path, split: str | None = None) -> Iterable[Path]:
    search_root = root / split if split and (root / split).exists() else root
    for path in sorted(search_root.rglob("*.json")):
        yield path


def _explicit_pair_keys(role_spec_payload: Dict[str, object]) -> set[str]:
    return {
        pair_key(int(rule["subject_label"]), int(rule["object_label"]))
        for rule in role_spec_payload.get("relations", [])
    }


def _label_name(role_spec_payload: Dict[str, object], label: int) -> str:
    names = role_spec_payload.get("label_names", {})
    return str(names.get(str(int(label)), names.get(int(label), str(label))))


def _face_labels(evidence_payload: Dict[str, object]) -> Dict[int, int]:
    return {int(face["id"]): int(face.get("label", -1)) for face in evidence_payload.get("faces", [])}


def _contact_pair_stats(evidence_payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    labels_by_face = _face_labels(evidence_payload)
    stats: Dict[str, Dict[str, object]] = {}
    for item in evidence_payload.get("adjacency", []):
        face_ids = [int(value) for value in item.get("faces", [])]
        if len(face_ids) != 2:
            continue
        if "labels" in item and len(item.get("labels", [])) == 2:
            labels = [int(item["labels"][0]), int(item["labels"][1])]
        else:
            if face_ids[0] not in labels_by_face or face_ids[1] not in labels_by_face:
                continue
            labels = [labels_by_face[face_ids[0]], labels_by_face[face_ids[1]]]
        if labels[0] == labels[1]:
            continue
        key = pair_key(labels[0], labels[1])
        bucket = stats.setdefault(
            key,
            {
                "pair": key,
                "labels": sorted([int(labels[0]), int(labels[1])]),
                "contact_count": 0,
                "shared_length": 0.0,
                "arc_count": 0,
                "face_pairs": [],
            },
        )
        bucket["contact_count"] = int(bucket["contact_count"]) + 1
        bucket["shared_length"] = float(bucket["shared_length"]) + float(item.get("shared_length", 0.0) or 0.0)
        bucket["arc_count"] = int(bucket["arc_count"]) + int(item.get("arc_count", len(item.get("arc_ids", []))) or 0)
        bucket["face_pairs"].append(face_ids)
    return stats


def _role_spec_pair_stats(
    evidence_payload: Dict[str, object],
    role_spec_payload: Dict[str, object],
) -> Dict[str, object]:
    explicit_keys = _explicit_pair_keys(role_spec_payload)
    contact_stats = _contact_pair_stats(evidence_payload)
    covered = {}
    uncovered = {}
    for key, value in contact_stats.items():
        target = covered if key in explicit_keys else uncovered
        labels = [int(v) for v in value["labels"]]
        target[key] = {
            "pair": key,
            "labels": labels,
            "label_names": [_label_name(role_spec_payload, labels[0]), _label_name(role_spec_payload, labels[1])],
            "contact_count": int(value["contact_count"]),
            "shared_length": float(value["shared_length"]),
            "arc_count": int(value["arc_count"]),
        }
    top_uncovered = sorted(
        uncovered.values(),
        key=lambda item: (float(item["shared_length"]), int(item["contact_count"])),
        reverse=True,
    )
    return {
        "label_pair_contact_count": int(len(contact_stats)),
        "explicit_contact_pair_count": int(len(covered)),
        "uncovered_contact_pair_count": int(len(uncovered)),
        "uncovered_contact_pairs": top_uncovered[:20],
    }


def _row_from_payload(
    source: Path,
    evidence_payload: Dict[str, object],
    explanation: Dict[str, object],
    role_spec_payload: Dict[str, object],
    runtime_ms: float,
) -> Dict[str, object]:
    diagnostics = explanation.get("diagnostics", {})
    validation = explanation.get("validation", {})
    pair_stats = _role_spec_pair_stats(evidence_payload, role_spec_payload)
    return {
        "source": str(source.as_posix()),
        "profile": diagnostics.get("profile"),
        "role_spec_name": diagnostics.get("role_spec_name"),
        "role_spec_relation_count": diagnostics.get("role_spec_relation_count"),
        "active_role_spec_relation_count": diagnostics.get("active_role_spec_relation_count"),
        "soft_role_spec_relation_count": diagnostics.get("soft_role_spec_relation_count"),
        "include_soft_rules": diagnostics.get("include_soft_rules"),
        "split_support_by_connected_components": diagnostics.get("split_support_by_connected_components"),
        "split_divider_by_connected_components": diagnostics.get("split_divider_by_connected_components"),
        "support_component_count": diagnostics.get("support_component_count"),
        "divider_component_count": diagnostics.get("divider_component_count"),
        "support_node_count": diagnostics.get("support_node_count"),
        "divider_node_count": diagnostics.get("divider_node_count"),
        "insert_group_count": diagnostics.get("insert_group_count"),
        "reference_support_node_count": diagnostics.get("reference_support_node_count"),
        "face_count": diagnostics.get("face_count"),
        "node_count": diagnostics.get("node_count"),
        "relation_count": diagnostics.get("relation_count"),
        "residual_face_count": diagnostics.get("residual_face_count"),
        "residual_area_ratio": diagnostics.get("residual_area_ratio"),
        "owned_face_count": diagnostics.get("owned_face_count"),
        "referenced_face_count": diagnostics.get("referenced_face_count"),
        "duplicate_owned_face_count": diagnostics.get("duplicate_owned_face_count"),
        "unowned_face_count": diagnostics.get("unowned_face_count"),
        "selection_method": diagnostics.get("selection_method"),
        "uses_ortools": diagnostics.get("uses_ortools"),
        "uses_candidate_search": diagnostics.get("uses_candidate_search"),
        "operation_histogram": diagnostics.get("operation_histogram", {}),
        "role_histogram": diagnostics.get("role_histogram", {}),
        "validation_is_valid": validation.get("is_valid"),
        "input_evidence_valid": validation.get("input_evidence_valid"),
        "all_faces_owned_exactly_once": validation.get("all_faces_owned_exactly_once"),
        "node_reference_valid": validation.get("node_reference_valid"),
        "relation_reference_valid": validation.get("relation_reference_valid"),
        "runtime_ms": float(runtime_ms),
        **pair_stats,
    }


def benchmark_one(
    path: Path,
    role_spec_payload: Dict[str, object],
    config: ManualRuleExplainerConfig,
) -> Dict[str, object] | None:
    evidence = load_json(path)
    if evidence.get("format") != "maskgen_explanation_evidence_v1":
        return None
    started = time.perf_counter()
    try:
        explanation = build_manual_rule_explanation_payload(
            evidence,
            role_spec_payload,
            config=config,
            source_tag=str(path.as_posix()),
        )
        runtime_ms = (time.perf_counter() - started) * 1000.0
        return _row_from_payload(path, evidence, explanation, role_spec_payload, runtime_ms)
    except Exception as exc:
        return {
            "source": str(path.as_posix()),
            "error": str(exc),
            "validation_is_valid": False,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        }


def run_benchmark(
    evidence_root: Path,
    role_spec_payload: Dict[str, object],
    config: ManualRuleExplainerConfig,
    *,
    split: str | None = None,
    max_samples: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_count = 0
    for path in iter_evidence_paths(evidence_root, split=split):
        row = benchmark_one(path, role_spec_payload, config)
        if row is None:
            continue
        rows.append(row)
        sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break
    return rows


def main() -> None:
    args = parse_args()
    role_spec = load_role_spec(args.role_spec)
    config = ManualRuleExplainerConfig(
        include_all_faces_of_support_labels=bool(args.include_all_faces_of_support_labels),
        include_all_faces_of_divider_labels=bool(args.include_all_faces_of_divider_labels),
        include_soft_rules=bool(args.include_soft_rules),
        split_support_by_connected_components=not bool(args.disable_support_component_split),
        split_divider_by_connected_components=not bool(args.disable_divider_component_split),
        min_shared_length=float(args.min_shared_length),
    )
    rows = run_benchmark(
        args.evidence_root,
        role_spec,
        config,
        split=args.split,
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid_count = sum(1 for row in rows if row.get("validation_is_valid") is True)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"valid={valid_count}/{len(rows)}")


if __name__ == "__main__":
    main()
