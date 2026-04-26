from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_rule_explainer import (  # noqa: E402
    ManualRuleExplainerConfig,
    build_manual_rule_explanation_payload,
)
from partition_gen.operation_role_spec import load_role_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual-rule generator target JSON files in batch.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--role-spec", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_generator_targets_manual_rule"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-evidence-fields", action="store_true")
    parser.add_argument("--only-training-usable", action="store_true")
    parser.add_argument("--include-all-faces-of-support-labels", action="store_true")
    parser.add_argument("--include-all-faces-of-divider-labels", action="store_true")
    parser.add_argument("--min-shared-length", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_evidence_paths(root: Path, split: str | None = None) -> Iterable[Path]:
    graph_dir = root / split / "graphs" if split else None
    if graph_dir and graph_dir.exists():
        yield from sorted(graph_dir.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    split_dir = root / split if split else None
    if split_dir and split_dir.exists():
        yield from sorted(split_dir.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    yield from sorted(root.rglob("*.json"), key=lambda path: (str(path.parent), len(path.stem), path.stem))


def _strip_node_for_training(node: dict) -> dict:
    cleaned = copy.deepcopy(node)
    cleaned.pop("evidence", None)
    cleaned.pop("source_face_id", None)
    cleaned.pop("source_atom_id", None)
    cleaned.pop("features", None)
    return cleaned


def _strip_relation_for_training(relation: dict) -> dict:
    cleaned = copy.deepcopy(relation)
    for key in ("face_ids", "source_face_ids", "induced_face_ids", "divider_face_ids", "arc_ids", "source_arc_ids", "rule_ids"):
        cleaned.pop(key, None)
    return cleaned


def sanitize_generator_target(target: dict) -> dict:
    target = copy.deepcopy(target)
    graph = target.setdefault("parse_graph", {})
    graph["nodes"] = [_strip_node_for_training(node) for node in graph.get("nodes", [])]
    graph["relations"] = [_strip_relation_for_training(relation) for relation in graph.get("relations", [])]
    for residual in graph.get("residuals", []):
        residual.pop("face_ids", None)
    metadata = target.setdefault("metadata", {})
    metadata["training_sanitized"] = True
    metadata["removed_fields"] = [
        "node.evidence",
        "node.source_face_id",
        "node.source_atom_id",
        "node.features",
        "relation.face_ids",
        "relation.source_face_ids",
        "relation.induced_face_ids",
        "relation.divider_face_ids",
        "relation.arc_ids",
        "relation.source_arc_ids",
        "relation.rule_ids",
        "residual.face_ids",
    ]
    return target


def _target_counts(target: dict) -> dict:
    graph = target.get("parse_graph", {})
    nodes = graph.get("nodes", [])
    relations = graph.get("relations", [])
    return {
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "residual_count": int(len(graph.get("residuals", []))),
        "support_count": int(sum(1 for node in nodes if node.get("role") == "support_region")),
        "divider_count": int(sum(1 for node in nodes if node.get("role") == "divider_region")),
        "insert_count": int(sum(1 for node in nodes if node.get("role") == "insert_object")),
        "insert_group_count": int(sum(1 for node in nodes if node.get("role") == "insert_object_group")),
        "reference_only_count": int(sum(1 for node in nodes if bool(node.get("is_reference_only", False)))),
    }


def _manifest_row(
    *,
    source_path: Path,
    split: str,
    output_path: Path | None,
    runtime_ms: float,
    success: bool,
    explanation: dict | None = None,
    target: dict | None = None,
    error: Exception | None = None,
) -> dict:
    diagnostics = explanation.get("diagnostics", {}) if explanation else {}
    validation = explanation.get("validation", {}) if explanation else {}
    counts = _target_counts(target) if target else {}
    training_usable = bool(success and validation.get("is_valid") and int(counts.get("residual_count", 0)) == 0)
    return {
        "source_file": str(source_path.as_posix()),
        "source_kind": "evidence",
        "split": split,
        "stem": source_path.stem,
        "success": bool(success),
        "target_written": bool(output_path is not None),
        "target_path": None if output_path is None else str(output_path.as_posix()),
        "training_usable": bool(training_usable),
        "manual_rule_valid": validation.get("is_valid"),
        "all_faces_owned_exactly_once": validation.get("all_faces_owned_exactly_once"),
        "face_count": diagnostics.get("face_count"),
        "residual_face_count": diagnostics.get("residual_face_count"),
        "residual_area_ratio": diagnostics.get("residual_area_ratio"),
        "duplicate_owned_face_count": diagnostics.get("duplicate_owned_face_count"),
        "unowned_face_count": diagnostics.get("unowned_face_count"),
        "operation_histogram": diagnostics.get("operation_histogram", {}),
        "role_histogram": diagnostics.get("role_histogram", {}),
        **counts,
        "runtime_ms": float(runtime_ms),
        "failure_reason": None if error is None else f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    role_spec = load_role_spec(args.role_spec)
    config = ManualRuleExplainerConfig(
        include_all_faces_of_support_labels=bool(args.include_all_faces_of_support_labels),
        include_all_faces_of_divider_labels=bool(args.include_all_faces_of_divider_labels),
        min_shared_length=float(args.min_shared_length),
    )
    split_root = args.output_root / args.split
    graph_root = split_root / "graphs"
    manifest_path = split_root / "manifest.jsonl"
    graph_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    usable = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for path in iter_evidence_paths(args.evidence_root, split=args.split):
            if args.max_samples is not None and processed >= int(args.max_samples):
                break
            started = time.perf_counter()
            output_path = None
            try:
                evidence = load_json(path)
                if evidence.get("format") != "maskgen_explanation_evidence_v1":
                    continue
                explanation = build_manual_rule_explanation_payload(
                    evidence,
                    role_spec,
                    config=config,
                    source_tag=str(path.as_posix()),
                )
                target = copy.deepcopy(explanation["generator_target"])
                metadata = target.setdefault("metadata", {})
                metadata.update(
                    {
                        "source_file": str(path.as_posix()),
                        "source_kind": "evidence",
                        "split": args.split,
                        "stem": path.stem,
                        "target_profile": "manual_role_spec_parse_graph_v1",
                        "manual_rule_valid": bool(explanation["validation"]["is_valid"]),
                        "training_usable": bool(
                            explanation["validation"]["is_valid"] and int(explanation["diagnostics"]["residual_face_count"]) == 0
                        ),
                        **_target_counts(target),
                    }
                )
                if not args.keep_evidence_fields:
                    target = sanitize_generator_target(target)
                    target["metadata"].update(metadata)
                training_usable = bool(target["metadata"].get("training_usable", False))
                if training_usable:
                    usable += 1
                if not args.only_training_usable or training_usable:
                    output_path = graph_root / f"{path.stem}.json"
                    dump_json(output_path, target)
                    written += 1
                row = _manifest_row(
                    source_path=path,
                    split=args.split,
                    output_path=output_path,
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                    success=True,
                    explanation=explanation,
                    target=target,
                )
            except Exception as error:  # noqa: BLE001 - batch builder records failures in manifest.
                row = _manifest_row(
                    source_path=path,
                    split=args.split,
                    output_path=None,
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                    success=False,
                    error=error,
                )
            processed += 1
            manifest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(
                f"{path.name}: success={row['success']} written={row['target_written']} "
                f"usable={row['training_usable']} nodes={row.get('node_count')} "
                f"relations={row.get('relation_count')} residual={row.get('residual_count')} "
                f"failure={row['failure_reason']}"
            )

    print(f"processed={processed} written={written} training_usable={usable} manifest={manifest_path} output_graphs={graph_root}")


if __name__ == "__main__":
    main()
