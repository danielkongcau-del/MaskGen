from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json  # noqa: E402
from partition_gen.explanation_evidence import ExplanationEvidenceConfig, build_explanation_evidence_payload  # noqa: E402
from partition_gen.global_approx_partition import GlobalApproxConfig, build_global_approx_partition_payload  # noqa: E402
from partition_gen.weak_explainer import WeakExplainerConfig, build_weak_explanation_payload  # noqa: E402
from partition_gen.weak_parse_graph_renderer import WeakRenderConfig, render_weak_explanation_payload  # noqa: E402


SourceKind = Literal["partition", "global", "evidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generator target parse_graph JSON files in batch.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--partition-root", type=Path, default=None)
    source.add_argument("--global-root", type=Path, default=None)
    source.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_generator_targets"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--convex-backend", type=str, default="cgal", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=256)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    parser.add_argument("--no-label-groups", action="store_true")
    parser.add_argument("--no-atom-nodes", action="store_true")
    parser.add_argument("--no-boundary-arcs", action="store_true")
    parser.add_argument("--keep-evidence-fields", action="store_true")
    parser.add_argument("--only-training-usable", action="store_true")
    return parser.parse_args()


def iter_json_paths(root: Path, split: str) -> Iterable[Path]:
    graph_dir = root / split / "graphs"
    if graph_dir.exists():
        yield from sorted(graph_dir.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    split_dir = root / split
    if split_dir.exists():
        yield from sorted(split_dir.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    yield from sorted(root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))


def infer_source(args: argparse.Namespace) -> tuple[SourceKind, Path]:
    if args.global_root is not None:
        return "global", args.global_root
    if args.evidence_root is not None:
        return "evidence", args.evidence_root
    return "partition", args.partition_root or Path("data/remote_256_partition")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _backend_counts(evidence_payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for face in evidence_payload.get("faces", []):
        backend = str(face.get("convex_partition", {}).get("backend", "unknown"))
        counts[backend] = int(counts.get(backend, 0)) + 1
    return dict(sorted(counts.items()))


def _target_counts(target: dict) -> dict[str, int]:
    graph = target.get("parse_graph", {})
    nodes = graph.get("nodes", [])
    relations = graph.get("relations", [])
    return {
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "residual_count": int(len(graph.get("residuals", []))),
        "label_group_count": int(sum(1 for node in nodes if node.get("role") == "label_group")),
        "semantic_face_count": int(sum(1 for node in nodes if node.get("role") == "semantic_face")),
        "convex_atom_count": int(sum(1 for node in nodes if node.get("role") == "convex_atom")),
    }


def _strip_node_for_training(node: dict) -> dict:
    cleaned = copy.deepcopy(node)
    role = str(cleaned.get("role"))
    cleaned.pop("evidence", None)
    cleaned.pop("source_face_id", None)
    cleaned.pop("source_atom_id", None)
    cleaned.pop("features", None)
    if role == "semantic_face":
        atom_ids = [str(value) for value in cleaned.get("atom_ids", [])]
        cleaned["geometry_model"] = "convex_atom_union"
        cleaned["geometry"] = {"atom_ids": atom_ids}
    return cleaned


def _strip_relation_for_training(relation: dict) -> dict:
    cleaned = copy.deepcopy(relation)
    cleaned.pop("source_face_ids", None)
    cleaned.pop("arc_ids", None)
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
        "relation.source_face_ids",
        "relation.arc_ids",
        "residual.face_ids",
    ]
    return target


def build_case(
    path: Path,
    *,
    source_kind: SourceKind,
    split: str,
    mask_root: Path,
    global_config: GlobalApproxConfig,
    evidence_config: ExplanationEvidenceConfig,
    weak_config: WeakExplainerConfig,
    render_config: WeakRenderConfig,
) -> tuple[dict, dict, dict, dict | None]:
    source_tag = str(path.as_posix())
    global_payload = None
    if source_kind == "partition":
        partition_payload = load_json(path)
        global_payload = build_global_approx_partition_payload(partition_payload, config=global_config, source_tag=source_tag)
        evidence_payload = build_explanation_evidence_payload(global_payload, config=evidence_config, source_tag=source_tag)
    elif source_kind == "global":
        global_payload = load_json(path)
        evidence_payload = build_explanation_evidence_payload(global_payload, config=evidence_config, source_tag=source_tag)
    elif source_kind == "evidence":
        evidence_payload = load_json(path)
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")

    weak_payload = build_weak_explanation_payload(evidence_payload, config=weak_config, source_tag=source_tag)
    render_payload = render_weak_explanation_payload(
        weak_payload,
        evidence_payload=evidence_payload,
        config=render_config,
        mask_root=mask_root,
    )
    return evidence_payload, weak_payload, render_payload, global_payload


def make_target_payload(
    *,
    source_path: Path,
    source_kind: SourceKind,
    split: str,
    evidence_payload: dict,
    weak_payload: dict,
    render_payload: dict,
    global_payload: dict | None,
    convex_backend: str,
    keep_evidence_fields: bool,
) -> dict:
    target = copy.deepcopy(weak_payload["generator_target"])
    if not keep_evidence_fields:
        target = sanitize_generator_target(target)

    render_validation = render_payload.get("validation", {})
    weak_diag = weak_payload.get("diagnostics", {})
    evidence_validation = evidence_payload.get("evidence_validation", {})
    counts = _target_counts(target)
    convex_failure_count = int(evidence_validation.get("convex_failure_count", 0))
    render_valid = bool(render_validation.get("is_valid", False))
    weak_valid = bool(weak_payload.get("validation", {}).get("is_valid", False))
    training_usable = bool(render_valid and weak_valid and convex_failure_count == 0 and counts["residual_count"] == 0)

    metadata = target.setdefault("metadata", {})
    metadata.update(
        {
            "source_file": str(source_path.as_posix()),
            "source_kind": source_kind,
            "source_partition_graph": evidence_payload.get("source_partition_graph"),
            "source_global_approx": evidence_payload.get("source_global_approx"),
            "source_mask": evidence_payload.get("source_mask"),
            "split": split,
            "stem": source_path.stem,
            "convex_backend_requested": convex_backend,
            "convex_backend_counts": _backend_counts(evidence_payload),
            "convex_failure_count": convex_failure_count,
            "global_valid": None if global_payload is None else bool(global_payload.get("validation", {}).get("is_valid", False)),
            "evidence_valid": bool(evidence_validation.get("is_valid", False)),
            "weak_valid": weak_valid,
            "render_valid": render_valid,
            "training_usable": training_usable,
            "render_iou": render_validation.get("full_iou"),
            "full_iou": render_validation.get("full_iou"),
            "mask_pixel_accuracy": render_validation.get("mask_pixel_accuracy"),
            "overlap_area": render_validation.get("overlap_area"),
            "gap_area": render_validation.get("gap_area"),
            "extra_area": render_validation.get("extra_area"),
            "low_iou_face_count": int(len(render_validation.get("low_iou_face_ids", []) or [])),
            "invalid_face_count": int(len(render_validation.get("invalid_face_ids", []) or [])),
            "code_length": float(weak_diag.get("total_code_length", metadata.get("code_length", 0.0))),
            **counts,
        }
    )
    return target


def make_manifest_row(
    *,
    source_path: Path,
    source_kind: SourceKind,
    split: str,
    output_path: Path | None,
    runtime_ms: float,
    success: bool,
    target: dict | None = None,
    error: Exception | None = None,
) -> dict:
    metadata = target.get("metadata", {}) if target else {}
    return {
        "source_file": str(source_path.as_posix()),
        "source_kind": source_kind,
        "split": split,
        "stem": source_path.stem,
        "success": bool(success),
        "target_written": bool(output_path is not None),
        "target_path": None if output_path is None else str(output_path.as_posix()),
        "training_usable": bool(metadata.get("training_usable", False)),
        "render_valid": metadata.get("render_valid"),
        "weak_valid": metadata.get("weak_valid"),
        "evidence_valid": metadata.get("evidence_valid"),
        "global_valid": metadata.get("global_valid"),
        "convex_backend_requested": metadata.get("convex_backend_requested"),
        "convex_backend_counts": metadata.get("convex_backend_counts"),
        "convex_failure_count": metadata.get("convex_failure_count"),
        "node_count": metadata.get("node_count"),
        "semantic_face_count": metadata.get("semantic_face_count"),
        "convex_atom_count": metadata.get("convex_atom_count"),
        "relation_count": metadata.get("relation_count"),
        "residual_count": metadata.get("residual_count"),
        "code_length": metadata.get("code_length"),
        "full_iou": metadata.get("full_iou"),
        "mask_pixel_accuracy": metadata.get("mask_pixel_accuracy"),
        "overlap_area": metadata.get("overlap_area"),
        "gap_area": metadata.get("gap_area"),
        "runtime_ms": float(runtime_ms),
        "failure_reason": None if error is None else f"{type(error).__name__}: {error}",
    }


def runtime_ms_since(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def main() -> None:
    args = parse_args()
    source_kind, root = infer_source(args)
    split_root = args.output_root / args.split
    graph_root = split_root / "graphs"
    manifest_path = split_root / "manifest.jsonl"
    graph_root.mkdir(parents=True, exist_ok=True)

    global_config = GlobalApproxConfig(
        face_simplify_tolerance=float(args.face_simplify_tolerance),
        face_area_epsilon=float(args.face_area_epsilon),
        validity_eps=float(args.validity_eps),
    )
    evidence_config = ExplanationEvidenceConfig(
        convex_backend=args.convex_backend,
        convex_cgal_cli=args.convex_cgal_cli,
        convex_max_bridge_sets=int(args.convex_max_bridge_sets),
        convex_cut_slit_scale=float(args.convex_cut_slit_scale),
    )
    weak_config = WeakExplainerConfig(
        include_label_groups=not args.no_label_groups,
        include_convex_atom_nodes=not args.no_atom_nodes,
        use_boundary_arcs_when_available=not args.no_boundary_arcs,
    )
    render_config = WeakRenderConfig(validity_eps=float(args.validity_eps))

    written = 0
    processed = 0
    usable = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for path in iter_json_paths(root, args.split):
            if args.max_samples is not None and processed >= int(args.max_samples):
                break
            start = time.perf_counter()
            output_path: Path | None = None
            try:
                evidence_payload, weak_payload, render_payload, global_payload = build_case(
                    path,
                    source_kind=source_kind,
                    split=args.split,
                    mask_root=args.mask_root,
                    global_config=global_config,
                    evidence_config=evidence_config,
                    weak_config=weak_config,
                    render_config=render_config,
                )
                target = make_target_payload(
                    source_path=path,
                    source_kind=source_kind,
                    split=args.split,
                    evidence_payload=evidence_payload,
                    weak_payload=weak_payload,
                    render_payload=render_payload,
                    global_payload=global_payload,
                    convex_backend=args.convex_backend,
                    keep_evidence_fields=bool(args.keep_evidence_fields),
                )
                if bool(target["metadata"]["training_usable"]):
                    usable += 1
                if not args.only_training_usable or bool(target["metadata"]["training_usable"]):
                    output_path = graph_root / f"{path.stem}.json"
                    dump_json(output_path, target)
                    written += 1
                row = make_manifest_row(
                    source_path=path,
                    source_kind=source_kind,
                    split=args.split,
                    output_path=output_path,
                    runtime_ms=runtime_ms_since(start),
                    success=True,
                    target=target,
                )
            except Exception as error:  # noqa: BLE001 - batch builder records failures in manifest.
                row = make_manifest_row(
                    source_path=path,
                    source_kind=source_kind,
                    split=args.split,
                    output_path=None,
                    runtime_ms=runtime_ms_since(start),
                    success=False,
                    error=error,
                )
            processed += 1
            manifest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(
                f"{path.name}: success={row['success']} written={row['target_written']} "
                f"usable={row['training_usable']} render_valid={row['render_valid']} "
                f"atoms={row['convex_atom_count']} backend={row['convex_backend_counts']} "
                f"failure={row['failure_reason']}"
            )

    print(
        f"processed={processed} written={written} training_usable={usable} "
        f"manifest={manifest_path} output_graphs={graph_root}"
    )


if __name__ == "__main__":
    main()
