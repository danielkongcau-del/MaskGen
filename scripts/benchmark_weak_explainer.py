from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Benchmark weak explanation and weak parse_graph rendering.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--partition-root", type=Path, default=None)
    source.add_argument("--global-root", type=Path, default=None)
    source.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/weak_explainer_benchmark_val.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--convex-backend", type=str, default="fallback_cdt_greedy", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=128)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    parser.add_argument("--no-label-groups", action="store_true")
    parser.add_argument("--no-atom-nodes", action="store_true")
    parser.add_argument("--no-boundary-arcs", action="store_true")
    return parser.parse_args()


def runtime_ms_since(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


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


def label_histogram(evidence_payload: dict) -> dict[str, int]:
    output: dict[str, int] = {}
    for face in evidence_payload.get("faces", []):
        key = str(int(face.get("label", -1)))
        output[key] = int(output.get(key, 0)) + 1
    return output


def make_success_row(
    path: Path,
    *,
    source_kind: SourceKind,
    split: str,
    runtime_ms: float,
    evidence_payload: dict,
    weak_payload: dict,
    render_payload: dict,
    global_payload: dict | None,
) -> dict:
    evidence_validation = evidence_payload.get("evidence_validation", {})
    weak_diag = weak_payload.get("diagnostics", {})
    render_validation = render_payload.get("validation", {})
    face_count = int(weak_diag.get("semantic_face_count", 0))
    atom_count = int(weak_diag.get("convex_atom_count", 0))
    relation_count = int(weak_diag.get("relation_count", 0))
    residual_count = int(weak_diag.get("residual_face_count", 0))
    return {
        "source_file": str(path.as_posix()),
        "source_kind": source_kind,
        "split": split,
        "stem": path.stem,
        "success": True,
        "failure_reason": None,
        "runtime_ms": runtime_ms,
        "global_valid": None if global_payload is None else bool(global_payload.get("validation", {}).get("is_valid", False)),
        "evidence_valid": bool(evidence_validation.get("is_valid", evidence_validation.get("usable_for_explainer", False))),
        "weak_valid": bool(weak_payload.get("validation", {}).get("is_valid", False)),
        "render_valid": bool(render_validation.get("is_valid", False)),
        "face_count": face_count,
        "atom_count": atom_count,
        "atom_per_face": float(atom_count / face_count) if face_count > 0 else None,
        "label_group_count": int(weak_diag.get("label_group_count", 0)),
        "relation_count": relation_count,
        "relation_per_face": float(relation_count / face_count) if face_count > 0 else None,
        "residual_face_count": residual_count,
        "code_length": float(weak_diag.get("total_code_length", 0.0)),
        "full_iou": render_validation.get("full_iou"),
        "mask_pixel_accuracy": render_validation.get("mask_pixel_accuracy"),
        "overlap_area": float(render_validation.get("overlap_area", 0.0)),
        "gap_area": render_validation.get("gap_area"),
        "extra_area": render_validation.get("extra_area"),
        "low_iou_face_count": int(len(render_validation.get("low_iou_face_ids", []) or [])),
        "invalid_face_count": int(len(render_validation.get("invalid_face_ids", []) or [])),
        "label_histogram": label_histogram(evidence_payload),
    }


def make_failure_row(path: Path, *, source_kind: SourceKind, split: str, runtime_ms: float, error: Exception) -> dict:
    return {
        "source_file": str(path.as_posix()),
        "source_kind": source_kind,
        "split": split,
        "stem": path.stem,
        "success": False,
        "failure_reason": f"{type(error).__name__}: {error}",
        "runtime_ms": runtime_ms,
    }


def main() -> None:
    args = parse_args()
    source_kind, root = infer_source(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    global_config = GlobalApproxConfig(
        face_simplify_tolerance=float(args.face_simplify_tolerance),
        face_area_epsilon=float(args.face_area_epsilon),
        validity_eps=float(args.validity_eps),
    )
    evidence_config = ExplanationEvidenceConfig(
        convex_backend=args.convex_backend,
        convex_cgal_cli=args.convex_cgal_cli,
        convex_max_bridge_sets=args.convex_max_bridge_sets,
        convex_cut_slit_scale=args.convex_cut_slit_scale,
    )
    weak_config = WeakExplainerConfig(
        include_label_groups=not args.no_label_groups,
        include_convex_atom_nodes=not args.no_atom_nodes,
        use_boundary_arcs_when_available=not args.no_boundary_arcs,
    )
    render_config = WeakRenderConfig(validity_eps=float(args.validity_eps))

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in iter_json_paths(root, args.split):
            if args.max_samples is not None and written >= int(args.max_samples):
                break
            start = time.perf_counter()
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
                row = make_success_row(
                    path,
                    source_kind=source_kind,
                    split=args.split,
                    runtime_ms=runtime_ms_since(start),
                    evidence_payload=evidence_payload,
                    weak_payload=weak_payload,
                    render_payload=render_payload,
                    global_payload=global_payload,
                )
            except Exception as error:  # noqa: BLE001 - benchmark rows must capture failures.
                row = make_failure_row(path, source_kind=source_kind, split=args.split, runtime_ms=runtime_ms_since(start), error=error)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            print(
                f"{path.name}: success={row.get('success')} render_valid={row.get('render_valid')} "
                f"faces={row.get('face_count')} atoms={row.get('atom_count')} "
                f"iou={row.get('full_iou')} gap={row.get('gap_area')} overlap={row.get('overlap_area')} "
                f"failure={row.get('failure_reason')}"
            )
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
