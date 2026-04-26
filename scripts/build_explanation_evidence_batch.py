from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.explanation_evidence import ExplanationEvidenceConfig, build_explanation_evidence_payload  # noqa: E402
from partition_gen.global_approx_partition import GlobalApproxConfig, build_global_approx_partition_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explanation evidence in batch from partition graphs.")
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    parser.add_argument("--convex-backend", type=str, default="fallback_cdt_greedy", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=128)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--thin-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--compactness-threshold", type=float, default=0.45)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_partition_paths(root: Path, split: str) -> Iterable[Path]:
    graph_root = root / split / "graphs"
    yield from sorted(graph_root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))


def _runtime_ms(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def build_evidence_for_partition(
    path: Path,
    *,
    global_config: GlobalApproxConfig,
    evidence_config: ExplanationEvidenceConfig,
) -> dict:
    graph_payload = load_json(path)
    global_payload = build_global_approx_partition_payload(graph_payload, config=global_config, source_tag=str(path.as_posix()))
    return build_explanation_evidence_payload(
        global_payload,
        config=evidence_config,
        source_tag=str(path.as_posix()),
    )


def _manifest_row(
    *,
    path: Path,
    output_path: Path | None,
    split: str,
    payload: dict | None,
    runtime_ms: float,
    error: Exception | None = None,
) -> dict:
    validation = payload.get("evidence_validation", {}) if payload else {}
    stats = payload.get("statistics", {}) if payload else {}
    return {
        "source_file": str(path.as_posix()),
        "split": split,
        "stem": path.stem,
        "success": bool(error is None),
        "output_path": None if output_path is None else str(output_path.as_posix()),
        "evidence_valid": validation.get("is_valid"),
        "usable_for_explainer": validation.get("usable_for_explainer"),
        "face_count": validation.get("face_count"),
        "arc_count": validation.get("arc_count"),
        "adjacency_count": validation.get("adjacency_count"),
        "convex_success_count": validation.get("convex_success_count"),
        "convex_failure_count": validation.get("convex_failure_count"),
        "total_convex_atom_count": stats.get("total_convex_atom_count"),
        "runtime_ms": float(runtime_ms),
        "failure_reason": None if error is None else f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    output_split_root = args.output_root / args.split
    graph_root = output_split_root / "graphs"
    graph_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_split_root / "manifest.jsonl"
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
        thin_aspect_ratio=float(args.thin_aspect_ratio),
        compactness_threshold=float(args.compactness_threshold),
    )

    rows = []
    for index, path in enumerate(iter_partition_paths(args.partition_root, args.split)):
        if args.max_samples is not None and index >= int(args.max_samples):
            break
        started = time.perf_counter()
        output_path = graph_root / f"{path.stem}.json"
        try:
            payload = build_evidence_for_partition(path, global_config=global_config, evidence_config=evidence_config)
            dump_json(output_path, payload)
            row = _manifest_row(path=path, output_path=output_path, split=args.split, payload=payload, runtime_ms=_runtime_ms(started))
        except Exception as error:  # noqa: BLE001 - batch builder records failures.
            row = _manifest_row(path=path, output_path=None, split=args.split, payload=None, runtime_ms=_runtime_ms(started), error=error)
        rows.append(row)
        print(
            f"{path.name}: success={row['success']} usable={row.get('usable_for_explainer')} "
            f"faces={row.get('face_count')} atoms={row.get('total_convex_atom_count')} failure={row['failure_reason']}"
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "format": "maskgen_explanation_evidence_batch_summary_v1",
        "partition_root": str(args.partition_root.as_posix()),
        "split": args.split,
        "output_root": str(args.output_root.as_posix()),
        "sample_count": int(len(rows)),
        "success_count": int(sum(1 for row in rows if row["success"])),
        "usable_count": int(sum(1 for row in rows if row.get("usable_for_explainer"))),
        "failure_count": int(sum(1 for row in rows if not row["success"])),
    }
    dump_json(output_split_root / "summary.json", summary)
    print(f"built evidence split={args.split}: samples={len(rows)} success={summary['success_count']} manifest={manifest_path}")


if __name__ == "__main__":
    main()
