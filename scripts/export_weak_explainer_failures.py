from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_weak_explainer import (  # noqa: E402
    build_case,
)
from partition_gen.explanation_evidence import ExplanationEvidenceConfig  # noqa: E402
from partition_gen.global_approx_partition import GlobalApproxConfig  # noqa: E402
from partition_gen.weak_explainer import WeakExplainerConfig  # noqa: E402
from partition_gen.weak_parse_graph_renderer import WeakRenderConfig, save_render_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export weak explainer benchmark failures for inspection.")
    parser.add_argument("--benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/weak_explainer_failures"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--max-exports", type=int, default=20)
    parser.add_argument("--full-iou-threshold", type=float, default=0.999)
    parser.add_argument("--area-eps", type=float, default=1e-6)
    parser.add_argument("--convex-backend", type=str, default="fallback_cdt_greedy", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=128)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def should_export(row: dict, *, full_iou_threshold: float, area_eps: float) -> bool:
    if not row.get("success"):
        return True
    if not row.get("render_valid"):
        return True
    full_iou = row.get("full_iou")
    if full_iou is not None and float(full_iou) < full_iou_threshold:
        return True
    if float(row.get("overlap_area") or 0.0) > area_eps:
        return True
    gap_area = row.get("gap_area")
    if gap_area is not None and float(gap_area) > area_eps:
        return True
    if int(row.get("low_iou_face_count") or 0) > 0:
        return True
    return False


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in load_jsonl(args.benchmark_jsonl)
        if should_export(row, full_iou_threshold=float(args.full_iou_threshold), area_eps=float(args.area_eps))
    ][: max(0, int(args.max_exports))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    weak_config = WeakExplainerConfig()
    render_config = WeakRenderConfig(validity_eps=float(args.validity_eps))
    manifest = []
    for index, row in enumerate(rows):
        source_path = Path(row["source_file"])
        stem = str(row.get("stem") or source_path.stem)
        prefix = f"{index:03d}_{row.get('source_kind', 'source')}_{stem}"
        try:
            evidence, weak, render_payload, _global = build_case(
                source_path,
                source_kind=row.get("source_kind", "partition"),
                split=str(row.get("split", "val")),
                mask_root=args.mask_root,
                global_config=global_config,
                evidence_config=evidence_config,
                weak_config=weak_config,
                render_config=render_config,
            )
            evidence_path = args.output_dir / f"{prefix}_evidence.json"
            weak_path = args.output_dir / f"{prefix}_weak.json"
            render_path = args.output_dir / f"{prefix}_render.json"
            mask_path = args.output_dir / f"{prefix}_mask.png"
            validation_path = args.output_dir / f"{prefix}_validation.json"
            image_path = args.output_dir / f"{prefix}_viz.png"
            write_json(evidence_path, evidence)
            write_json(weak_path, weak)
            save_render_outputs(render_payload, partition_path=render_path, mask_path=mask_path, validation_path=validation_path)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "visualize_weak_explanation.py"),
                    "--weak-json",
                    str(weak_path),
                    "--evidence-json",
                    str(evidence_path),
                    "--mask-root",
                    str(args.mask_root),
                    "--output",
                    str(image_path),
                ],
                check=False,
            )
            manifest.append(
                {
                    "source_row": row,
                    "evidence_json": str(evidence_path.as_posix()),
                    "weak_json": str(weak_path.as_posix()),
                    "render_json": str(render_path.as_posix()),
                    "mask_png": str(mask_path.as_posix()),
                    "validation_json": str(validation_path.as_posix()),
                    "visualization_png": str(image_path.as_posix()),
                }
            )
            print(f"exported {prefix}")
        except Exception as error:  # noqa: BLE001 - exporter should continue.
            manifest.append({"source_row": row, "export_failure": f"{type(error).__name__}: {error}"})
            print(f"failed {prefix}: {type(error).__name__}: {error}")
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, {"count": len(manifest), "items": manifest})
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
