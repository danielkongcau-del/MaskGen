from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json  # noqa: E402
from partition_gen.global_arc_regularizer import (  # noqa: E402
    GlobalArcRegularizationConfig,
    regularize_global_arc_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regularize staircase-like arcs in a full-image global approximation.")
    parser.add_argument("--global-json", type=Path, required=True)
    parser.add_argument("--partition-graph", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplify-tolerance", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=1.25)
    parser.add_argument("--min-vertex-reduction", type=int, default=1)
    parser.add_argument("--min-arc-length", type=float, default=4.0)
    parser.add_argument("--disable-subsegment-smoothing", action="store_true")
    parser.add_argument("--max-subsegment-span", type=int, default=64)
    parser.add_argument("--max-candidates-per-arc", type=int, default=64)
    parser.add_argument("--enable-face-chain-smoothing", action="store_true")
    parser.add_argument("--face-chain-max-distance", type=float, default=2.0)
    parser.add_argument("--face-chain-min-length", type=float, default=6.0)
    parser.add_argument("--face-chain-max-span", type=int, default=96)
    parser.add_argument("--max-face-chain-candidates", type=int, default=256)
    parser.add_argument("--enable-strip-face-smoothing", action="store_true")
    parser.add_argument("--strip-min-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--strip-max-width", type=float, default=14.0)
    parser.add_argument("--strip-min-length", type=float, default=10.0)
    parser.add_argument("--strip-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--max-strip-face-candidates", type=int, default=64)
    parser.add_argument("--allow-polyline-smoothing", action="store_true")
    parser.add_argument("--include-exterior-arcs", action="store_true")
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    global_payload = load_json(args.global_json)
    graph_data = load_json(args.partition_graph) if args.partition_graph is not None else None
    payload = regularize_global_arc_payload(
        global_payload,
        graph_data=graph_data,
        config=GlobalArcRegularizationConfig(
            simplify_tolerance=float(args.simplify_tolerance),
            max_distance=float(args.max_distance),
            min_vertex_reduction=int(args.min_vertex_reduction),
            min_arc_length=float(args.min_arc_length),
            enable_subsegment_smoothing=not bool(args.disable_subsegment_smoothing),
            max_subsegment_span=int(args.max_subsegment_span),
            max_candidates_per_arc=int(args.max_candidates_per_arc),
            enable_face_chain_smoothing=bool(args.enable_face_chain_smoothing),
            face_chain_max_distance=float(args.face_chain_max_distance),
            face_chain_min_length=float(args.face_chain_min_length),
            face_chain_max_span=int(args.face_chain_max_span),
            max_face_chain_candidates=int(args.max_face_chain_candidates),
            enable_strip_face_smoothing=bool(args.enable_strip_face_smoothing),
            strip_min_aspect_ratio=float(args.strip_min_aspect_ratio),
            strip_max_width=float(args.strip_max_width),
            strip_min_length=float(args.strip_min_length),
            strip_max_area_ratio=float(args.strip_max_area_ratio),
            max_strip_face_candidates=int(args.max_strip_face_candidates),
            allow_polyline_smoothing=bool(args.allow_polyline_smoothing),
            include_exterior_arcs=bool(args.include_exterior_arcs),
            validity_eps=float(args.validity_eps),
        ),
    )
    dump_json(args.output, payload)
    validation = payload["validation"]
    regularization = payload["arc_regularization"]
    print(
        f"regularized global approx {args.global_json}: "
        f"candidates={regularization['candidate_count']}, "
        f"accepted={regularization['accepted_count']}, rejected={regularization['rejected_count']}, "
        f"vertices={regularization['input_arc_vertex_count']}->{regularization['output_arc_vertex_count']}, "
        f"valid={validation['is_valid']}, iou={validation['union_iou']:.6f}, overlap={validation['overlap_area']:.6f}"
    )


if __name__ == "__main__":
    main()
