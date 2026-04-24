from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.convex_partition import ConvexMergeConfig, build_convex_partition_from_geometry_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build constrained-triangulation + greedy convex-merge partition from geometry approximator JSON.")
    parser.add_argument("--approx-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-rel-eps", type=float, default=1e-6)
    parser.add_argument("--convex-abs-eps", type=float, default=1e-8)
    parser.add_argument("--shared-edge-eps", type=float, default=1e-6)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--vertex-round-digits", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), indent=2)


def main() -> None:
    args = parse_args()
    config = ConvexMergeConfig(
        convex_rel_eps=args.convex_rel_eps,
        convex_abs_eps=args.convex_abs_eps,
        shared_edge_eps=args.shared_edge_eps,
        area_eps=args.area_eps,
        vertex_round_digits=args.vertex_round_digits,
    )
    geometry_payload = load_json(args.approx_json)
    payload = build_convex_partition_from_geometry_payload(
        geometry_payload,
        config=config,
        source_tag=str(args.approx_json.as_posix()),
    )
    dump_json(args.output, payload)
    print(
        f"built face {payload['face_id']} from {args.approx_json.name}: "
        f"triangles={payload['triangle_count']}, final={payload['final_primitive_count']}, "
        f"holes={payload['hole_count']}, iou={payload['approx_iou']:.6f}"
    )


if __name__ == "__main__":
    main()
