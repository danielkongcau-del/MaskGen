from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.convex_partition import ConvexMergeConfig, build_face_convex_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build constrained-triangulation + greedy convex-merge partition for one face.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--face-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-rel-eps", type=float, default=1e-6)
    parser.add_argument("--convex-abs-eps", type=float, default=1e-8)
    parser.add_argument("--shared-edge-eps", type=float, default=1e-6)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--vertex-round-digits", type=int, default=8)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), indent=2)


def main() -> None:
    args = parse_args()
    graph_path = args.partition_root / args.split / "graphs" / f"{args.stem}.json"
    config = ConvexMergeConfig(
        convex_rel_eps=args.convex_rel_eps,
        convex_abs_eps=args.convex_abs_eps,
        shared_edge_eps=args.shared_edge_eps,
        area_eps=args.area_eps,
        vertex_round_digits=args.vertex_round_digits,
    )
    payload = build_face_convex_partition(
        graph_path,
        face_id=args.face_id,
        config=config,
    )
    dump_json(args.output, payload)
    print(
        f"built face {args.face_id} from {graph_path.name}: "
        f"triangles={payload['triangle_count']}, final={payload['final_primitive_count']}, "
        f"holes={payload['hole_count']}, iou={payload['approx_iou']:.6f}"
    )


if __name__ == "__main__":
    main()
