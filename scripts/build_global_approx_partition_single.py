from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.global_approx_partition import (  # noqa: E402
    GlobalApproxConfig,
    build_global_approx_partition_from_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-image shared-arc geometry approximation.")
    parser.add_argument("--partition-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplify-tolerance", type=float, default=1.0, help="Reserved for raw fallback experiments.")
    parser.add_argument("--simplify-backoff", type=float, default=0.5)
    parser.add_argument("--max-simplify-attempts", type=int, default=8)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    payload = build_global_approx_partition_from_path(
        args.partition_graph,
        config=GlobalApproxConfig(
            simplify_tolerance=args.simplify_tolerance,
            simplify_backoff=args.simplify_backoff,
            max_simplify_attempts=args.max_simplify_attempts,
            face_simplify_tolerance=args.face_simplify_tolerance,
            face_area_epsilon=args.face_area_epsilon,
            area_eps=args.area_eps,
            validity_eps=args.validity_eps,
        ),
    )
    dump_json(args.output, payload)
    validation = payload["validation"]
    print(
        f"built global approx {args.partition_graph}: "
        f"faces={validation['face_count']}, arcs={validation['arc_count']}, "
        f"shared_arcs={validation['shared_arc_count']}, "
        f"accepted_owner_arcs={payload['reconciliation']['accepted_count']}, "
        f"valid={validation['is_valid']}, iou={validation['union_iou']:.6f}, "
        f"overlap={validation['overlap_area']:.6f}"
    )


if __name__ == "__main__":
    main()
