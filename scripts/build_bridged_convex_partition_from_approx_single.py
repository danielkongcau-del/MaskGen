from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    build_bridged_convex_partition_from_geometry_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bridged convex partition from geometry approximator JSON.")
    parser.add_argument("--approx-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cgal", "fallback_hm", "fallback_cdt_greedy"])
    parser.add_argument("--cgal-cli", type=str, default=None)
    parser.add_argument("--max-bridge-sets", type=int, default=256)
    parser.add_argument("--vertex-round-digits", type=int, default=8)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--validity-eps", type=float, default=1e-7)
    parser.add_argument("--cut-slit-scale", type=float, default=1e-6)
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
    geometry_payload = load_json(args.approx_json)
    payload = build_bridged_convex_partition_from_geometry_payload(
        geometry_payload,
        config=BridgedPartitionConfig(
            max_bridge_sets=args.max_bridge_sets,
            vertex_round_digits=args.vertex_round_digits,
            area_eps=args.area_eps,
            validity_eps=args.validity_eps,
            cut_slit_scale=args.cut_slit_scale,
            backend=args.backend,
            cgal_cli=args.cgal_cli,
        ),
        source_tag=str(args.approx_json.as_posix()),
    )
    dump_json(args.output, payload)
    backend_info = payload["backend_info"]
    bridge_set = payload.get("selected_bridge_set") or {"bridge_ids": []}
    print(
        f"built bridged partition face {payload['face_id']}: "
        f"holes={payload['hole_count']}, bridges={len(bridge_set['bridge_ids'])}, "
        f"pieces={payload['final_primitive_count']}, iou={payload['validation']['iou']:.6f}, "
        f"backend={backend_info['backend']}, optimal={backend_info['optimal']}"
    )


if __name__ == "__main__":
    main()
