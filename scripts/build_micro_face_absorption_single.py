from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json  # noqa: E402
from partition_gen.micro_face_absorber import (  # noqa: E402
    MicroFaceAbsorptionConfig,
    absorb_micro_faces_from_global_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Absorb tiny closed faces into adjacent larger faces.")
    parser.add_argument("--global-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-area", type=float, default=4.0)
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--min-shared-length", type=float, default=0.5)
    parser.add_argument("--labels", type=int, nargs="*", default=[])
    parser.add_argument("--absorb-small-islands", action="store_true")
    parser.add_argument("--island-max-area", type=float, default=4.0)
    parser.add_argument("--island-min-shared-length", type=float, default=0.5)
    parser.add_argument("--island-labels", type=int, nargs="*", default=[])
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    global_payload = load_json(args.global_json)
    payload = absorb_micro_faces_from_global_payload(
        global_payload,
        config=MicroFaceAbsorptionConfig(
            max_area=float(args.max_area),
            max_vertices=int(args.max_vertices),
            min_shared_length=float(args.min_shared_length),
            labels=tuple(int(value) for value in args.labels),
            absorb_small_islands=bool(args.absorb_small_islands),
            island_max_area=float(args.island_max_area),
            island_min_shared_length=float(args.island_min_shared_length),
            island_labels=tuple(int(value) for value in args.island_labels),
            validity_eps=float(args.validity_eps),
        ),
    )
    dump_json(args.output, payload)
    summary = payload["micro_face_absorption"]
    validation = payload["validation"]
    print(
        f"absorbed micro faces {args.global_json}: "
        f"candidates={summary['candidate_count']}, absorbed={summary['absorbed_count']}, "
        f"rejected={summary['rejected_count']}, faces={summary['input_face_count']}->{summary['output_face_count']}, "
        f"valid={validation['is_valid']}, iou={validation['union_iou']:.6f}, overlap={validation['overlap_area']:.6f}"
    )


if __name__ == "__main__":
    main()
