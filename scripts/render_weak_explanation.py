from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.weak_parse_graph_renderer import (  # noqa: E402
    WeakRenderConfig,
    render_weak_explanation_payload,
    save_render_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and validate one weak explanation parse_graph.")
    parser.add_argument("--weak-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, default=None)
    parser.add_argument("--output-validation", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    weak_payload = load_json(args.weak_json)
    evidence_payload = load_json(args.evidence_json)
    render_payload = render_weak_explanation_payload(
        weak_payload,
        evidence_payload=evidence_payload,
        config=WeakRenderConfig(area_eps=args.area_eps, validity_eps=args.validity_eps),
        mask_root=args.mask_root,
    )
    validation = render_payload["validation"]
    save_render_outputs(
        render_payload,
        partition_path=args.output_json,
        mask_path=args.output_mask,
        validation_path=args.output_validation,
    )
    print(
        "rendered weak explanation: "
        f"faces={validation['face_count']}, "
        f"valid={validation['is_valid']}, "
        f"full_iou={validation['full_iou']}, "
        f"mask_pixel_accuracy={validation['mask_pixel_accuracy']}, "
        f"overlap_area={validation['overlap_area']:.6f}, "
        f"gap_area={validation['gap_area']}, "
        f"low_iou_faces={len(validation['low_iou_face_ids'])}"
    )


if __name__ == "__main__":
    main()
