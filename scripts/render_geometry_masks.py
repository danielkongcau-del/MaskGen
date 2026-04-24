from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.geometry_render import load_json, remap_ids_to_values, render_geometry_prediction
from partition_gen.joint_render import render_joint_partition_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render predicted geometry JSONs into 256x256 mask PNGs.")
    parser.add_argument("--prediction-root", type=Path, required=True, help="Root containing prediction graphs/*.json and summary.json.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Compare rendered masks with source ground-truth masks when available.")
    parser.add_argument("--renderer", type=str, choices=["independent", "joint_boundary"], default="independent")
    parser.add_argument("--boundary-dilation", type=int, default=1)
    parser.add_argument("--joint-use-all-faces", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.prediction_root / "rendered")
    output_dir.mkdir(parents=True, exist_ok=True)
    id_map = load_json(args.dual_root / "meta" / "class_map.json")["id_to_value"]

    prediction_paths = sorted((args.prediction_root / "graphs").glob("*.json"))
    pixel_accuracies: List[float] = []
    report_items = []

    for prediction_path in prediction_paths:
        prediction = load_json(prediction_path)
        dual_path = Path(prediction["source_dual_graph"])
        if not dual_path.is_absolute():
            dual_path = Path(prediction["source_dual_graph"])
        dual_graph = load_json(dual_path)
        size = tuple(int(value) for value in dual_graph["size"])

        if args.renderer == "independent":
            mask_id, render_meta = render_geometry_prediction(
                dual_graph=dual_graph,
                prediction=prediction,
                size=size,
            )
        else:
            mask_id, render_meta = render_joint_partition_prediction(
                dual_graph=dual_graph,
                prediction=prediction,
                size=size,
                boundary_dilation=args.boundary_dilation,
                use_all_faces=bool(args.joint_use_all_faces),
            )
        mask_value = remap_ids_to_values(mask_id, id_map)

        split = dual_path.parent.parent.name
        stem = dual_path.stem
        (output_dir / "masks_id").mkdir(parents=True, exist_ok=True)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_id, mode="L").save(output_dir / "masks_id" / f"{stem}.png")
        Image.fromarray(mask_value, mode="L").save(output_dir / "masks" / f"{stem}.png")

        item = {
            "prediction_path": str(prediction_path.as_posix()),
            "dual_path": str(dual_path.as_posix()),
            "split": split,
            "stem": stem,
            "renderer": args.renderer,
            **render_meta["summary"],
        }

        if args.evaluate and dual_graph.get("source_mask") is not None:
            gt_mask_path = args.mask_root / str(dual_graph["source_mask"])
            if gt_mask_path.exists():
                gt_mask = np.array(Image.open(gt_mask_path))
                pixel_accuracy = float((gt_mask == mask_id).mean())
                pixel_accuracies.append(pixel_accuracy)
                item["pixel_accuracy"] = pixel_accuracy

        report_items.append(item)

    summary = {
        "prediction_root": str(args.prediction_root.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "renderer": args.renderer,
        "num_predictions": len(report_items),
        "mean_pixel_accuracy": float(np.mean(pixel_accuracies)) if pixel_accuracies else None,
        "reports": report_items,
    }
    save_json(output_dir / "render_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
