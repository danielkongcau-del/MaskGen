from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.boundary_dataset import BoundaryGraphDataset, collate_boundary_graphs
from partition_gen.geometry_render import load_json, remap_ids_to_values
from partition_gen.joint_render import render_partition_from_boundaries
from partition_gen.models.boundary_predictor import build_boundary_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict shared boundaries from dual graphs and render masks.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/boundary_predictions"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ["boundary_probs", "boundary_masks", "masks_id", "masks", "reports"]:
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)
    id_map = load_json(args.dual_root / "meta" / "class_map.json")["id_to_value"]

    max_neighbors = int(train_args.get("max_neighbors", 0))
    dataset = BoundaryGraphDataset(
        dual_root=args.dual_root,
        boundary_root=args.boundary_root,
        split=args.split,
        binners=binners,
        max_faces=int(train_args["max_faces"]),
        max_neighbors=max_neighbors if max_neighbors > 0 else None,
    )
    subset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_boundary_graphs)

    model = build_boundary_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=int(train_args["max_faces"]),
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_layers=int(train_args["num_layers"]),
        base_size=int(train_args.get("base_size", 64)),
        raster_channels=int(train_args.get("raster_channels", 96)),
        scene_channels=int(train_args.get("scene_channels", 32)),
        decoder_hidden=int(train_args.get("decoder_hidden", 128)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    summaries: List[Dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                node_features=batch["node_features"].to(device),
                face_mask=batch["face_mask"].to(device),
                neighbor_indices=batch["neighbor_indices"].to(device),
                neighbor_tokens=batch["neighbor_tokens"].to(device),
                neighbor_mask=batch["neighbor_mask"].to(device),
                centroid_ratios=batch["centroid_ratios"].to(device),
                bbox_ratios=batch["bbox_ratios"].to(device),
                labels=batch["labels"].to(device),
            )
            probs = torch.sigmoid(outputs["boundary_logits"]).cpu().numpy()
            targets = batch["boundary_mask"].numpy()

            for row, dual_path_str in enumerate(batch["paths"]):
                dual_path = Path(dual_path_str)
                dual_graph = load_json(dual_path)
                prob = probs[row, 0]
                boundary_mask = prob >= args.threshold
                gt_boundary = targets[row, 0] > 0.5

                Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L").save(
                    args.output_dir / "boundary_probs" / f"{sample_index:04d}.png"
                )
                Image.fromarray(boundary_mask.astype(np.uint8) * 255, mode="L").save(
                    args.output_dir / "boundary_masks" / f"{sample_index:04d}.png"
                )

                mask_id, render_meta = render_partition_from_boundaries(
                    dual_graph=dual_graph,
                    boundary_mask=boundary_mask,
                    size=tuple(int(v) for v in dual_graph["size"]),
                    use_all_faces=True,
                )
                mask_value = remap_ids_to_values(mask_id, id_map)
                Image.fromarray(mask_id, mode="L").save(args.output_dir / "masks_id" / f"{sample_index:04d}.png")
                Image.fromarray(mask_value, mode="L").save(args.output_dir / "masks" / f"{sample_index:04d}.png")

                boundary_intersection = float(np.logical_and(boundary_mask, gt_boundary).sum())
                boundary_union = float(np.logical_or(boundary_mask, gt_boundary).sum())
                boundary_iou = boundary_intersection / max(boundary_union, 1.0)

                pixel_accuracy = None
                if dual_graph.get("source_mask") is not None:
                    gt_mask = np.array(Image.open(args.mask_root / str(dual_graph["source_mask"])))
                    pixel_accuracy = float((mask_id == gt_mask).mean())

                item = {
                    "sample_index": sample_index,
                    "dual_path": str(dual_path.as_posix()),
                    "boundary_iou": boundary_iou,
                    "pixel_accuracy": pixel_accuracy,
                    **render_meta["summary"],
                }
                summaries.append(item)
                save_json(args.output_dir / "reports" / f"{sample_index:04d}.json", item)
                sample_index += 1

    summary = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "split": args.split,
        "num_graphs": len(summaries),
        "threshold": args.threshold,
        "mean_boundary_iou": float(np.mean([item["boundary_iou"] for item in summaries])) if summaries else None,
        "mean_pixel_accuracy": (
            float(np.mean([item["pixel_accuracy"] for item in summaries if item["pixel_accuracy"] is not None]))
            if [item["pixel_accuracy"] for item in summaries if item["pixel_accuracy"] is not None]
            else None
        ),
        "reports": summaries,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
