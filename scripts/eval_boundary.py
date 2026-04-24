from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.boundary_dataset import BoundaryGraphDataset, collate_boundary_graphs
from partition_gen.geometry_render import load_json
from partition_gen.joint_render import render_partition_from_boundaries
from partition_gen.models.boundary_predictor import build_boundary_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph-conditioned boundary prediction.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    max_neighbors = int(train_args.get("max_neighbors", 0))
    dataset = BoundaryGraphDataset(
        dual_root=args.dual_root,
        boundary_root=args.boundary_root,
        split=args.split,
        binners=binners,
        max_faces=int(train_args["max_faces"]),
        max_neighbors=max_neighbors if max_neighbors > 0 else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_boundary_graphs)

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

    tp = fp = fn = 0.0
    rendered_pixel_accuracies = []

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

            pred_mask = probs >= args.threshold
            tp += float(np.logical_and(pred_mask, targets > 0.5).sum())
            fp += float(np.logical_and(pred_mask, targets <= 0.5).sum())
            fn += float(np.logical_and(~pred_mask, targets > 0.5).sum())

            for row, dual_path in enumerate(batch["paths"]):
                dual_graph = load_json(Path(dual_path))
                boundary_mask = pred_mask[row, 0]
                label_map, _ = render_partition_from_boundaries(
                    dual_graph=dual_graph,
                    boundary_mask=boundary_mask,
                    size=tuple(int(v) for v in dual_graph["size"]),
                    use_all_faces=True,
                )
                gt_mask = dual_graph.get("source_mask")
                if gt_mask is not None:
                    gt = np.array(Image.open(args.mask_root / str(gt_mask)))
                    rendered_pixel_accuracies.append(float((label_map == gt).mean()))

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1.0)

    results = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "split": args.split,
        "num_graphs": len(dataset),
        "threshold": args.threshold,
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
        "boundary_iou": float(iou),
        "rendered_mean_pixel_accuracy": float(np.mean(rendered_pixel_accuracies)) if rendered_pixel_accuracies else None,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
