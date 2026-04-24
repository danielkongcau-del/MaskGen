from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.boundary_dataset import PairBoundaryGraphDataset, collate_pair_boundary_graphs
from partition_gen.geometry_render import load_json
from partition_gen.joint_render import render_partition_from_boundaries
from partition_gen.models.pair_boundary_predictor import build_pair_boundary_model_from_metadata
from partition_gen.pair_render import render_partition_from_pair_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pair-level boundary prediction.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--renderer", type=str, choices=["union", "pair_aware"], default="pair_aware")
    parser.add_argument("--max-candidate-pairs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return intersection / max(union, 1.0)


def upsample_boundary_mask(mask: np.ndarray, render_size: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    up = F.interpolate(tensor, size=(render_size, render_size), mode="bilinear", align_corners=False)
    return (up[0, 0].numpy() >= 0.5)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    max_neighbors = int(train_args.get("max_neighbors", 0))
    max_pairs = int(train_args.get("max_pairs", 0))
    target_size = int(train_args.get("target_size", 64))
    dataset = PairBoundaryGraphDataset(
        dual_root=args.dual_root,
        boundary_root=args.boundary_root,
        split=args.split,
        binners=binners,
        max_faces=int(train_args["max_faces"]),
        max_neighbors=max_neighbors if max_neighbors > 0 else None,
        max_pairs=max_pairs if max_pairs > 0 else None,
        target_size=target_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pair_boundary_graphs)

    model = build_pair_boundary_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=int(train_args["max_faces"]),
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_layers=int(train_args["num_layers"]),
        target_size=target_size,
        pair_hidden=int(train_args.get("pair_hidden", 128)),
        pair_token_channels=int(train_args.get("pair_token_channels", 32)),
        decoder_hidden=int(train_args.get("decoder_hidden", 64)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    pair_ious = []
    union_ious = []
    rendered_pixel_accuracies = []
    ambiguous_boundary_pixels = []
    boundary_pixels = []

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
                pair_indices=batch["pair_indices"].to(device),
                pair_features=batch["pair_features"].to(device),
                pair_valid=batch["pair_valid"].to(device),
            )
            pair_probs = torch.sigmoid(outputs["pair_logits"]).cpu().numpy()
            pair_preds = pair_probs >= args.threshold
            pair_targets = batch["pair_masks"].numpy() > 0.5
            union_pred = pair_preds.max(axis=1)[:, 0]
            union_target = batch["union_mask"].numpy()[:, 0] > 0.5

            for row, dual_path_str in enumerate(batch["paths"]):
                valid_pairs = batch["pair_valid"][row].numpy().astype(bool)
                for pair_index, is_valid in enumerate(valid_pairs):
                    if not is_valid:
                        continue
                    pair_ious.append(mask_iou(pair_preds[row, pair_index, 0], pair_targets[row, pair_index, 0]))
                union_ious.append(mask_iou(union_pred[row], union_target[row]))

                dual_graph = load_json(Path(dual_path_str))
                if args.renderer == "union":
                    boundary_256 = upsample_boundary_mask(union_pred[row], args.render_size)
                    label_map, _ = render_partition_from_boundaries(
                        dual_graph=dual_graph,
                        boundary_mask=boundary_256,
                        size=(args.render_size, args.render_size),
                        use_all_faces=True,
                    )
                else:
                    label_map, render_meta = render_partition_from_pair_masks(
                        dual_graph=dual_graph,
                        pair_masks=pair_probs[row],
                        pair_indices=batch["pair_indices"][row].numpy(),
                        pair_valid=batch["pair_valid"][row].numpy().astype(bool),
                        threshold=args.threshold,
                        render_size=args.render_size,
                        max_candidate_pairs=args.max_candidate_pairs,
                    )
                    ambiguous_boundary_pixels.append(float(render_meta["summary"]["ambiguous_boundary_pixels"]))
                    boundary_pixels.append(float(render_meta["summary"]["boundary_pixels"]))
                if dual_graph.get("source_mask") is not None:
                    gt_mask = np.array(Image.open(args.mask_root / str(dual_graph["source_mask"])))
                    rendered_pixel_accuracies.append(float((label_map == gt_mask).mean()))

    results = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "split": args.split,
        "num_graphs": len(dataset),
        "threshold": args.threshold,
        "renderer": args.renderer,
        "max_candidate_pairs": args.max_candidate_pairs,
        "pair_mean_iou": float(np.mean(pair_ious)) if pair_ious else None,
        "union_mean_iou": float(np.mean(union_ious)) if union_ious else None,
        "rendered_mean_pixel_accuracy": float(np.mean(rendered_pixel_accuracies)) if rendered_pixel_accuracies else None,
        "mean_ambiguous_boundary_pixels": float(np.mean(ambiguous_boundary_pixels)) if ambiguous_boundary_pixels else None,
        "mean_boundary_pixels": float(np.mean(boundary_pixels)) if boundary_pixels else None,
        "mean_ambiguity_ratio": (
            float(np.mean(np.asarray(ambiguous_boundary_pixels) / np.maximum(np.asarray(boundary_pixels), 1.0)))
            if ambiguous_boundary_pixels
            else None
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
