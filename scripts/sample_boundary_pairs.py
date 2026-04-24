from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
from partition_gen.geometry_render import load_json, remap_ids_to_values
from partition_gen.joint_render import render_partition_from_boundaries
from partition_gen.models.pair_boundary_predictor import build_pair_boundary_model_from_metadata
from partition_gen.pair_render import render_partition_from_pair_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pair-level boundary masks and render their union.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/boundary_pair_predictions"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--renderer", type=str, choices=["union", "pair_aware"], default="pair_aware")
    parser.add_argument("--max-candidate-pairs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def upsample_boundary_mask(mask: np.ndarray, render_size: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    up = F.interpolate(tensor, size=(render_size, render_size), mode="bilinear", align_corners=False)
    return (up[0, 0].numpy() >= 0.5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ["pair_masks", "union_masks", "masks_id", "masks", "reports"]:
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)
    id_map = load_json(args.dual_root / "meta" / "class_map.json")["id_to_value"]

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
    subset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pair_boundary_graphs)

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
                pair_indices=batch["pair_indices"].to(device),
                pair_features=batch["pair_features"].to(device),
                pair_valid=batch["pair_valid"].to(device),
            )
            pair_probs = torch.sigmoid(outputs["pair_logits"]).cpu().numpy()
            pair_preds = pair_probs >= args.threshold
            union_pred = pair_preds.max(axis=1)[:, 0]
            union_target = batch["union_mask"].numpy()[:, 0] > 0.5
            pair_targets = batch["pair_masks"].numpy() > 0.5

            for row, dual_path_str in enumerate(batch["paths"]):
                dual_path = Path(dual_path_str)
                dual_graph = load_json(dual_path)
                valid_pairs = batch["pair_valid"][row].numpy().astype(bool)
                pair_ious = []
                for pair_index, is_valid in enumerate(valid_pairs):
                    if not is_valid:
                        continue
                    pred_mask = pair_preds[row, pair_index, 0]
                    target_mask = pair_targets[row, pair_index, 0]
                    intersection = float(np.logical_and(pred_mask, target_mask).sum())
                    union = float(np.logical_or(pred_mask, target_mask).sum())
                    pair_ious.append(intersection / max(union, 1.0))
                    Image.fromarray(pred_mask.astype(np.uint8) * 255, mode="L").save(
                        args.output_dir / "pair_masks" / f"{sample_index:04d}_{pair_index:03d}.png"
                    )

                union_64 = union_pred[row]
                union_256 = upsample_boundary_mask(union_64, args.render_size)
                Image.fromarray(union_64.astype(np.uint8) * 255, mode="L").save(
                    args.output_dir / "union_masks" / f"{sample_index:04d}_64.png"
                )
                Image.fromarray(union_256.astype(np.uint8) * 255, mode="L").save(
                    args.output_dir / "union_masks" / f"{sample_index:04d}_256.png"
                )
                if args.renderer == "union":
                    mask_id, render_meta = render_partition_from_boundaries(
                        dual_graph=dual_graph,
                        boundary_mask=union_256,
                        size=(args.render_size, args.render_size),
                        use_all_faces=True,
                    )
                else:
                    mask_id, render_meta = render_partition_from_pair_masks(
                        dual_graph=dual_graph,
                        pair_masks=pair_probs[row],
                        pair_indices=batch["pair_indices"][row].numpy(),
                        pair_valid=batch["pair_valid"][row].numpy().astype(bool),
                        threshold=args.threshold,
                        render_size=args.render_size,
                        max_candidate_pairs=args.max_candidate_pairs,
                    )
                mask_value = remap_ids_to_values(mask_id, id_map)
                Image.fromarray(mask_id, mode="L").save(args.output_dir / "masks_id" / f"{sample_index:04d}.png")
                Image.fromarray(mask_value, mode="L").save(args.output_dir / "masks" / f"{sample_index:04d}.png")

                union_intersection = float(np.logical_and(union_64, union_target[row]).sum())
                union_union = float(np.logical_or(union_64, union_target[row]).sum())
                union_iou = union_intersection / max(union_union, 1.0)

                pixel_accuracy = None
                if dual_graph.get("source_mask") is not None:
                    gt_mask = np.array(Image.open(args.mask_root / str(dual_graph["source_mask"])))
                    pixel_accuracy = float((mask_id == gt_mask).mean())

                item = {
                    "sample_index": sample_index,
                    "dual_path": str(dual_path.as_posix()),
                    "mean_pair_iou": float(np.mean(pair_ious)) if pair_ious else None,
                    "union_iou": union_iou,
                    "pixel_accuracy": pixel_accuracy,
                    "num_valid_pairs": int(valid_pairs.sum()),
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
        "renderer": args.renderer,
        "max_candidate_pairs": args.max_candidate_pairs,
        "mean_pair_iou": (
            float(np.mean([item["mean_pair_iou"] for item in summaries if item["mean_pair_iou"] is not None]))
            if [item["mean_pair_iou"] for item in summaries if item["mean_pair_iou"] is not None]
            else None
        ),
        "mean_union_iou": float(np.mean([item["union_iou"] for item in summaries])) if summaries else None,
        "mean_pixel_accuracy": (
            float(np.mean([item["pixel_accuracy"] for item in summaries if item["pixel_accuracy"] is not None]))
            if [item["pixel_accuracy"] for item in summaries if item["pixel_accuracy"] is not None]
            else None
        ),
        "mean_ambiguous_boundary_pixels": (
            float(np.mean([item["ambiguous_boundary_pixels"] for item in summaries]))
            if summaries and "ambiguous_boundary_pixels" in summaries[0]
            else None
        ),
        "mean_boundary_pixels": (
            float(np.mean([item["boundary_pixels"] for item in summaries]))
            if summaries and "boundary_pixels" in summaries[0]
            else None
        ),
        "mean_ambiguity_ratio": (
            float(
                np.mean(
                    [
                        item["ambiguous_boundary_pixels"] / max(item["boundary_pixels"], 1)
                        for item in summaries
                        if "ambiguous_boundary_pixels" in item and "boundary_pixels" in item
                    ]
                )
            )
            if summaries and "ambiguous_boundary_pixels" in summaries[0]
            else None
        ),
        "reports": summaries,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
