from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.geometry_dataset import GeometryGraphDataset, collate_geometry_graphs
from partition_gen.models.geometry_decoder import build_geometry_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the graph-conditioned geometry decoder.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--geometry-root", type=Path, default=Path("data/remote_256_geometry"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
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
    dataset = GeometryGraphDataset(
        dual_root=args.dual_root,
        geometry_root=args.geometry_root,
        split=args.split,
        binners=binners,
        max_faces=int(train_args["max_faces"]),
        max_neighbors=max_neighbors if max_neighbors > 0 else None,
        max_vertices=int(train_args["max_vertices"]),
        max_holes=int(train_args.get("max_holes", 0)),
        max_hole_vertices=int(train_args.get("max_hole_vertices", 0)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_geometry_graphs)

    model = build_geometry_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=int(train_args["max_faces"]),
        max_neighbors=max(1, max_neighbors) if max_neighbors > 0 else 32,
        max_vertices=int(train_args["max_vertices"]),
        max_holes=int(train_args.get("max_holes", 0)),
        max_hole_vertices=int(train_args.get("max_hole_vertices", 0)),
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_layers=int(train_args["num_layers"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    support_correct = 0
    support_total = 0
    count_correct = 0
    count_total = 0
    count_abs_error = 0.0
    coord_abs_error = 0.0
    coord_total = 0
    hole_count_correct = 0
    hole_count_total = 0
    hole_count_abs_error = 0.0
    hole_vertex_count_correct = 0
    hole_vertex_count_total = 0
    hole_vertex_count_abs_error = 0.0
    hole_coord_abs_error = 0.0
    hole_coord_total = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                node_features=batch["node_features"].to(device),
                face_mask=batch["face_mask"].to(device),
                neighbor_indices=batch["neighbor_indices"].to(device),
                neighbor_tokens=batch["neighbor_tokens"].to(device),
                neighbor_mask=batch["neighbor_mask"].to(device),
            )
            face_mask = batch["face_mask"]
            gt_support = batch["geometry_support"]
            pred_support = torch.argmax(outputs["support_logits"], dim=-1).cpu()
            mask = face_mask.bool()
            support_correct += int((pred_support[mask] == gt_support[mask]).sum().item())
            support_total += int(mask.sum().item())

            gt_counts = batch["vertex_counts"]
            pred_counts = torch.argmax(outputs["vertex_count_logits"], dim=-1).cpu()
            supported_mask = mask & gt_support.bool()
            count_correct += int((pred_counts[supported_mask] == gt_counts[supported_mask]).sum().item())
            count_total += int(supported_mask.sum().item())
            count_abs_error += float((pred_counts[supported_mask] - gt_counts[supported_mask]).abs().sum().item())

            vertex_mask = batch["vertex_mask"] & gt_support.bool().unsqueeze(-1)
            pred_coords = outputs["vertex_coords"].cpu()
            coord_abs_error += float((pred_coords[vertex_mask] - batch["vertices"][vertex_mask]).abs().sum().item())
            coord_total += int(vertex_mask.sum().item()) * 2

            if outputs["hole_count_logits"].shape[-1] > 0:
                gt_hole_counts = batch["hole_counts"]
                pred_hole_counts = torch.argmax(outputs["hole_count_logits"], dim=-1).cpu()
                hole_count_correct += int((pred_hole_counts[supported_mask] == gt_hole_counts[supported_mask]).sum().item())
                hole_count_total += int(supported_mask.sum().item())
                hole_count_abs_error += float((pred_hole_counts[supported_mask] - gt_hole_counts[supported_mask]).abs().sum().item())

            if outputs["hole_vertex_count_logits"].numel() > 0:
                gt_hole_vertex_counts = batch["hole_vertex_counts"]
                pred_hole_vertex_counts = torch.argmax(outputs["hole_vertex_count_logits"], dim=-1).cpu()
                hole_slot_mask = supported_mask.unsqueeze(-1).expand_as(gt_hole_vertex_counts)
                hole_vertex_count_correct += int((pred_hole_vertex_counts[hole_slot_mask] == gt_hole_vertex_counts[hole_slot_mask]).sum().item())
                hole_vertex_count_total += int(hole_slot_mask.sum().item())
                hole_vertex_count_abs_error += float((pred_hole_vertex_counts[hole_slot_mask] - gt_hole_vertex_counts[hole_slot_mask]).abs().sum().item())

            if outputs["hole_vertex_coords"].numel() > 0:
                pred_hole_coords = outputs["hole_vertex_coords"].cpu()
                hole_vertex_mask = batch["hole_vertex_mask"] & batch["hole_mask"].unsqueeze(-1) & gt_support.bool().unsqueeze(-1).unsqueeze(-1)
                hole_coord_abs_error += float((pred_hole_coords[hole_vertex_mask] - batch["hole_vertices"][hole_vertex_mask]).abs().sum().item())
                hole_coord_total += int(hole_vertex_mask.sum().item()) * 2

    results = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "split": args.split,
        "num_graphs": len(dataset),
        "support_accuracy": float(support_correct / max(1, support_total)),
        "vertex_count_accuracy": float(count_correct / max(1, count_total)),
        "vertex_count_mae": float(count_abs_error / max(1, count_total)),
        "coord_l1": float(coord_abs_error / max(1, coord_total)),
        "hole_count_accuracy": float(hole_count_correct / max(1, hole_count_total)),
        "hole_count_mae": float(hole_count_abs_error / max(1, hole_count_total)),
        "hole_vertex_count_accuracy": float(hole_vertex_count_correct / max(1, hole_vertex_count_total)),
        "hole_vertex_count_mae": float(hole_vertex_count_abs_error / max(1, hole_vertex_count_total)),
        "hole_coord_l1": float(hole_coord_abs_error / max(1, hole_coord_total)),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
