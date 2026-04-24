from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.geometry_dataset import GeometryGraphDataset, collate_geometry_graphs
from partition_gen.models.geometry_decoder import build_geometry_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict simplified face polygons from dual graphs.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--geometry-root", type=Path, default=Path("data/remote_256_geometry"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_predictions"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    subset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_geometry_graphs)

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

    summaries = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                node_features=batch["node_features"].to(device),
                face_mask=batch["face_mask"].to(device),
                neighbor_indices=batch["neighbor_indices"].to(device),
                neighbor_tokens=batch["neighbor_tokens"].to(device),
                neighbor_mask=batch["neighbor_mask"].to(device),
            )

            support_pred = torch.argmax(outputs["support_logits"], dim=-1).cpu()
            count_pred = torch.argmax(outputs["vertex_count_logits"], dim=-1).cpu()
            coords_pred = outputs["vertex_coords"].cpu()
            hole_count_pred = torch.argmax(outputs["hole_count_logits"], dim=-1).cpu() if outputs["hole_count_logits"].shape[-1] > 0 else None
            hole_vertex_count_pred = (
                torch.argmax(outputs["hole_vertex_count_logits"], dim=-1).cpu()
                if outputs["hole_vertex_count_logits"].numel() > 0
                else None
            )
            hole_coords_pred = outputs["hole_vertex_coords"].cpu() if outputs["hole_vertex_coords"].numel() > 0 else None

            for row, path in enumerate(batch["paths"]):
                num_faces = int(batch["num_faces"][row])
                faces = []
                supported = 0
                for face_id in range(num_faces):
                    face_support = int(support_pred[row, face_id].item())
                    vertex_count = int(count_pred[row, face_id].item())
                    vertex_count = min(vertex_count, coords_pred.shape[2])
                    vertices_local = coords_pred[row, face_id, :vertex_count].tolist()
                    hole_count = int(hole_count_pred[row, face_id].item()) if hole_count_pred is not None else 0
                    hole_vertex_counts = []
                    hole_vertices_local = []
                    if hole_vertex_count_pred is not None and hole_coords_pred is not None and hole_count > 0:
                        hole_count = min(hole_count, hole_vertex_count_pred.shape[2])
                        for hole_index in range(hole_count):
                            hv_count = int(hole_vertex_count_pred[row, face_id, hole_index].item())
                            hv_count = min(hv_count, hole_coords_pred.shape[3])
                            hole_vertex_counts.append(hv_count)
                            hole_vertices_local.append(
                                hole_coords_pred[row, face_id, hole_index, :hv_count].tolist()
                            )
                    if face_support:
                        supported += 1
                    faces.append(
                        {
                            "id": face_id,
                            "support_pred": face_support,
                            "vertex_count_pred": vertex_count,
                            "vertices_local_pred": vertices_local,
                            "hole_count_pred": hole_count,
                            "hole_vertex_counts_pred": hole_vertex_counts,
                            "hole_vertices_local_pred": hole_vertices_local,
                        }
                    )

                payload = {
                    "source_dual_graph": path,
                    "checkpoint": str(args.checkpoint.as_posix()),
                    "faces": faces,
                    "stats": {
                        "num_faces": num_faces,
                        "predicted_supported_faces": supported,
                    },
                }
                save_json(args.output_dir / "graphs" / f"{sample_index:04d}.json", payload)
                summaries.append(payload["stats"])
                sample_index += 1

    summary = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "split": args.split,
        "num_graphs": len(summaries),
        "mean_faces": float(np.mean([item["num_faces"] for item in summaries])) if summaries else 0.0,
        "mean_predicted_supported_faces": float(np.mean([item["predicted_supported_faces"] for item in summaries])) if summaries else 0.0,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
