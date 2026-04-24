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
    parser = argparse.ArgumentParser(description="Run a forward-pass smoke test on the geometry decoder.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--geometry-root", type=Path, default=Path("data/remote_256_geometry"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=96)
    parser.add_argument("--max-neighbors", type=int, default=0, help="0 disables neighbor-cap filtering.")
    parser.add_argument("--max-vertices", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    dataset = GeometryGraphDataset(
        dual_root=args.dual_root,
        geometry_root=args.geometry_root,
        split=args.split,
        binners=binners,
        max_faces=args.max_faces,
        max_neighbors=args.max_neighbors if args.max_neighbors > 0 else None,
        max_vertices=args.max_vertices,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_geometry_graphs)
    batch = next(iter(loader))

    model = build_geometry_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=args.max_faces,
        max_neighbors=args.max_neighbors,
        max_vertices=args.max_vertices,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    outputs = model(
        node_features=batch["node_features"],
        face_mask=batch["face_mask"],
        neighbor_indices=batch["neighbor_indices"],
        neighbor_tokens=batch["neighbor_tokens"],
        neighbor_mask=batch["neighbor_mask"],
    )

    print("dataset_size", len(dataset))
    print("batch_paths", batch["paths"])
    print("node_features", tuple(batch["node_features"].shape))
    print("neighbor_indices", tuple(batch["neighbor_indices"].shape))
    print("vertices", tuple(batch["vertices"].shape))
    print("support_logits", tuple(outputs["support_logits"].shape))
    print("vertex_count_logits", tuple(outputs["vertex_count_logits"].shape))
    print("vertex_coords", tuple(outputs["vertex_coords"].shape))


if __name__ == "__main__":
    main()
