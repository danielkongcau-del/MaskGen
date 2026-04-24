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

from partition_gen.ar_dataset import SparseARDualGraphDataset, collate_sparse_ar, load_binner_meta
from partition_gen.models.topology_transformer import build_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward-pass smoke test on the sparse AR topology model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=96)
    parser.add_argument("--max-prev-neighbors", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binner_path = args.data_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    dataset = SparseARDualGraphDataset(
        graph_root=args.data_root,
        split=args.split,
        max_faces=args.max_faces,
        max_prev_neighbors=args.max_prev_neighbors,
        binners=binners,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sparse_ar)
    batch = next(iter(loader))

    model = build_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=args.max_faces,
        max_prev_neighbors=args.max_prev_neighbors,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

    outputs = model(
        node_features=batch["node_features"],
        face_mask=batch["face_mask"],
    )

    print("dataset_size", len(dataset))
    print("batch_paths", batch["paths"])
    print("node_features", tuple(batch["node_features"].shape))
    print("prev_neighbor_indices", tuple(batch["prev_neighbor_indices"].shape))
    print("hidden", tuple(outputs["hidden"].shape))
    print("node_feature_logits", [tuple(logits.shape) for logits in outputs["node_feature_logits"]])
    print("prev_count_logits", tuple(outputs["prev_count_logits"].shape))
    print("prev_neighbor_logits", tuple(outputs["prev_neighbor_logits"].shape))
    print("edge_token_logits", tuple(outputs["edge_token_logits"].shape))


if __name__ == "__main__":
    main()
