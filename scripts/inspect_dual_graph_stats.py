from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import build_binners_from_graphs, save_binner_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dual-graph statistics and fit sparse-AR quantizers.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/remote_256_dual"),
        help="Root containing split/graphs JSON files.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Split used to fit quantile bin edges.",
    )
    return parser.parse_args()


def percentile_dict(values: List[int | float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {}
    return {
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def collect_split_stats(graph_paths: Iterable[Path]) -> Dict[str, object]:
    face_counts: List[int] = []
    edge_counts: List[int] = []
    prev_neighbor_caps: List[int] = []
    degree_values: List[int] = []
    prev_neighbor_counts: List[int] = []
    prev_shared_lengths: List[int] = []
    token_lengths: List[int] = []

    for path in graph_paths:
        with path.open("r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        face_counts.append(int(graph_data["stats"]["num_faces"]))
        edge_counts.append(int(graph_data["stats"]["num_edges"]))
        prev_neighbor_caps.append(int(graph_data["stats"]["max_prev_neighbors"]))
        token_length = 0
        for face in graph_data["faces"]:
            degree_values.append(int(face["degree"]))
            prev_neighbor_counts.append(len(face["prev_neighbors"]))
            prev_shared_lengths.extend(int(neighbor["shared_length"]) for neighbor in face["prev_neighbors"])
            token_length += 1 + len(face["prev_neighbors"])
        token_lengths.append(token_length)

    return {
        "num_samples": len(face_counts),
        "faces": percentile_dict(face_counts),
        "edges": percentile_dict(edge_counts),
        "degrees": percentile_dict(degree_values),
        "prev_neighbor_count": percentile_dict(prev_neighbor_counts),
        "max_prev_neighbors_per_graph": percentile_dict(prev_neighbor_caps),
        "prev_shared_length": percentile_dict(prev_shared_lengths),
        "ar_step_length": percentile_dict(token_lengths),
    }


def main() -> None:
    args = parse_args()
    split_stats: Dict[str, object] = {}
    for split_dir in sorted(args.input_root.glob("*/graphs")):
        split = split_dir.parent.name
        graph_paths = sorted(split_dir.glob("*.json"))
        split_stats[split] = collect_split_stats(graph_paths)

    train_graph_paths = sorted((args.input_root / args.train_split / "graphs").glob("*.json"))
    binners = build_binners_from_graphs(train_graph_paths)
    save_binner_meta(args.input_root / "meta" / "ar_binners.json", binners)

    recommendations = {
        "max_faces_v1": 96,
        "max_prev_neighbors_v1": 8,
        "notes": [
            "Most training graphs fit comfortably under 96 faces.",
            "Sparse autoregressive decoding should use previous-neighbor lists instead of dense adjacency tokens.",
        ],
    }

    payload = {
        "input_root": str(args.input_root.as_posix()),
        "train_split": args.train_split,
        "splits": split_stats,
        "recommendations": recommendations,
    }
    with (args.input_root / "meta" / "dual_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
