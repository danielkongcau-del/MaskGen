from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import NODE_FEATURE_NAMES, load_binner_meta
from partition_gen.models.topology_transformer import build_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample dual-graph topologies from a trained AR transformer.")
    parser.add_argument("--data-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topology_samples"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered
    return logits


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, greedy: bool) -> int:
    logits = filter_logits(logits, temperature=temperature, top_k=top_k)
    if greedy:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum()) <= 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_feature(name: str, token: int, binners: Dict[str, object]) -> float | int | bool:
    if name == "label":
        return int(token)
    if name == "touches_border":
        return bool(token)
    if name in {"outer_vertices", "hole_count", "degree"}:
        return int(round(binners[name].decode(token, low=0.0, high=max(1.0, float(binners[name].edges[-1] + 1.0)) if len(binners[name].edges) else 1.0)))
    return float(binners[name].decode(token, low=0.0, high=1.0))


def build_face_record(
    face_id: int,
    feature_tokens: List[int],
    prev_neighbors: List[Dict[str, int]],
    binners: Dict[str, object],
) -> Dict[str, object]:
    decoded = {
        name: decode_feature(name, token, binners)
        for name, token in zip(NODE_FEATURE_NAMES, feature_tokens)
    }
    return {
        "id": int(face_id),
        "feature_tokens": {name: int(token) for name, token in zip(NODE_FEATURE_NAMES, feature_tokens)},
        "decoded": decoded,
        "prev_neighbors": prev_neighbors,
    }


def build_edge_list(faces: List[Dict[str, object]]) -> List[Dict[str, int]]:
    edges = []
    for face in faces:
        for neighbor in face["prev_neighbors"]:
            u = int(neighbor["id"])
            v = int(face["id"])
            if u > v:
                u, v = v, u
            edges.append({"u": u, "v": v, "shared_length_token": int(neighbor["shared_length_token"])})
    edges.sort(key=lambda item: (item["u"], item["v"]))
    return edges


def sample_graph(
    *,
    model,
    max_faces: int,
    max_prev_neighbors: int,
    binners: Dict[str, object],
    device: torch.device,
    temperature: float,
    top_k: int,
    greedy: bool,
) -> Dict[str, object]:
    model.eval()
    num_features = len(NODE_FEATURE_NAMES)
    node_features = torch.zeros((1, max_faces, num_features), dtype=torch.long, device=device)
    face_mask = torch.ones((1, max_faces), dtype=torch.bool, device=device)
    faces: List[Dict[str, object]] = []

    with torch.no_grad():
        for face_index in range(max_faces):
            outputs = model(node_features=node_features, face_mask=face_mask)

            if face_index > 0:
                exists_token = sample_token(
                    outputs["face_exists_logits"][0, face_index],
                    temperature=temperature,
                    top_k=2,
                    greedy=greedy,
                )
                if exists_token == 0:
                    break

            feature_tokens: List[int] = []
            for feature_index in range(num_features):
                token = sample_token(
                    outputs["node_feature_logits"][feature_index][0, face_index],
                    temperature=temperature,
                    top_k=top_k,
                    greedy=greedy,
                )
                feature_tokens.append(token)
                node_features[0, face_index, feature_index] = token

            max_count = min(face_index, max_prev_neighbors)
            prev_count = 0
            if max_count > 0:
                prev_count = sample_token(
                    outputs["prev_count_logits"][0, face_index, : max_count + 1],
                    temperature=temperature,
                    top_k=min(top_k, max_count + 1) if top_k > 0 else 0,
                    greedy=greedy,
                )
            prev_count = min(prev_count, max_count)

            used_neighbors = set()
            prev_neighbors = []
            for slot_index in range(prev_count):
                pointer_logits = outputs["prev_neighbor_logits"][0, face_index, slot_index, :face_index].clone()
                if pointer_logits.numel() == 0:
                    break
                if used_neighbors:
                    pointer_logits[list(used_neighbors)] = float("-inf")
                if torch.isinf(pointer_logits).all():
                    break
                neighbor_id = sample_token(
                    pointer_logits,
                    temperature=temperature,
                    top_k=min(top_k, face_index) if top_k > 0 else 0,
                    greedy=greedy,
                )
                used_neighbors.add(neighbor_id)
                shared_length_token = sample_token(
                    outputs["edge_token_logits"][0, face_index, slot_index],
                    temperature=temperature,
                    top_k=top_k,
                    greedy=greedy,
                )
                prev_neighbors.append(
                    {
                        "id": int(neighbor_id),
                        "shared_length_token": int(shared_length_token),
                        "shared_length_ratio": float(binners["shared_length_ratio"].decode(shared_length_token, low=0.0, high=1.0)),
                    }
                )

            faces.append(
                build_face_record(
                    face_id=face_index,
                    feature_tokens=feature_tokens,
                    prev_neighbors=prev_neighbors,
                    binners=binners,
                )
            )

    edges = build_edge_list(faces)
    return {
        "faces": faces,
        "edges": edges,
        "stats": {
            "num_faces": len(faces),
            "num_edges": len(edges),
            "max_prev_neighbors": max((len(face["prev_neighbors"]) for face in faces), default=0),
        },
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    binner_path = args.data_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    model = build_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=int(train_args["max_faces"]),
        max_prev_neighbors=int(train_args["max_prev_neighbors"]),
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_layers=int(train_args["num_layers"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    all_graphs = []
    for sample_index in range(args.num_samples):
        graph = sample_graph(
            model=model,
            max_faces=int(train_args["max_faces"]),
            max_prev_neighbors=int(train_args["max_prev_neighbors"]),
            binners=binners,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )
        graph["sample_id"] = sample_index
        graph["checkpoint"] = str(args.checkpoint.as_posix())
        graph["sampling"] = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "greedy": bool(args.greedy),
        }
        save_json(args.output_dir / "graphs" / f"{sample_index:04d}.json", graph)
        all_graphs.append(graph["stats"])

    summary = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "greedy": bool(args.greedy),
        "mean_faces": float(np.mean([item["num_faces"] for item in all_graphs])) if all_graphs else 0.0,
        "mean_edges": float(np.mean([item["num_edges"] for item in all_graphs])) if all_graphs else 0.0,
        "max_faces": int(max((item["num_faces"] for item in all_graphs), default=0)),
        "max_edges": int(max((item["num_edges"] for item in all_graphs), default=0)),
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
