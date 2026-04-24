from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


NODE_FEATURE_NAMES = [
    "label",
    "area_ratio",
    "centroid_x",
    "centroid_y",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "perimeter_ratio",
    "border_ratio",
    "outer_vertices",
    "hole_count",
    "degree",
    "touches_border",
]

EDGE_FEATURE_NAMES = [
    "shared_length_ratio",
]


class QuantileBinner:
    def __init__(self, edges: Sequence[float]) -> None:
        self.edges = np.asarray(edges, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.edges.size + 1)

    def encode(self, value: float) -> int:
        return int(np.searchsorted(self.edges, value, side="right"))

    def decode(self, index: int, low: float = 0.0, high: float | None = None) -> float:
        if self.edges.size == 0:
            return float(low)
        if high is None:
            high = float(self.edges[-1] + 1.0)
        if index <= 0:
            left, right = low, float(self.edges[0])
        elif index >= len(self):
            left, right = float(self.edges[-1]), high
        elif index == len(self.edges):
            left, right = float(self.edges[-1]), high
        else:
            left = float(self.edges[index - 1])
            right = float(self.edges[index])
        return float((left + right) * 0.5)


def fit_quantile_edges(values: Sequence[float], num_bins: int) -> List[float]:
    if not values:
        return []
    if num_bins <= 1:
        return []
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    edges = np.quantile(np.asarray(values, dtype=np.float32), quantiles).astype(np.float32)
    deduped = []
    for edge in edges.tolist():
        if deduped and abs(edge - deduped[-1]) < 1e-8:
            continue
        deduped.append(float(edge))
    return deduped


def build_binners_from_graphs(
    graph_paths: Iterable[Path],
    num_bins: Dict[str, int] | None = None,
) -> Dict[str, QuantileBinner]:
    num_bins = num_bins or {
        "area_ratio": 32,
        "centroid_x": 32,
        "centroid_y": 32,
        "bbox_width_ratio": 32,
        "bbox_height_ratio": 32,
        "perimeter_ratio": 32,
        "border_ratio": 16,
        "outer_vertices": 32,
        "hole_count": 8,
        "degree": 16,
        "shared_length_ratio": 32,
    }
    values: Dict[str, List[float]] = {name: [] for name in num_bins}
    for path in graph_paths:
        with path.open("r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        size = float(graph_data["size"][0])
        for face in graph_data["faces"]:
            values["area_ratio"].append(float(face["area_ratio"]))
            values["centroid_x"].append(float(face["centroid_ratio"][0]))
            values["centroid_y"].append(float(face["centroid_ratio"][1]))
            values["bbox_width_ratio"].append(float(face["bbox_width_ratio"]))
            values["bbox_height_ratio"].append(float(face["bbox_height_ratio"]))
            values["perimeter_ratio"].append(float(face["perimeter_ratio"]))
            values["border_ratio"].append(float(face["border_ratio"]))
            values["outer_vertices"].append(float(face["outer_vertices"]))
            values["hole_count"].append(float(face["hole_count"]))
            values["degree"].append(float(face["degree"]))
            for neighbor in face["prev_neighbors"]:
                values["shared_length_ratio"].append(float(neighbor["shared_length"]) / size)
    return {
        name: QuantileBinner(fit_quantile_edges(values[name], bins))
        for name, bins in num_bins.items()
    }


def save_binner_meta(path: Path, binners: Dict[str, QuantileBinner]) -> None:
    payload = {name: binner.edges.tolist() for name, binner in binners.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def load_binner_meta(path: Path) -> Dict[str, QuantileBinner]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {name: QuantileBinner(edges) for name, edges in payload.items()}


class SparseARDualGraphDataset(Dataset):
    def __init__(
        self,
        graph_root: Path | str,
        split: str,
        max_faces: int | None = None,
        max_prev_neighbors: int | None = None,
        binners: Dict[str, QuantileBinner] | None = None,
    ) -> None:
        self.graph_root = Path(graph_root)
        self.split = split
        self.max_faces = max_faces
        self.max_prev_neighbors = max_prev_neighbors
        self.graph_paths = sorted((self.graph_root / split / "graphs").glob("*.json"))
        self.binners = binners or {}

        if self.max_faces is not None or self.max_prev_neighbors is not None:
            filtered_paths = []
            for path in self.graph_paths:
                with path.open("r", encoding="utf-8") as handle:
                    graph_data = json.load(handle)
                if self.max_faces is not None and graph_data["stats"]["num_faces"] > self.max_faces:
                    continue
                if (
                    self.max_prev_neighbors is not None
                    and graph_data["stats"]["max_prev_neighbors"] > self.max_prev_neighbors
                ):
                    continue
                filtered_paths.append(path)
            self.graph_paths = filtered_paths

    def __len__(self) -> int:
        return len(self.graph_paths)

    def _encode_face(self, face: Dict[str, object]) -> List[int]:
        feature_values = [
            int(face["label"]),
            self.binners["area_ratio"].encode(float(face["area_ratio"])),
            self.binners["centroid_x"].encode(float(face["centroid_ratio"][0])),
            self.binners["centroid_y"].encode(float(face["centroid_ratio"][1])),
            self.binners["bbox_width_ratio"].encode(float(face["bbox_width_ratio"])),
            self.binners["bbox_height_ratio"].encode(float(face["bbox_height_ratio"])),
            self.binners["perimeter_ratio"].encode(float(face["perimeter_ratio"])),
            self.binners["border_ratio"].encode(float(face["border_ratio"])),
            self.binners["outer_vertices"].encode(float(face["outer_vertices"])),
            self.binners["hole_count"].encode(float(face["hole_count"])),
            self.binners["degree"].encode(float(face["degree"])),
            int(bool(face["touches_border"])),
        ]
        return feature_values

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.graph_paths[index]
        with path.open("r", encoding="utf-8") as handle:
            graph_data = json.load(handle)

        faces = graph_data["faces"]
        num_faces = len(faces)
        max_prev_neighbors = max((len(face["prev_neighbors"]) for face in faces), default=0)
        node_features = np.asarray([self._encode_face(face) for face in faces], dtype=np.int64)

        prev_indices = np.full((num_faces, max_prev_neighbors), -1, dtype=np.int64)
        prev_shared = np.zeros((num_faces, max_prev_neighbors), dtype=np.int64)
        prev_mask = np.zeros((num_faces, max_prev_neighbors), dtype=bool)
        size = float(graph_data["size"][0])

        for row, face in enumerate(faces):
            for column, neighbor in enumerate(face["prev_neighbors"]):
                prev_indices[row, column] = int(neighbor["id"])
                prev_shared[row, column] = self.binners["shared_length_ratio"].encode(
                    float(neighbor["shared_length"]) / size
                )
                prev_mask[row, column] = True

        return {
            "path": str(path.as_posix()),
            "num_faces": torch.tensor(num_faces, dtype=torch.long),
            "node_features": torch.from_numpy(node_features),
            "prev_neighbor_indices": torch.from_numpy(prev_indices),
            "prev_neighbor_tokens": torch.from_numpy(prev_shared),
            "prev_neighbor_mask": torch.from_numpy(prev_mask),
        }


def collate_sparse_ar(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    max_faces = max(int(item["num_faces"]) for item in batch)
    max_prev_neighbors = max(item["prev_neighbor_indices"].shape[1] for item in batch)
    feature_dim = batch[0]["node_features"].shape[1]

    node_features = torch.zeros((len(batch), max_faces, feature_dim), dtype=torch.long)
    prev_indices = torch.full((len(batch), max_faces, max_prev_neighbors), -1, dtype=torch.long)
    prev_tokens = torch.zeros((len(batch), max_faces, max_prev_neighbors), dtype=torch.long)
    prev_mask = torch.zeros((len(batch), max_faces, max_prev_neighbors), dtype=torch.bool)
    face_mask = torch.zeros((len(batch), max_faces), dtype=torch.bool)
    num_faces = torch.zeros((len(batch),), dtype=torch.long)
    paths: List[str] = []

    for batch_index, item in enumerate(batch):
        length = int(item["num_faces"])
        width = item["prev_neighbor_indices"].shape[1]
        node_features[batch_index, :length] = item["node_features"]
        prev_indices[batch_index, :length, :width] = item["prev_neighbor_indices"]
        prev_tokens[batch_index, :length, :width] = item["prev_neighbor_tokens"]
        prev_mask[batch_index, :length, :width] = item["prev_neighbor_mask"]
        face_mask[batch_index, :length] = True
        num_faces[batch_index] = length
        paths.append(str(item["path"]))

    return {
        "paths": paths,
        "num_faces": num_faces,
        "face_mask": face_mask,
        "node_features": node_features,
        "prev_neighbor_indices": prev_indices,
        "prev_neighbor_tokens": prev_tokens,
        "prev_neighbor_mask": prev_mask,
    }
