from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from partition_gen.ar_dataset import QuantileBinner


GEOMETRY_FEATURE_NAMES = [
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


def _encode_face(face: Dict[str, object], binners: Dict[str, QuantileBinner]) -> List[int]:
    return [
        int(face["label"]),
        binners["area_ratio"].encode(float(face["area_ratio"])),
        binners["centroid_x"].encode(float(face["centroid_ratio"][0])),
        binners["centroid_y"].encode(float(face["centroid_ratio"][1])),
        binners["bbox_width_ratio"].encode(float(face["bbox_width_ratio"])),
        binners["bbox_height_ratio"].encode(float(face["bbox_height_ratio"])),
        binners["perimeter_ratio"].encode(float(face["perimeter_ratio"])),
        binners["border_ratio"].encode(float(face["border_ratio"])),
        binners["outer_vertices"].encode(float(face["outer_vertices"])),
        binners["hole_count"].encode(float(face["hole_count"])),
        binners["degree"].encode(float(face["degree"])),
        int(bool(face["touches_border"])),
    ]


class GeometryGraphDataset(Dataset):
    def __init__(
        self,
        *,
        dual_root: Path | str,
        geometry_root: Path | str,
        split: str,
        binners: Dict[str, QuantileBinner],
        max_faces: int | None = None,
        max_neighbors: int | None = None,
        max_vertices: int = 32,
        max_holes: int = 0,
        max_hole_vertices: int = 0,
    ) -> None:
        self.dual_root = Path(dual_root)
        self.geometry_root = Path(geometry_root)
        self.split = split
        self.binners = binners
        self.max_faces = max_faces
        self.max_neighbors = max_neighbors
        self.max_vertices = max_vertices
        self.max_holes = max_holes
        self.max_hole_vertices = max_hole_vertices
        self.graph_paths = sorted((self.dual_root / split / "graphs").glob("*.json"))

        filtered = []
        for path in self.graph_paths:
            with path.open("r", encoding="utf-8") as handle:
                dual_graph = json.load(handle)
            geometry_path = self.geometry_root / split / "graphs" / path.name
            with geometry_path.open("r", encoding="utf-8") as handle:
                geometry_graph = json.load(handle)

            if self.max_faces is not None and dual_graph["stats"]["num_faces"] > self.max_faces:
                continue
            neighbor_cap = max((len(face["neighbors"]) for face in dual_graph["faces"]), default=0)
            if self.max_neighbors is not None and neighbor_cap > self.max_neighbors:
                continue
            supported_vertex_cap = max(
                (
                    int(face["vertex_count"])
                    for face in geometry_graph["faces"]
                    if bool(face["supported"])
                ),
                default=0,
            )
            if supported_vertex_cap > self.max_vertices:
                continue
            if self.max_holes > 0:
                supported_hole_cap = max(
                    (
                        int(face.get("hole_count", 0))
                        for face in geometry_graph["faces"]
                        if bool(face["supported"])
                    ),
                    default=0,
                )
                supported_hole_vertex_cap = max(
                    (
                        int(count)
                        for face in geometry_graph["faces"]
                        if bool(face["supported"])
                        for count in face.get("hole_vertex_counts", [])
                    ),
                    default=0,
                )
                if supported_hole_cap > self.max_holes:
                    continue
                if self.max_hole_vertices > 0 and supported_hole_vertex_cap > self.max_hole_vertices:
                    continue
            filtered.append(path)
        self.graph_paths = filtered

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        dual_path = self.graph_paths[index]
        geometry_path = self.geometry_root / self.split / "graphs" / dual_path.name
        with dual_path.open("r", encoding="utf-8") as handle:
            dual_graph = json.load(handle)
        with geometry_path.open("r", encoding="utf-8") as handle:
            geometry_graph = json.load(handle)

        faces = dual_graph["faces"]
        geometry_faces = {int(face["id"]): face for face in geometry_graph["faces"]}
        num_faces = len(faces)
        neighbor_cap = max((len(face["neighbors"]) for face in faces), default=0)

        node_features = np.asarray([_encode_face(face, self.binners) for face in faces], dtype=np.int64)
        neighbor_indices = np.full((num_faces, neighbor_cap), -1, dtype=np.int64)
        neighbor_tokens = np.zeros((num_faces, neighbor_cap), dtype=np.int64)
        neighbor_mask = np.zeros((num_faces, neighbor_cap), dtype=bool)

        size = float(dual_graph["size"][0])
        for row, face in enumerate(faces):
            for column, neighbor in enumerate(face["neighbors"]):
                neighbor_indices[row, column] = int(neighbor["id"])
                neighbor_tokens[row, column] = self.binners["shared_length_ratio"].encode(float(neighbor["shared_length"]) / size)
                neighbor_mask[row, column] = True

        support = np.zeros((num_faces,), dtype=np.int64)
        vertex_counts = np.zeros((num_faces,), dtype=np.int64)
        vertices = np.zeros((num_faces, self.max_vertices, 2), dtype=np.float32)
        vertex_mask = np.zeros((num_faces, self.max_vertices), dtype=bool)
        hole_counts = np.zeros((num_faces,), dtype=np.int64)
        hole_vertex_counts = np.zeros((num_faces, self.max_holes), dtype=np.int64)
        hole_vertices = np.zeros((num_faces, self.max_holes, self.max_hole_vertices, 2), dtype=np.float32)
        hole_mask = np.zeros((num_faces, self.max_holes), dtype=bool)
        hole_vertex_mask = np.zeros((num_faces, self.max_holes, self.max_hole_vertices), dtype=bool)

        for row, face in enumerate(faces):
            geometry = geometry_faces[int(face["id"])]
            if not bool(geometry["supported"]):
                continue
            coords = geometry["vertices_local"]
            count = min(len(coords), self.max_vertices)
            support[row] = 1
            vertex_counts[row] = count
            vertices[row, :count] = np.asarray(coords[:count], dtype=np.float32)
            vertex_mask[row, :count] = True
            if self.max_holes > 0:
                hole_count = min(int(geometry.get("hole_count", 0)), self.max_holes)
                hole_counts[row] = hole_count
                for hole_index in range(hole_count):
                    hole_coords = geometry.get("hole_vertices_local", [])[hole_index]
                    hole_count_i = min(int(geometry.get("hole_vertex_counts", [])[hole_index]), self.max_hole_vertices)
                    hole_vertex_counts[row, hole_index] = hole_count_i
                    hole_vertices[row, hole_index, :hole_count_i] = np.asarray(hole_coords[:hole_count_i], dtype=np.float32)
                    hole_mask[row, hole_index] = hole_count_i > 0
                    hole_vertex_mask[row, hole_index, :hole_count_i] = True

        return {
            "path": str(dual_path.as_posix()),
            "num_faces": torch.tensor(num_faces, dtype=torch.long),
            "node_features": torch.from_numpy(node_features),
            "neighbor_indices": torch.from_numpy(neighbor_indices),
            "neighbor_tokens": torch.from_numpy(neighbor_tokens),
            "neighbor_mask": torch.from_numpy(neighbor_mask),
            "geometry_support": torch.from_numpy(support),
            "vertex_counts": torch.from_numpy(vertex_counts),
            "vertices": torch.from_numpy(vertices),
            "vertex_mask": torch.from_numpy(vertex_mask),
            "hole_counts": torch.from_numpy(hole_counts),
            "hole_vertex_counts": torch.from_numpy(hole_vertex_counts),
            "hole_vertices": torch.from_numpy(hole_vertices),
            "hole_mask": torch.from_numpy(hole_mask),
            "hole_vertex_mask": torch.from_numpy(hole_vertex_mask),
        }


def collate_geometry_graphs(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(batch)
    max_faces = max(int(item["num_faces"]) for item in batch)
    max_neighbors = max(item["neighbor_indices"].shape[1] for item in batch)
    max_vertices = batch[0]["vertices"].shape[1]
    feature_dim = batch[0]["node_features"].shape[1]

    node_features = torch.zeros((batch_size, max_faces, feature_dim), dtype=torch.long)
    neighbor_indices = torch.full((batch_size, max_faces, max_neighbors), -1, dtype=torch.long)
    neighbor_tokens = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.long)
    neighbor_mask = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.bool)
    geometry_support = torch.zeros((batch_size, max_faces), dtype=torch.long)
    vertex_counts = torch.zeros((batch_size, max_faces), dtype=torch.long)
    vertices = torch.zeros((batch_size, max_faces, max_vertices, 2), dtype=torch.float32)
    vertex_mask = torch.zeros((batch_size, max_faces, max_vertices), dtype=torch.bool)
    max_holes = batch[0]["hole_vertex_counts"].shape[1]
    max_hole_vertices = batch[0]["hole_vertices"].shape[2]
    hole_counts = torch.zeros((batch_size, max_faces), dtype=torch.long)
    hole_vertex_counts = torch.zeros((batch_size, max_faces, max_holes), dtype=torch.long)
    hole_vertices = torch.zeros((batch_size, max_faces, max_holes, max_hole_vertices, 2), dtype=torch.float32)
    hole_mask = torch.zeros((batch_size, max_faces, max_holes), dtype=torch.bool)
    hole_vertex_mask = torch.zeros((batch_size, max_faces, max_holes, max_hole_vertices), dtype=torch.bool)
    face_mask = torch.zeros((batch_size, max_faces), dtype=torch.bool)
    num_faces = torch.zeros((batch_size,), dtype=torch.long)
    paths: List[str] = []

    for batch_index, item in enumerate(batch):
        face_count = int(item["num_faces"])
        neigh_count = item["neighbor_indices"].shape[1]
        node_features[batch_index, :face_count] = item["node_features"]
        neighbor_indices[batch_index, :face_count, :neigh_count] = item["neighbor_indices"]
        neighbor_tokens[batch_index, :face_count, :neigh_count] = item["neighbor_tokens"]
        neighbor_mask[batch_index, :face_count, :neigh_count] = item["neighbor_mask"]
        geometry_support[batch_index, :face_count] = item["geometry_support"]
        vertex_counts[batch_index, :face_count] = item["vertex_counts"]
        vertices[batch_index, :face_count] = item["vertices"]
        vertex_mask[batch_index, :face_count] = item["vertex_mask"]
        hole_counts[batch_index, :face_count] = item["hole_counts"]
        hole_vertex_counts[batch_index, :face_count] = item["hole_vertex_counts"]
        hole_vertices[batch_index, :face_count] = item["hole_vertices"]
        hole_mask[batch_index, :face_count] = item["hole_mask"]
        hole_vertex_mask[batch_index, :face_count] = item["hole_vertex_mask"]
        face_mask[batch_index, :face_count] = True
        num_faces[batch_index] = face_count
        paths.append(str(item["path"]))

    return {
        "paths": paths,
        "num_faces": num_faces,
        "face_mask": face_mask,
        "node_features": node_features,
        "neighbor_indices": neighbor_indices,
        "neighbor_tokens": neighbor_tokens,
        "neighbor_mask": neighbor_mask,
        "geometry_support": geometry_support,
        "vertex_counts": vertex_counts,
        "vertices": vertices,
        "vertex_mask": vertex_mask,
        "hole_counts": hole_counts,
        "hole_vertex_counts": hole_vertex_counts,
        "hole_vertices": hole_vertices,
        "hole_mask": hole_mask,
        "hole_vertex_mask": hole_vertex_mask,
    }
