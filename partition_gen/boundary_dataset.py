from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from partition_gen.ar_dataset import QuantileBinner


BOUNDARY_FEATURE_NAMES = [
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

PAIR_FEATURE_NAMES = [
    "shared_length_ratio",
    "centroid_dx",
    "centroid_dy",
    "centroid_distance",
    "union_x0",
    "union_y0",
    "union_x1",
    "union_y1",
    "touch_left",
    "touch_top",
    "touch_right",
    "touch_bottom",
    "is_border",
]


def _line_pixels(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    size: Tuple[int, int],
) -> List[Tuple[int, int]]:
    height, width = size
    x0, y0 = point_a
    x1, y1 = point_b
    steps = max(int(np.ceil(abs(x1 - x0))), int(np.ceil(abs(y1 - y0))), 1) * 2
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    coords: List[Tuple[int, int]] = []
    for x, y in zip(xs, ys):
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        if not coords or coords[-1] != (ix, iy):
            coords.append((ix, iy))
    return coords


def _draw_segment(
    mask: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
) -> None:
    for x, y in _line_pixels(start_xy, end_xy, size=mask.shape):
        mask[y, x] = True


def _pair_key_from_faces(face_ids: Sequence[int]) -> Tuple[int, int]:
    if len(face_ids) == 1:
        return (int(face_ids[0]), -1)
    a, b = sorted(int(value) for value in face_ids[:2])
    return (a, b)


def _scale_vertex(
    point_xy: Sequence[float],
    *,
    source_size: int,
    target_size: int,
) -> Tuple[float, float]:
    scale = float(target_size) / float(source_size)
    x = float(point_xy[0]) * scale
    y = float(point_xy[1]) * scale
    return x, y


def _build_pair_target_mask(
    group: Dict[str, object],
    *,
    source_size: int,
    target_size: int,
) -> np.ndarray:
    mask = np.zeros((target_size, target_size), dtype=np.float32)
    for segment in group["segments"]:
        start_xy = _scale_vertex(segment["vertices"][0], source_size=source_size, target_size=target_size)
        end_xy = _scale_vertex(segment["vertices"][1], source_size=source_size, target_size=target_size)
        _draw_segment(mask, start_xy, end_xy)
    return mask


def _touch_flags(face: Dict[str, object]) -> np.ndarray:
    bbox = np.asarray(face["bbox_ratio"], dtype=np.float32)
    eps = 1e-6
    return np.asarray(
        [
            float(abs(float(bbox[0])) <= eps),
            float(abs(float(bbox[1])) <= eps),
            float(abs(float(bbox[2]) - 1.0) <= eps),
            float(abs(float(bbox[3]) - 1.0) <= eps),
        ],
        dtype=np.float32,
    )


def build_graph_pair_specs(dual_graph: Dict[str, object]) -> List[Dict[str, object]]:
    faces = dual_graph["faces"]
    by_id = {int(face["id"]): face for face in faces}
    specs: List[Dict[str, object]] = []
    for edge in dual_graph["edges"]:
        u = int(edge["u"])
        v = int(edge["v"])
        face_u = by_id[u]
        face_v = by_id[v]
        centroid_u = np.asarray(face_u["centroid_ratio"], dtype=np.float32)
        centroid_v = np.asarray(face_v["centroid_ratio"], dtype=np.float32)
        bbox_u = np.asarray(face_u["bbox_ratio"], dtype=np.float32)
        bbox_v = np.asarray(face_v["bbox_ratio"], dtype=np.float32)
        union_bbox = np.asarray(
            [
                min(float(bbox_u[0]), float(bbox_v[0])),
                min(float(bbox_u[1]), float(bbox_v[1])),
                max(float(bbox_u[2]), float(bbox_v[2])),
                max(float(bbox_u[3]), float(bbox_v[3])),
            ],
            dtype=np.float32,
        )
        delta = centroid_v - centroid_u
        specs.append(
            {
                "pair": (u, v),
                "u": u,
                "v": v,
                "is_border": False,
                "shared_length_ratio": float(edge["shared_length"]) / float(dual_graph["size"][0]),
                "pair_features": np.concatenate(
                    [
                        np.asarray(
                            [
                                float(edge["shared_length"]) / float(dual_graph["size"][0]),
                                float(delta[0]),
                                float(delta[1]),
                                float(np.linalg.norm(delta)),
                            ],
                            dtype=np.float32,
                        ),
                        union_bbox,
                        np.zeros((4,), dtype=np.float32),
                        np.asarray([0.0], dtype=np.float32),
                    ]
                ),
            }
        )

    for face in faces:
        if float(face["border_ratio"]) <= 0.0:
            continue
        bbox = np.asarray(face["bbox_ratio"], dtype=np.float32)
        specs.append(
            {
                "pair": (int(face["id"]), -1),
                "u": int(face["id"]),
                "v": -1,
                "is_border": True,
                "shared_length_ratio": float(face["border_ratio"]),
                "pair_features": np.concatenate(
                    [
                        np.asarray(
                            [
                                float(face["border_ratio"]),
                                0.0,
                                0.0,
                                0.0,
                            ],
                            dtype=np.float32,
                        ),
                        bbox,
                        _touch_flags(face),
                        np.asarray([1.0], dtype=np.float32),
                    ]
                ),
            }
        )

    specs.sort(key=lambda item: (item["is_border"], item["pair"][0], item["pair"][1]))
    return specs


def encode_face(face: Dict[str, object], binners: Dict[str, QuantileBinner]) -> List[int]:
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


class BoundaryGraphDataset(Dataset):
    def __init__(
        self,
        *,
        dual_root: Path | str,
        boundary_root: Path | str,
        split: str,
        binners: Dict[str, QuantileBinner],
        max_faces: int | None = None,
        max_neighbors: int | None = None,
    ) -> None:
        self.dual_root = Path(dual_root)
        self.boundary_root = Path(boundary_root)
        self.split = split
        self.binners = binners
        self.max_faces = max_faces
        self.max_neighbors = max_neighbors
        graph_paths = sorted((self.dual_root / split / "graphs").glob("*.json"))

        filtered = []
        for path in graph_paths:
            with path.open("r", encoding="utf-8") as handle:
                dual_graph = json.load(handle)
            if self.max_faces is not None and int(dual_graph["stats"]["num_faces"]) > self.max_faces:
                continue
            neighbor_cap = max((len(face["neighbors"]) for face in dual_graph["faces"]), default=0)
            if self.max_neighbors is not None and neighbor_cap > self.max_neighbors:
                continue
            filtered.append(path)
        self.graph_paths = filtered

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        dual_path = self.graph_paths[index]
        boundary_graph_path = self.boundary_root / self.split / "graphs" / dual_path.name
        boundary_mask_path = self.boundary_root / self.split / "boundary_masks" / f"{dual_path.stem}.png"

        with dual_path.open("r", encoding="utf-8") as handle:
            dual_graph = json.load(handle)
        with boundary_graph_path.open("r", encoding="utf-8") as handle:
            boundary_graph = json.load(handle)

        faces = dual_graph["faces"]
        num_faces = len(faces)
        neighbor_cap = max((len(face["neighbors"]) for face in faces), default=0)

        node_features = np.asarray([encode_face(face, self.binners) for face in faces], dtype=np.int64)
        neighbor_indices = np.full((num_faces, neighbor_cap), -1, dtype=np.int64)
        neighbor_tokens = np.zeros((num_faces, neighbor_cap), dtype=np.int64)
        neighbor_mask = np.zeros((num_faces, neighbor_cap), dtype=bool)

        size = float(dual_graph["size"][0])
        centroid_ratios = np.zeros((num_faces, 2), dtype=np.float32)
        bbox_ratios = np.zeros((num_faces, 4), dtype=np.float32)
        labels = np.zeros((num_faces,), dtype=np.int64)
        for row, face in enumerate(faces):
            centroid_ratios[row] = np.asarray(face["centroid_ratio"], dtype=np.float32)
            bbox_ratios[row] = np.asarray(face["bbox_ratio"], dtype=np.float32)
            labels[row] = int(face["label"])
            for column, neighbor in enumerate(face["neighbors"]):
                neighbor_indices[row, column] = int(neighbor["id"])
                neighbor_tokens[row, column] = self.binners["shared_length_ratio"].encode(float(neighbor["shared_length"]) / size)
                neighbor_mask[row, column] = True

        boundary_mask = (np.array(Image.open(boundary_mask_path), dtype=np.uint8) > 0).astype(np.float32)
        seed_points = np.asarray([seed["centroid_ratio"] for seed in boundary_graph["seeds"]], dtype=np.float32)

        return {
            "path": str(dual_path.as_posix()),
            "num_faces": torch.tensor(num_faces, dtype=torch.long),
            "node_features": torch.from_numpy(node_features),
            "neighbor_indices": torch.from_numpy(neighbor_indices),
            "neighbor_tokens": torch.from_numpy(neighbor_tokens),
            "neighbor_mask": torch.from_numpy(neighbor_mask),
            "centroid_ratios": torch.from_numpy(centroid_ratios),
            "bbox_ratios": torch.from_numpy(bbox_ratios),
            "labels": torch.from_numpy(labels),
            "seed_points": torch.from_numpy(seed_points),
            "boundary_mask": torch.from_numpy(boundary_mask[None, :, :]),
        }


def collate_boundary_graphs(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(batch)
    max_faces = max(int(item["num_faces"]) for item in batch)
    max_neighbors = max(item["neighbor_indices"].shape[1] for item in batch)
    feature_dim = batch[0]["node_features"].shape[1]
    max_seeds = max(item["seed_points"].shape[0] for item in batch)
    height, width = batch[0]["boundary_mask"].shape[-2:]

    node_features = torch.zeros((batch_size, max_faces, feature_dim), dtype=torch.long)
    neighbor_indices = torch.full((batch_size, max_faces, max_neighbors), -1, dtype=torch.long)
    neighbor_tokens = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.long)
    neighbor_mask = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.bool)
    centroid_ratios = torch.zeros((batch_size, max_faces, 2), dtype=torch.float32)
    bbox_ratios = torch.zeros((batch_size, max_faces, 4), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_faces), dtype=torch.long)
    seed_points = torch.zeros((batch_size, max_seeds, 2), dtype=torch.float32)
    seed_mask = torch.zeros((batch_size, max_seeds), dtype=torch.bool)
    face_mask = torch.zeros((batch_size, max_faces), dtype=torch.bool)
    num_faces = torch.zeros((batch_size,), dtype=torch.long)
    boundary_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    paths: List[str] = []

    for batch_index, item in enumerate(batch):
        face_count = int(item["num_faces"])
        neigh_count = item["neighbor_indices"].shape[1]
        seed_count = item["seed_points"].shape[0]
        node_features[batch_index, :face_count] = item["node_features"]
        neighbor_indices[batch_index, :face_count, :neigh_count] = item["neighbor_indices"]
        neighbor_tokens[batch_index, :face_count, :neigh_count] = item["neighbor_tokens"]
        neighbor_mask[batch_index, :face_count, :neigh_count] = item["neighbor_mask"]
        centroid_ratios[batch_index, :face_count] = item["centroid_ratios"]
        bbox_ratios[batch_index, :face_count] = item["bbox_ratios"]
        labels[batch_index, :face_count] = item["labels"]
        seed_points[batch_index, :seed_count] = item["seed_points"]
        seed_mask[batch_index, :seed_count] = True
        face_mask[batch_index, :face_count] = True
        num_faces[batch_index] = face_count
        boundary_mask[batch_index] = item["boundary_mask"]
        paths.append(str(item["path"]))

    return {
        "paths": paths,
        "num_faces": num_faces,
        "face_mask": face_mask,
        "node_features": node_features,
        "neighbor_indices": neighbor_indices,
        "neighbor_tokens": neighbor_tokens,
        "neighbor_mask": neighbor_mask,
        "centroid_ratios": centroid_ratios,
        "bbox_ratios": bbox_ratios,
        "labels": labels,
        "seed_points": seed_points,
        "seed_mask": seed_mask,
        "boundary_mask": boundary_mask,
    }


class PairBoundaryGraphDataset(Dataset):
    def __init__(
        self,
        *,
        dual_root: Path | str,
        boundary_root: Path | str,
        split: str,
        binners: Dict[str, QuantileBinner],
        max_faces: int | None = None,
        max_neighbors: int | None = None,
        max_pairs: int | None = None,
        target_size: int = 64,
    ) -> None:
        self.dual_root = Path(dual_root)
        self.boundary_root = Path(boundary_root)
        self.split = split
        self.binners = binners
        self.max_faces = max_faces
        self.max_neighbors = max_neighbors
        self.max_pairs = max_pairs
        self.target_size = target_size

        graph_paths = sorted((self.dual_root / split / "graphs").glob("*.json"))
        filtered: List[Path] = []
        for path in graph_paths:
            with path.open("r", encoding="utf-8") as handle:
                dual_graph = json.load(handle)
            if self.max_faces is not None and int(dual_graph["stats"]["num_faces"]) > self.max_faces:
                continue
            neighbor_cap = max((len(face["neighbors"]) for face in dual_graph["faces"]), default=0)
            if self.max_neighbors is not None and neighbor_cap > self.max_neighbors:
                continue
            if self.max_pairs is not None:
                pair_specs = build_graph_pair_specs(dual_graph)
                if len(pair_specs) > self.max_pairs:
                    continue
            filtered.append(path)
        self.graph_paths = filtered

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        dual_path = self.graph_paths[index]
        boundary_graph_path = self.boundary_root / self.split / "graphs" / dual_path.name

        with dual_path.open("r", encoding="utf-8") as handle:
            dual_graph = json.load(handle)
        with boundary_graph_path.open("r", encoding="utf-8") as handle:
            boundary_graph = json.load(handle)

        faces = dual_graph["faces"]
        num_faces = len(faces)
        neighbor_cap = max((len(face["neighbors"]) for face in faces), default=0)

        node_features = np.asarray([encode_face(face, self.binners) for face in faces], dtype=np.int64)
        neighbor_indices = np.full((num_faces, neighbor_cap), -1, dtype=np.int64)
        neighbor_tokens = np.zeros((num_faces, neighbor_cap), dtype=np.int64)
        neighbor_mask = np.zeros((num_faces, neighbor_cap), dtype=bool)

        size = float(dual_graph["size"][0])
        centroid_ratios = np.zeros((num_faces, 2), dtype=np.float32)
        bbox_ratios = np.zeros((num_faces, 4), dtype=np.float32)
        labels = np.zeros((num_faces,), dtype=np.int64)
        for row, face in enumerate(faces):
            centroid_ratios[row] = np.asarray(face["centroid_ratio"], dtype=np.float32)
            bbox_ratios[row] = np.asarray(face["bbox_ratio"], dtype=np.float32)
            labels[row] = int(face["label"])
            for column, neighbor in enumerate(face["neighbors"]):
                neighbor_indices[row, column] = int(neighbor["id"])
                neighbor_tokens[row, column] = self.binners["shared_length_ratio"].encode(float(neighbor["shared_length"]) / size)
                neighbor_mask[row, column] = True

        pair_specs = build_graph_pair_specs(dual_graph)
        target_by_pair = {
            _pair_key_from_faces(group["faces"]): group
            for group in boundary_graph["edge_groups"]
        }
        num_pairs = len(pair_specs)
        pair_indices = np.full((num_pairs, 2), -1, dtype=np.int64)
        pair_features = np.zeros((num_pairs, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
        pair_masks = np.zeros((num_pairs, self.target_size, self.target_size), dtype=np.float32)
        pair_is_border = np.zeros((num_pairs,), dtype=bool)
        pair_valid = np.ones((num_pairs,), dtype=bool)

        for row, spec in enumerate(pair_specs):
            pair_indices[row, 0] = int(spec["u"])
            pair_indices[row, 1] = int(spec["v"])
            pair_features[row] = np.asarray(spec["pair_features"], dtype=np.float32)
            pair_is_border[row] = bool(spec["is_border"])
            group = target_by_pair.get(spec["pair"])
            if group is None:
                pair_valid[row] = False
                continue
            pair_masks[row] = _build_pair_target_mask(
                group,
                source_size=int(dual_graph["size"][0]),
                target_size=self.target_size,
            )

        union_mask = (pair_masks.max(axis=0, keepdims=True) > 0).astype(np.float32)
        seed_points = np.asarray([seed["centroid_ratio"] for seed in boundary_graph["seeds"]], dtype=np.float32)

        return {
            "path": str(dual_path.as_posix()),
            "num_faces": torch.tensor(num_faces, dtype=torch.long),
            "node_features": torch.from_numpy(node_features),
            "neighbor_indices": torch.from_numpy(neighbor_indices),
            "neighbor_tokens": torch.from_numpy(neighbor_tokens),
            "neighbor_mask": torch.from_numpy(neighbor_mask),
            "centroid_ratios": torch.from_numpy(centroid_ratios),
            "bbox_ratios": torch.from_numpy(bbox_ratios),
            "labels": torch.from_numpy(labels),
            "seed_points": torch.from_numpy(seed_points),
            "num_pairs": torch.tensor(num_pairs, dtype=torch.long),
            "pair_indices": torch.from_numpy(pair_indices),
            "pair_features": torch.from_numpy(pair_features),
            "pair_masks": torch.from_numpy(pair_masks[:, None, :, :]),
            "pair_is_border": torch.from_numpy(pair_is_border),
            "pair_valid": torch.from_numpy(pair_valid),
            "union_mask": torch.from_numpy(union_mask),
        }


def collate_pair_boundary_graphs(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(batch)
    max_faces = max(int(item["num_faces"]) for item in batch)
    max_neighbors = max(item["neighbor_indices"].shape[1] for item in batch)
    feature_dim = batch[0]["node_features"].shape[1]
    max_seeds = max(item["seed_points"].shape[0] for item in batch)
    max_pairs = max(int(item["num_pairs"]) for item in batch)
    _, height, width = batch[0]["union_mask"].shape

    node_features = torch.zeros((batch_size, max_faces, feature_dim), dtype=torch.long)
    neighbor_indices = torch.full((batch_size, max_faces, max_neighbors), -1, dtype=torch.long)
    neighbor_tokens = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.long)
    neighbor_mask = torch.zeros((batch_size, max_faces, max_neighbors), dtype=torch.bool)
    centroid_ratios = torch.zeros((batch_size, max_faces, 2), dtype=torch.float32)
    bbox_ratios = torch.zeros((batch_size, max_faces, 4), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_faces), dtype=torch.long)
    seed_points = torch.zeros((batch_size, max_seeds, 2), dtype=torch.float32)
    seed_mask = torch.zeros((batch_size, max_seeds), dtype=torch.bool)
    face_mask = torch.zeros((batch_size, max_faces), dtype=torch.bool)
    num_faces = torch.zeros((batch_size,), dtype=torch.long)

    num_pairs = torch.zeros((batch_size,), dtype=torch.long)
    pair_indices = torch.full((batch_size, max_pairs, 2), -1, dtype=torch.long)
    pair_features = torch.zeros((batch_size, max_pairs, len(PAIR_FEATURE_NAMES)), dtype=torch.float32)
    pair_masks = torch.zeros((batch_size, max_pairs, 1, height, width), dtype=torch.float32)
    pair_is_border = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
    pair_valid = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
    union_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    paths: List[str] = []

    for batch_index, item in enumerate(batch):
        face_count = int(item["num_faces"])
        neigh_count = item["neighbor_indices"].shape[1]
        seed_count = item["seed_points"].shape[0]
        pair_count = int(item["num_pairs"])

        node_features[batch_index, :face_count] = item["node_features"]
        neighbor_indices[batch_index, :face_count, :neigh_count] = item["neighbor_indices"]
        neighbor_tokens[batch_index, :face_count, :neigh_count] = item["neighbor_tokens"]
        neighbor_mask[batch_index, :face_count, :neigh_count] = item["neighbor_mask"]
        centroid_ratios[batch_index, :face_count] = item["centroid_ratios"]
        bbox_ratios[batch_index, :face_count] = item["bbox_ratios"]
        labels[batch_index, :face_count] = item["labels"]
        seed_points[batch_index, :seed_count] = item["seed_points"]
        seed_mask[batch_index, :seed_count] = True
        face_mask[batch_index, :face_count] = True
        num_faces[batch_index] = face_count

        num_pairs[batch_index] = pair_count
        pair_indices[batch_index, :pair_count] = item["pair_indices"]
        pair_features[batch_index, :pair_count] = item["pair_features"]
        pair_masks[batch_index, :pair_count] = item["pair_masks"]
        pair_is_border[batch_index, :pair_count] = item["pair_is_border"]
        pair_valid[batch_index, :pair_count] = item["pair_valid"]
        union_mask[batch_index] = item["union_mask"]
        paths.append(str(item["path"]))

    return {
        "paths": paths,
        "num_faces": num_faces,
        "face_mask": face_mask,
        "node_features": node_features,
        "neighbor_indices": neighbor_indices,
        "neighbor_tokens": neighbor_tokens,
        "neighbor_mask": neighbor_mask,
        "centroid_ratios": centroid_ratios,
        "bbox_ratios": bbox_ratios,
        "labels": labels,
        "seed_points": seed_points,
        "seed_mask": seed_mask,
        "num_pairs": num_pairs,
        "pair_indices": pair_indices,
        "pair_features": pair_features,
        "pair_masks": pair_masks,
        "pair_is_border": pair_is_border,
        "pair_valid": pair_valid,
        "union_mask": union_mask,
    }
