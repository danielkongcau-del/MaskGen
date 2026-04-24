from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from partition_gen.joint_render import _nearest_open_pixel


def upsample_pair_masks(pair_masks: np.ndarray, render_size: int) -> np.ndarray:
    tensor = torch.from_numpy(pair_masks.astype(np.float32))
    up = F.interpolate(
        tensor,
        size=(render_size, render_size),
        mode="bilinear",
        align_corners=False,
    )
    return up.numpy()


def render_partition_from_pair_masks(
    *,
    dual_graph: Dict[str, object],
    pair_masks: np.ndarray,
    pair_indices: np.ndarray,
    pair_valid: np.ndarray,
    threshold: float = 0.5,
    render_size: int = 256,
    max_candidate_faces: int = 2,
    max_candidate_pairs: int = 2,
) -> Tuple[np.ndarray, Dict[str, object]]:
    num_faces = len(dual_graph["faces"])
    height = width = render_size

    upsampled = upsample_pair_masks(pair_masks, render_size=render_size)
    pair_scores = upsampled[:, 0] * pair_valid[:, None, None].astype(np.float32)
    boundary_mask = pair_scores.max(axis=0) >= threshold
    face_owner = np.full((height, width), -1, dtype=np.int32)
    distance = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[Tuple[int, int]] = deque()

    seed_records = []
    seed_collisions = 0
    for face in dual_graph["faces"]:
        face_id = int(face["id"])
        x, y = _nearest_open_pixel(boundary_mask, (float(face["centroid"][0]), float(face["centroid"][1])))
        if boundary_mask[y, x]:
            continue
        if face_owner[y, x] != -1:
            seed_collisions += 1
            continue
        face_owner[y, x] = face_id
        distance[y, x] = 0
        queue.append((x, y))
        seed_records.append({"face_id": face_id, "x": x, "y": y, "label": int(face["label"])})

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        x, y = queue.popleft()
        owner = face_owner[y, x]
        next_distance = distance[y, x] + 1
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if boundary_mask[ny, nx]:
                continue
            if next_distance < distance[ny, nx]:
                distance[ny, nx] = next_distance
                face_owner[ny, nx] = owner
                queue.append((nx, ny))

    centroids = np.asarray(
        [[float(face["centroid"][0]), float(face["centroid"][1])] for face in dual_graph["faces"]],
        dtype=np.float32,
    )
    labels = np.asarray([int(face["label"]) for face in dual_graph["faces"]], dtype=np.int16)

    boundary_pixels = np.argwhere(boundary_mask)
    ambiguous_boundary_pixels = 0
    for y, x in boundary_pixels:
        pixel_pair_scores = pair_scores[:, y, x]
        pair_candidates = np.flatnonzero(pixel_pair_scores >= threshold)
        if pair_candidates.size == 0:
            pair_candidates = np.asarray([int(pixel_pair_scores.argmax())], dtype=np.int64)
        if pair_candidates.size > 1:
            ambiguous_boundary_pixels += 1
        if max_candidate_pairs > 0 and pair_candidates.size > max_candidate_pairs:
            order = np.argsort(pixel_pair_scores[pair_candidates])
            pair_candidates = pair_candidates[order[-max_candidate_pairs:]]

        local_face_support: Dict[int, float] = {}
        for pair_index in pair_candidates:
            score = float(pixel_pair_scores[pair_index])
            u = int(pair_indices[pair_index, 0])
            v = int(pair_indices[pair_index, 1])
            if 0 <= u < num_faces:
                local_face_support[u] = max(local_face_support.get(u, 0.0), score)
            if 0 <= v < num_faces:
                local_face_support[v] = max(local_face_support.get(v, 0.0), score)

        if not local_face_support:
            continue
        candidates = np.asarray(sorted(local_face_support), dtype=np.int64)
        if max_candidate_faces > 0 and candidates.size > max_candidate_faces:
            ranked = np.argsort([local_face_support[int(face_id)] for face_id in candidates])
            candidates = candidates[ranked[-max_candidate_faces:]]

        neighbor_faces = []
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            owner = face_owner[ny, nx]
            if owner >= 0 and owner in candidates:
                neighbor_faces.append(int(owner))
        if neighbor_faces:
            values, counts = np.unique(np.asarray(neighbor_faces), return_counts=True)
            weighted = counts.astype(np.float32) + 0.25 * np.asarray(
                [local_face_support[int(value)] for value in values],
                dtype=np.float32,
            )
            face_owner[y, x] = int(values[int(weighted.argmax())])
            continue

        candidate_centroids = centroids[candidates]
        squared_dist = ((candidate_centroids - np.asarray([x + 0.5, y + 0.5], dtype=np.float32)) ** 2).sum(axis=1)
        score_rank = np.asarray([local_face_support[int(face_id)] for face_id in candidates], dtype=np.float32) - 0.01 * squared_dist
        face_owner[y, x] = int(candidates[int(score_rank.argmax())])

    missing = face_owner < 0
    if missing.any():
        missing_y, missing_x = np.nonzero(missing)
        pixel_centers = np.stack([missing_x.astype(np.float32), missing_y.astype(np.float32)], axis=1)
        squared_dist = ((pixel_centers[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        nearest = squared_dist.argmin(axis=1)
        face_owner[missing_y, missing_x] = nearest.astype(np.int32)

    label_map = labels[face_owner]
    summary = {
        "seed_count": len(seed_records),
        "seed_collisions": seed_collisions,
        "boundary_pixels": int(boundary_mask.sum()),
        "ambiguous_boundary_pixels": int(ambiguous_boundary_pixels),
        "unfilled_pixels": int((face_owner < 0).sum()),
    }
    return label_map.astype(np.uint8), {"summary": summary, "boundary_mask": boundary_mask, "seeds": seed_records}
