from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from partition_gen.manual_geometry_conditioning import _resolve_path, iter_jsonl, load_json
from partition_gen.manual_layout_frame import (
    _frame_error_row,
    build_layout_frame_example,
)
from partition_gen.manual_layout_retrieval import (
    build_layout_retrieval_fallbacks,
    build_layout_retrieval_library,
    layout_rows_from_split_targets,
    load_split_row,
    map_retrieved_layout_frames,
    retrieve_layout_entry,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


MAPPING_MODE_TO_ID = {
    "retrieved_exact_order": 0,
    "retrieved_role_label_order": 1,
    "retrieved_role_order": 2,
    "fallback_exact_median": 3,
    "fallback_role_label_median": 4,
    "fallback_role_median": 5,
    "fallback_global_median": 6,
    "unknown": 7,
}
ID_TO_MAPPING_MODE = {value: key for key, value in MAPPING_MODE_TO_ID.items()}
RESIDUAL_KEYS = ("delta_origin_x", "delta_origin_y", "delta_log_scale", "delta_orientation")
DEFAULT_GEOMETRY_MAX_BBOX_SIDE_RATIO = 1.5


def _frame_origin(frame: dict) -> tuple[float, float]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return float(origin[0]), float(origin[1])


def _wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_scale(value: object) -> float:
    return max(1e-6, float(value))


def _scale_bounds(config: ParseGraphTokenizerConfig) -> tuple[float, float]:
    return max(1.0, float(config.scale_min)), float(config.scale_max)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def default_geometry_max_bbox_side(config: ParseGraphTokenizerConfig) -> float:
    return float(config.position_max) * float(DEFAULT_GEOMETRY_MAX_BBOX_SIDE_RATIO)


def _numeric_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "max": float(max(floats)),
    }


def frame_residual_target(
    target_frame: dict,
    retrieved_frame: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> list[float]:
    config = config or ParseGraphTokenizerConfig()
    target_x, target_y = _frame_origin(target_frame)
    retrieved_x, retrieved_y = _frame_origin(retrieved_frame)
    target_scale = _safe_scale(target_frame.get("scale", 1.0))
    retrieved_scale = _safe_scale(retrieved_frame.get("scale", 1.0))
    return [
        float(target_x - retrieved_x) / float(config.position_max),
        float(target_y - retrieved_y) / float(config.position_max),
        float(math.log(target_scale / retrieved_scale)),
        float(_wrap_angle(float(target_frame.get("orientation", 0.0)) - float(retrieved_frame.get("orientation", 0.0))) / math.pi),
    ]


def residual_values_to_raw_scale(residual_values: Sequence[float] | torch.Tensor, retrieved_frame: dict) -> float:
    if isinstance(residual_values, torch.Tensor):
        residual_values = residual_values.detach().cpu().tolist()
    delta_log_scale = max(-6.0, min(6.0, float(residual_values[2])))
    return float(_safe_scale(retrieved_frame.get("scale", 1.0)) * math.exp(delta_log_scale))


def geometry_aware_scale_max(
    local_bbox: dict | None,
    *,
    config: ParseGraphTokenizerConfig | None = None,
    max_bbox_side: float | None = None,
) -> float:
    config = config or ParseGraphTokenizerConfig()
    scale_min, scale_max = _scale_bounds(config)
    side_limit = float(default_geometry_max_bbox_side(config) if max_bbox_side is None else max_bbox_side)
    if not local_bbox:
        return float(scale_max)
    limits = [float(scale_max)]
    width = abs(float(local_bbox.get("width", 0.0)))
    height = abs(float(local_bbox.get("height", 0.0)))
    if width > 1e-6:
        limits.append(side_limit / width)
    if height > 1e-6:
        limits.append(side_limit / height)
    return max(float(scale_min), float(min(limits)))


def clamp_frame_to_local_bbox(
    frame: dict,
    local_bbox: dict | None,
    *,
    config: ParseGraphTokenizerConfig | None = None,
    max_bbox_side: float | None = None,
    strong_clamp_ratio: float = 0.5,
) -> tuple[dict, dict]:
    config = config or ParseGraphTokenizerConfig()
    scale_min, tokenizer_scale_max = _scale_bounds(config)
    effective_scale_max = geometry_aware_scale_max(local_bbox, config=config, max_bbox_side=max_bbox_side)
    raw_scale = float(frame.get("scale", 1.0))
    final_scale = _clamp(raw_scale, scale_min, effective_scale_max)
    output = copy.deepcopy(frame)
    output["scale"] = float(final_scale)
    scale_ratio = float(final_scale / raw_scale) if abs(raw_scale) > 1e-12 else 1.0
    return output, {
        "raw_scale": float(raw_scale),
        "final_scale": float(final_scale),
        "scale_ratio": float(scale_ratio),
        "scale_min": float(scale_min),
        "tokenizer_scale_max": float(tokenizer_scale_max),
        "effective_scale_max": float(effective_scale_max),
        "max_bbox_side": float(default_geometry_max_bbox_side(config) if max_bbox_side is None else max_bbox_side),
        "scale_clamped": bool(abs(final_scale - raw_scale) > 1e-9),
        "tokenizer_scale_clamped": bool(raw_scale > tokenizer_scale_max or raw_scale < scale_min),
        "geometry_scale_clamped": bool(raw_scale <= tokenizer_scale_max and raw_scale > effective_scale_max),
        "geometry_frame_clamp_strong": bool(scale_ratio < float(strong_clamp_ratio)),
    }


def residual_values_to_frame(
    residual_values: Sequence[float] | torch.Tensor,
    retrieved_frame: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
    local_bbox: dict | None = None,
    max_bbox_side: float | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    if isinstance(residual_values, torch.Tensor):
        residual_values = residual_values.detach().cpu().tolist()
    retrieved_x, retrieved_y = _frame_origin(retrieved_frame)
    delta_x = max(-2.0, min(2.0, float(residual_values[0]))) * float(config.position_max)
    delta_y = max(-2.0, min(2.0, float(residual_values[1]))) * float(config.position_max)
    raw_scale = residual_values_to_raw_scale(residual_values, retrieved_frame)
    delta_orientation = max(-2.0, min(2.0, float(residual_values[3]))) * math.pi
    raw_frame = {
        "origin": [
            float(retrieved_x + delta_x),
            float(retrieved_y + delta_y),
        ],
        "scale": float(raw_scale),
        "orientation": _wrap_angle(float(retrieved_frame.get("orientation", 0.0)) + delta_orientation),
    }
    frame, _diagnostics = clamp_frame_to_local_bbox(
        raw_frame,
        local_bbox,
        config=config,
        max_bbox_side=max_bbox_side,
    )
    return frame


def geometry_renderable_local_points(geometry_target: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def add_points(values: object) -> None:
        for point in values or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

    geometry = geometry_target.get("geometry", {}) or {}
    polygons = geometry.get("polygons_local") or [
        {"outer_local": geometry.get("outer_local", []), "holes_local": geometry.get("holes_local", [])}
    ]
    for polygon in polygons:
        add_points((polygon or {}).get("outer_local", []))
        for hole in (polygon or {}).get("holes_local", []) or []:
            add_points(hole)
    for atom in geometry_target.get("atoms", []) or []:
        add_points((atom or {}).get("outer_local", []))
    return points


def geometry_local_bbox(geometry_target: dict) -> dict:
    points = geometry_renderable_local_points(geometry_target)
    if not points:
        return {
            "min_x": -0.5,
            "min_y": -0.5,
            "max_x": 0.5,
            "max_y": 0.5,
            "width": 1.0,
            "height": 1.0,
            "point_count": 0,
            "has_points": False,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = float(max(xs) - min(xs))
    height = float(max(ys) - min(ys))
    return {
        "min_x": float(min(xs)),
        "min_y": float(min(ys)),
        "max_x": float(max(xs)),
        "max_y": float(max(ys)),
        "width": width,
        "height": height,
        "point_count": int(len(points)),
        "has_points": True,
    }


def scaled_bbox_metrics(local_bbox: dict | None, scale: float) -> dict:
    if not local_bbox:
        local_bbox = {"width": 1.0, "height": 1.0}
    width = abs(float(local_bbox.get("width", 1.0))) * float(scale)
    height = abs(float(local_bbox.get("height", 1.0))) * float(scale)
    return {"width": float(width), "height": float(height), "area": float(width * height)}


def retrieved_frame_features(
    retrieved_frame: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> list[float]:
    config = config or ParseGraphTokenizerConfig()
    origin_x, origin_y = _frame_origin(retrieved_frame)
    scale = _safe_scale(retrieved_frame.get("scale", 1.0))
    orientation = float(retrieved_frame.get("orientation", 0.0))
    return [
        float(origin_x) / float(config.position_max),
        float(origin_y) / float(config.position_max),
        float(scale) / float(config.scale_max),
        float(math.log(scale) / math.log(max(2.0, float(config.scale_max)))),
        float(math.sin(orientation)),
        float(math.cos(orientation)),
    ]


def build_layout_residual_example(
    topology_target: dict,
    *,
    node_index: int,
    retrieved_frame: dict,
    target_frame: dict | None = None,
    local_bbox: dict | None = None,
    retrieval_score: float = 0.0,
    mapping_mode: str = "unknown",
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    base = build_layout_frame_example(
        topology_target,
        node_index=int(node_index),
        frame=target_frame,
        config=config,
    )
    example = {
        **base,
        "mapping_mode": str(mapping_mode),
        "mapping_mode_id": int(MAPPING_MODE_TO_ID.get(str(mapping_mode), MAPPING_MODE_TO_ID["unknown"])),
        "retrieval_score": float(retrieval_score),
        "retrieval_score_value": [min(1.0, max(0.0, float(retrieval_score) / 512.0))],
        "retrieved_frame": copy.deepcopy(retrieved_frame),
        "retrieved_frame_values": retrieved_frame_features(retrieved_frame, config=config),
        "local_bbox": copy.deepcopy(local_bbox) if local_bbox is not None else None,
    }
    if target_frame is not None:
        example["target_frame"] = copy.deepcopy(target_frame)
        example["target_residual"] = frame_residual_target(target_frame, retrieved_frame, config=config)
    return example


def iter_layout_residual_examples_from_split(
    split_root: Path,
    *,
    library_entries: Sequence[dict],
    fallback_frames: dict | None = None,
    config: ParseGraphTokenizerConfig | None = None,
    max_samples: int | None = None,
    exclude_same_stem: bool = False,
) -> Iterable[dict]:
    split_root = Path(split_root)
    config = config or ParseGraphTokenizerConfig()
    fallback_frames = fallback_frames or build_layout_retrieval_fallbacks(library_entries)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    for row in rows:
        topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        retrieved_entry, retrieval_score = retrieve_layout_entry(
            topology_target,
            library_entries,
            exclude_stem=row.get("stem") if bool(exclude_same_stem) else None,
        )
        frame_by_index, mapping_diagnostics = map_retrieved_layout_frames(
            topology_target,
            retrieved_entry,
            fallback_frames=fallback_frames,
        )
        node_mapping_modes = {
            int(key): str(value)
            for key, value in (mapping_diagnostics.get("node_mapping_modes", {}) or {}).items()
        }
        geometry_by_source_id = {
            str(target.get("source_node_id")): target
            for target in geometry_targets
            if target.get("source_node_id") is not None
        }
        for layout_row in layout_rows_from_split_targets(topology_target, geometry_targets):
            node_index = int(layout_row["node_index"])
            retrieved_frame = frame_by_index.get(node_index)
            if retrieved_frame is None:
                continue
            geometry_target = geometry_by_source_id.get(str(layout_row.get("geometry_ref")))
            yield {
                "source_topology": str(topology_path.as_posix()),
                "source_node_id": str(layout_row.get("node_id", "")),
                "stem": row.get("stem"),
                "retrieved_stem": retrieved_entry.get("stem"),
                "retrieved_library_index": int(retrieved_entry.get("library_index", -1)),
                **build_layout_residual_example(
                    topology_target,
                    node_index=node_index,
                    retrieved_frame=retrieved_frame,
                    target_frame=layout_row["frame"],
                    local_bbox=geometry_local_bbox(geometry_target) if geometry_target is not None else None,
                    retrieval_score=float(retrieval_score),
                    mapping_mode=node_mapping_modes.get(node_index, "unknown"),
                    config=config,
                ),
            }


class ManualLayoutResidualDataset(Dataset):
    def __init__(
        self,
        split_root: str | Path,
        *,
        library_entries: Sequence[dict],
        fallback_frames: dict | None = None,
        config: ParseGraphTokenizerConfig | None = None,
        max_samples: int | None = None,
        max_examples: int | None = None,
        exclude_same_stem: bool = False,
    ) -> None:
        self.split_root = Path(split_root)
        self.config = config or ParseGraphTokenizerConfig()
        self.rows = list(
            iter_layout_residual_examples_from_split(
                self.split_root,
                library_entries=library_entries,
                fallback_frames=fallback_frames,
                config=self.config,
                max_samples=max_samples,
                exclude_same_stem=exclude_same_stem,
            )
        )
        if max_examples is not None:
            self.rows = self.rows[: int(max_examples)]
        if not self.rows:
            raise RuntimeError(f"No layout residual examples found in {self.split_root}")
        self.numeric_dim = len(self.rows[0]["numeric"])
        self.retrieved_frame_dim = len(self.rows[0]["retrieved_frame_values"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[int(index)]
        return {
            "role_id": torch.tensor(int(row["role_id"]), dtype=torch.long),
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "geometry_model_id": torch.tensor(int(row["geometry_model_id"]), dtype=torch.long),
            "mapping_mode_id": torch.tensor(int(row["mapping_mode_id"]), dtype=torch.long),
            "numeric": torch.tensor(row["numeric"], dtype=torch.float32),
            "retrieved_frame_values": torch.tensor(row["retrieved_frame_values"], dtype=torch.float32),
            "retrieval_score_value": torch.tensor(row["retrieval_score_value"], dtype=torch.float32),
            "target_residual": torch.tensor(row["target_residual"], dtype=torch.float32),
            "metadata": {
                key: copy.deepcopy(row.get(key))
                for key in (
                    "source_topology",
                    "source_node_id",
                    "node_id",
                    "node_index",
                    "role",
                    "label",
                    "geometry_model",
                    "mapping_mode",
                    "retrieval_score",
                    "retrieved_stem",
                    "retrieved_library_index",
                    "retrieved_frame",
                    "target_frame",
                    "local_bbox",
                )
            },
        }


def collate_layout_residual_examples(batch: Sequence[dict]) -> dict:
    return {
        "role_id": torch.stack([item["role_id"] for item in batch], dim=0),
        "label_id": torch.stack([item["label_id"] for item in batch], dim=0),
        "geometry_model_id": torch.stack([item["geometry_model_id"] for item in batch], dim=0),
        "mapping_mode_id": torch.stack([item["mapping_mode_id"] for item in batch], dim=0),
        "numeric": torch.stack([item["numeric"] for item in batch], dim=0),
        "retrieved_frame_values": torch.stack([item["retrieved_frame_values"] for item in batch], dim=0),
        "retrieval_score_value": torch.stack([item["retrieval_score_value"] for item in batch], dim=0),
        "target_residual": torch.stack([item["target_residual"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


@dataclass
class ManualLayoutResidualRegressorConfig:
    numeric_dim: int
    retrieved_frame_dim: int = 6
    retrieval_score_dim: int = 1
    role_count: int = 6
    label_count: int = 64
    geometry_model_count: int = 4
    mapping_mode_count: int = 8
    role_emb_dim: int = 16
    label_emb_dim: int = 16
    geometry_emb_dim: int = 8
    mapping_mode_emb_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1


class ManualLayoutResidualRegressor(nn.Module):
    def __init__(self, config: ManualLayoutResidualRegressorConfig) -> None:
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(int(config.role_count), int(config.role_emb_dim))
        self.label_embedding = nn.Embedding(int(config.label_count), int(config.label_emb_dim))
        self.geometry_embedding = nn.Embedding(int(config.geometry_model_count), int(config.geometry_emb_dim))
        self.mapping_embedding = nn.Embedding(int(config.mapping_mode_count), int(config.mapping_mode_emb_dim))
        input_dim = int(
            config.numeric_dim
            + config.retrieved_frame_dim
            + config.retrieval_score_dim
            + config.role_emb_dim
            + config.label_emb_dim
            + config.geometry_emb_dim
            + config.mapping_mode_emb_dim
        )
        layers: list[nn.Module] = []
        for layer_index in range(int(config.num_layers)):
            layers.append(nn.Linear(input_dim if layer_index == 0 else int(config.hidden_dim), int(config.hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(config.dropout)))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(int(config.hidden_dim), 4)

    def forward(self, batch: dict) -> torch.Tensor:
        role = self.role_embedding(batch["role_id"].clamp(0, self.config.role_count - 1))
        label = self.label_embedding(batch["label_id"].clamp(0, self.config.label_count - 1))
        geometry = self.geometry_embedding(
            batch["geometry_model_id"].clamp(0, self.config.geometry_model_count - 1)
        )
        mapping = self.mapping_embedding(batch["mapping_mode_id"].clamp(0, self.config.mapping_mode_count - 1))
        x = torch.cat(
            [
                role,
                label,
                geometry,
                mapping,
                batch["numeric"].float(),
                batch["retrieved_frame_values"].float(),
                batch["retrieval_score_value"].float(),
            ],
            dim=-1,
        )
        return self.head(self.backbone(x))


def layout_residual_loss(predictions: torch.Tensor, batch: dict) -> torch.Tensor:
    return F.smooth_l1_loss(predictions.float(), batch["target_residual"].float())


def move_layout_residual_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    for key in (
        "role_id",
        "label_id",
        "geometry_model_id",
        "mapping_mode_id",
        "numeric",
        "retrieved_frame_values",
        "retrieval_score_value",
        "target_residual",
    ):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def _batch_from_example(example: dict, *, device: torch.device) -> dict:
    return {
        "role_id": torch.tensor([int(example["role_id"])], dtype=torch.long, device=device),
        "label_id": torch.tensor([int(example["label_id"])], dtype=torch.long, device=device),
        "geometry_model_id": torch.tensor([int(example["geometry_model_id"])], dtype=torch.long, device=device),
        "mapping_mode_id": torch.tensor([int(example["mapping_mode_id"])], dtype=torch.long, device=device),
        "numeric": torch.tensor([example["numeric"]], dtype=torch.float32, device=device),
        "retrieved_frame_values": torch.tensor([example["retrieved_frame_values"]], dtype=torch.float32, device=device),
        "retrieval_score_value": torch.tensor([example["retrieval_score_value"]], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def predict_residual_frame(
    model: ManualLayoutResidualRegressor,
    example: dict,
    *,
    device: torch.device,
    config: ParseGraphTokenizerConfig | None = None,
) -> tuple[dict, list[float]]:
    model.eval()
    prediction = model(_batch_from_example(example, device=device))[0].detach().cpu()
    frame = residual_values_to_frame(
        prediction,
        example["retrieved_frame"],
        config=config,
        local_bbox=example.get("local_bbox"),
    )
    return frame, [float(value) for value in prediction.tolist()]


@torch.no_grad()
def evaluate_layout_residual_regressor(
    model: ManualLayoutResidualRegressor,
    loader,
    *,
    device: torch.device,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    model.eval()
    losses: list[float] = []
    residual_errors = defaultdict(list)
    baseline_errors = defaultdict(list)
    role_errors = defaultdict(lambda: defaultdict(list))
    mapping_modes = Counter()
    retrieval_scores: list[float] = []
    role_counts = Counter()
    raw_scales: list[float] = []
    predicted_scales: list[float] = []
    raw_bbox_widths: list[float] = []
    raw_bbox_heights: list[float] = []
    raw_bbox_areas: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    scale_min, tokenizer_scale_max = _scale_bounds(config)
    scale_below_min = 0
    scale_above_max = 0
    scale_above_tokenizer_max = 0
    geometry_scale_clamped = 0
    effective_scale_max_values: list[float] = []
    bbox_huge_count = 0
    raw_bbox_huge_count = 0
    max_bbox_side = default_geometry_max_bbox_side(config)
    huge_side_threshold = float(config.position_max) * 2.0
    huge_area_threshold = float(config.position_max) * float(config.position_max) * 4.0

    for batch in loader:
        moved = move_layout_residual_batch_to_device(batch, device)
        predictions = model(moved)
        losses.append(float(layout_residual_loss(predictions, moved).item()))
        predictions = predictions.detach().cpu()
        for row_index, metadata in enumerate(batch["metadata"]):
            role = str(metadata.get("role", "unknown"))
            role_counts[role] += 1
            mapping_modes[str(metadata.get("mapping_mode", "unknown"))] += 1
            retrieval_scores.append(float(metadata.get("retrieval_score", 0.0)))
            retrieved_frame = metadata["retrieved_frame"]
            target_frame = metadata["target_frame"]
            local_bbox = metadata.get("local_bbox")
            raw_scale = residual_values_to_raw_scale(predictions[row_index], retrieved_frame)
            effective_scale_max = geometry_aware_scale_max(local_bbox, config=config, max_bbox_side=max_bbox_side)
            pred_frame = residual_values_to_frame(
                predictions[row_index],
                retrieved_frame,
                config=config,
                local_bbox=local_bbox,
                max_bbox_side=max_bbox_side,
            )
            raw_bbox = scaled_bbox_metrics(local_bbox, raw_scale)
            pred_bbox = scaled_bbox_metrics(local_bbox, float(pred_frame.get("scale", 1.0)))
            raw_scales.append(float(raw_scale))
            predicted_scales.append(float(pred_frame.get("scale", 1.0)))
            effective_scale_max_values.append(float(effective_scale_max))
            raw_bbox_widths.append(float(raw_bbox["width"]))
            raw_bbox_heights.append(float(raw_bbox["height"]))
            raw_bbox_areas.append(float(raw_bbox["area"]))
            bbox_widths.append(float(pred_bbox["width"]))
            bbox_heights.append(float(pred_bbox["height"]))
            bbox_areas.append(float(pred_bbox["area"]))
            scale_below_min += int(raw_scale < scale_min)
            scale_above_max += int(raw_scale > effective_scale_max)
            scale_above_tokenizer_max += int(raw_scale > tokenizer_scale_max)
            geometry_scale_clamped += int(raw_scale <= tokenizer_scale_max and raw_scale > effective_scale_max)
            raw_bbox_huge_count += int(
                raw_bbox["width"] > huge_side_threshold
                or raw_bbox["height"] > huge_side_threshold
                or raw_bbox["area"] > huge_area_threshold
            )
            bbox_huge_count += int(
                pred_bbox["width"] > huge_side_threshold
                or pred_bbox["height"] > huge_side_threshold
                or pred_bbox["area"] > huge_area_threshold
            )
            for key, value in _frame_error_row(pred_frame, target_frame).items():
                residual_errors[key].append(float(value))
                role_errors[role][f"residual_{key}"].append(float(value))
            for key, value in _frame_error_row(retrieved_frame, target_frame).items():
                baseline_errors[key].append(float(value))
                role_errors[role][f"baseline_{key}"].append(float(value))

    def avg(values: Sequence[float]) -> float | None:
        return float(mean(values)) if values else None

    baseline_origin = avg(baseline_errors["origin_mae"])
    residual_origin = avg(residual_errors["origin_mae"])
    return {
        "loss": avg(losses),
        "example_count": int(sum(role_counts.values())),
        "baseline_origin_mae": baseline_origin,
        "baseline_scale_mae": avg(baseline_errors["scale_mae"]),
        "baseline_orientation_mae": avg(baseline_errors["orientation_mae"]),
        "residual_origin_mae": residual_origin,
        "residual_scale_mae": avg(residual_errors["scale_mae"]),
        "residual_orientation_mae": avg(residual_errors["orientation_mae"]),
        "origin_mae_delta": None if baseline_origin is None or residual_origin is None else float(residual_origin - baseline_origin),
        "origin_mae_improvement_fraction": None
        if baseline_origin in (None, 0.0) or residual_origin is None
        else float((baseline_origin - residual_origin) / baseline_origin),
        "mapping_mode_histogram": {key: int(value) for key, value in sorted(mapping_modes.items())},
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
        "scale_min": float(scale_min),
        "scale_max": float(tokenizer_scale_max),
        "geometry_max_bbox_side": float(max_bbox_side),
        "effective_scale_max_stats": _numeric_stats(effective_scale_max_values),
        "raw_predicted_scale_stats": _numeric_stats(raw_scales),
        "predicted_scale_stats": _numeric_stats(predicted_scales),
        "scale_below_min_count": int(scale_below_min),
        "scale_above_max_count": int(scale_above_max),
        "scale_above_tokenizer_max_count": int(scale_above_tokenizer_max),
        "geometry_scale_clamped_count": int(geometry_scale_clamped),
        "scale_out_of_range_count": int(scale_below_min + scale_above_max),
        "scale_clamped_count": int(scale_below_min + scale_above_max),
        "raw_bbox_width_stats": _numeric_stats(raw_bbox_widths),
        "raw_bbox_height_stats": _numeric_stats(raw_bbox_heights),
        "raw_bbox_area_stats": _numeric_stats(raw_bbox_areas),
        "bbox_width_stats": _numeric_stats(bbox_widths),
        "bbox_height_stats": _numeric_stats(bbox_heights),
        "bbox_area_stats": _numeric_stats(bbox_areas),
        "raw_bbox_huge_count": int(raw_bbox_huge_count),
        "bbox_huge_count": int(bbox_huge_count),
        "bbox_huge_threshold": {
            "side": float(huge_side_threshold),
            "area": float(huge_area_threshold),
        },
        "role_metrics": {
            role: {
                "count": int(role_counts[role]),
                "baseline_origin_mae": avg(values["baseline_origin_mae"]),
                "residual_origin_mae": avg(values["residual_origin_mae"]),
                "baseline_scale_mae": avg(values["baseline_scale_mae"]),
                "residual_scale_mae": avg(values["residual_scale_mae"]),
                "baseline_orientation_mae": avg(values["baseline_orientation_mae"]),
                "residual_orientation_mae": avg(values["residual_orientation_mae"]),
            }
            for role, values in sorted(role_errors.items())
        },
    }


def _geometry_by_source_node_id(geometry_targets: Sequence[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _attach_geometry_payload(node: dict, geometry_target: dict, *, frame: dict) -> dict:
    output = copy.deepcopy(node)
    output.pop("geometry_ref", None)
    output["geometry_model"] = copy.deepcopy(geometry_target.get("geometry_model", output.get("geometry_model")))
    output["frame"] = copy.deepcopy(frame)
    if "geometry" in geometry_target:
        output["geometry"] = copy.deepcopy(geometry_target["geometry"])
    if "atoms" in geometry_target:
        output["atoms"] = copy.deepcopy(geometry_target["atoms"])
    output["layout_frame_source"] = "retrieved_residual_layout"
    return output


def attach_retrieved_residual_layout_to_split_targets(
    split_root: Path,
    *,
    library_entries: Sequence[dict],
    fallback_frames: dict | None,
    model: ManualLayoutResidualRegressor,
    tokenizer_config: ParseGraphTokenizerConfig,
    device: torch.device,
    max_samples: int | None = None,
    exclude_same_stem: bool = False,
) -> list[dict]:
    split_root = Path(split_root)
    fallback_frames = fallback_frames or build_layout_retrieval_fallbacks(library_entries)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    targets: list[dict] = []
    for row_index, row in enumerate(rows):
        topology_path = _resolve_path(row["topology_path"], split_root=split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        geometry_by_id = _geometry_by_source_node_id(geometry_targets)
        retrieved_entry, retrieval_score = retrieve_layout_entry(
            topology_target,
            library_entries,
            exclude_stem=row.get("stem") if bool(exclude_same_stem) else None,
        )
        frame_by_index, mapping_diagnostics = map_retrieved_layout_frames(
            topology_target,
            retrieved_entry,
            fallback_frames=fallback_frames,
        )
        node_mapping_modes = {
            int(key): str(value)
            for key, value in (mapping_diagnostics.get("node_mapping_modes", {}) or {}).items()
        }
        graph = topology_target.get("parse_graph", {}) or {}
        output_nodes: list[dict] = []
        attached = 0
        missing = 0
        attach_modes: Counter[str] = Counter()
        residual_rows: list[dict] = []

        for node_index, node in enumerate(graph.get("nodes", []) or []):
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.get("geometry_ref")
            if geometry_ref:
                geometry_target = geometry_by_id.get(str(geometry_ref))
                retrieved_frame = frame_by_index.get(int(node_index))
                if geometry_target is None or retrieved_frame is None:
                    missing += 1
                    attach_modes["missing"] += 1
                    output_node.pop("geometry_ref", None)
                    output_node["retrieved_residual_layout_error"] = "missing source geometry or retrieved frame"
                else:
                    example = build_layout_residual_example(
                        topology_target,
                        node_index=int(node_index),
                        retrieved_frame=retrieved_frame,
                        local_bbox=geometry_local_bbox(geometry_target),
                        retrieval_score=float(retrieval_score),
                        mapping_mode=node_mapping_modes.get(int(node_index), "unknown"),
                        config=tokenizer_config,
                    )
                    predicted_frame, residual = predict_residual_frame(
                        model,
                        example,
                        device=device,
                        config=tokenizer_config,
                    )
                    output_node = _attach_geometry_payload(output_node, geometry_target, frame=predicted_frame)
                    output_node["retrieved_frame"] = copy.deepcopy(retrieved_frame)
                    output_node["layout_residual"] = residual
                    output_node["layout_shape_attach_mode"] = "retrieved_residual_frame_true_shape"
                    attached += 1
                    attach_modes["retrieved_residual_frame_true_shape"] += 1
                    residual_rows.append(
                        {
                            "node_index": int(node_index),
                            "node_id": str(output_node.get("id", "")),
                            "mapping_mode": node_mapping_modes.get(int(node_index), "unknown"),
                            "retrieved_frame": copy.deepcopy(retrieved_frame),
                            "predicted_frame": copy.deepcopy(predicted_frame),
                            "layout_residual": residual,
                        }
                    )
            else:
                output_node.pop("geometry_ref", None)
            output_nodes.append(output_node)

        targets.append(
            {
                "format": "maskgen_generator_target_v1",
                "target_type": "parse_graph",
                "size": copy.deepcopy(topology_target.get("size", [256, 256])),
                "parse_graph": {
                    "nodes": output_nodes,
                    "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
                    "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
                },
                "metadata": {
                    "retrieved_residual_layout": True,
                    "source_topology": str(topology_path.as_posix()),
                    "sample_index": int(row_index),
                    "retrieved_stem": retrieved_entry.get("stem"),
                    "retrieved_library_index": int(retrieved_entry.get("library_index", -1)),
                    "retrieved_topology_path": retrieved_entry.get("topology_path"),
                    "retrieval_score": float(retrieval_score),
                    "mapping_diagnostics": mapping_diagnostics,
                    "attached_geometry_count": int(attached),
                    "missing_geometry_count": int(missing),
                    "attach_modes": dict(attach_modes),
                    "residual_rows": residual_rows,
                },
            }
        )
    return targets


def summarize_retrieved_residual_layout_targets(targets: Sequence[dict]) -> dict:
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    attached = 0
    missing = 0
    for target in targets:
        metadata = target.get("metadata", {}) or {}
        attached += int(metadata.get("attached_geometry_count", 0))
        missing += int(metadata.get("missing_geometry_count", 0))
        attach_modes.update(metadata.get("attach_modes", {}) or {})
        mapping_modes.update((metadata.get("mapping_diagnostics", {}) or {}).get("mapping_mode_histogram", {}) or {})
        if metadata.get("retrieval_score") is not None:
            retrieval_scores.append(float(metadata["retrieval_score"]))
    return {
        "attached_geometry_count": int(attached),
        "missing_geometry_count": int(missing),
        "attach_modes": dict(attach_modes),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
    }


def save_layout_residual_checkpoint(
    path: Path,
    *,
    model: ManualLayoutResidualRegressor,
    optimizer: torch.optim.Optimizer | None,
    model_config: ManualLayoutResidualRegressorConfig,
    tokenizer_config: ParseGraphTokenizerConfig,
    train_config: dict,
    metrics: dict,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "maskgen_manual_layout_residual_checkpoint_v1",
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "model_config": asdict(model_config),
            "tokenizer_config": asdict(tokenizer_config),
            "train_config": train_config,
            "metrics": metrics,
            "epoch": int(epoch),
        },
        path,
    )


def load_layout_residual_checkpoint(path: Path, *, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_config = ManualLayoutResidualRegressorConfig(**checkpoint["model_config"])
    tokenizer_config = ParseGraphTokenizerConfig(**checkpoint["tokenizer_config"])
    model = ManualLayoutResidualRegressor(model_config)
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model, tokenizer_config


def build_residual_library_from_split(
    library_split_root: Path,
    *,
    max_library_samples: int | None = None,
) -> tuple[list[dict], dict, dict]:
    library_entries, library_summary = build_layout_retrieval_library(
        library_split_root,
        max_samples=max_library_samples,
    )
    fallback_frames = build_layout_retrieval_fallbacks(library_entries)
    return library_entries, library_summary, fallback_frames
