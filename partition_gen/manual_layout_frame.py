from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from partition_gen.manual_topology_placeholder_geometry import (
    GeometryPlaceholderLibrary,
    decode_topology_tokens_to_target,
    iter_jsonl,
    load_json,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_relations import divides_target, inserted_in_container
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, dequantize, quantize


ROLE_TO_ID = {
    "support_region": 0,
    "divider_region": 1,
    "insert_object": 2,
    "insert_object_group": 3,
    "residual_region": 4,
    "unknown": 5,
}
ID_TO_ROLE = {value: key for key, value in ROLE_TO_ID.items()}
GEOMETRY_MODEL_TO_ID = {"none": 0, "polygon_code": 1, "convex_atoms": 2, "unknown": 3}
ID_TO_GEOMETRY_MODEL = {value: key for key, value in GEOMETRY_MODEL_TO_ID.items()}


def _resolve_path(value: object, *, split_root: Path, manifest_parent: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    for base in (manifest_parent, split_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def _safe_norm(index: int | None, count: int) -> float:
    if index is None or count <= 1:
        return 0.0
    return float(max(0, min(int(index), int(count) - 1))) / float(count - 1)


def _count_role(nodes: Sequence[dict], role: str) -> int:
    return sum(1 for node in nodes if str(node.get("role")) == role)


def _node_index_by_id(nodes: Sequence[dict]) -> Dict[str, int]:
    return {str(node.get("id")): index for index, node in enumerate(nodes) if "id" in node}


def _relation_features(topology_target: dict) -> dict:
    graph = topology_target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    node_index = _node_index_by_id(nodes)
    features = {
        "contains_parent": Counter(),
        "contains_child": Counter(),
        "parent_group": {},
        "sibling_position": {},
        "sibling_count": {},
        "inserted_object": Counter(),
        "inserted_container": Counter(),
        "inserted_container_index": {},
        "divides_divider": Counter(),
        "divides_target": Counter(),
        "divides_target_index": {},
        "adjacent": Counter(),
        "relation_counts": Counter(),
    }

    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for relation in relations:
        relation_type = str(relation.get("type"))
        features["relation_counts"][relation_type] += 1
        if relation_type == "contains":
            parent = str(relation.get("parent", ""))
            child = str(relation.get("child", ""))
            if parent in node_index and child in node_index:
                features["contains_parent"][parent] += 1
                features["contains_child"][child] += 1
                features["parent_group"][child] = node_index[parent]
                children_by_parent[parent].append(child)
        elif relation_type == "inserted_in":
            obj = str(relation.get("object", ""))
            container = inserted_in_container(relation)
            if obj in node_index:
                features["inserted_object"][obj] += 1
            if container is not None and container in node_index:
                features["inserted_container"][container] += 1
            if obj in node_index and container is not None and container in node_index:
                features["inserted_container_index"][obj] = node_index[container]
        elif relation_type == "divides":
            divider = str(relation.get("divider", ""))
            target = divides_target(relation)
            if divider in node_index:
                features["divides_divider"][divider] += 1
            if target is not None and target in node_index:
                features["divides_target"][target] += 1
            if divider in node_index and target is not None and target in node_index:
                features["divides_target_index"][divider] = node_index[target]
        elif relation_type == "adjacent_to":
            face_ids = [str(value) for value in relation.get("faces", []) or [] if str(value) in node_index]
            for face_id in face_ids:
                features["adjacent"][face_id] += max(0, len(face_ids) - 1)

    for parent, children in children_by_parent.items():
        total = len(children)
        for position, child in enumerate(children):
            features["sibling_position"][child] = position
            features["sibling_count"][child] = total

    return features


def _numeric_features(topology_target: dict, node_index: int) -> List[float]:
    graph = topology_target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    node = nodes[int(node_index)]
    node_id = str(node.get("id", ""))
    node_count = max(1, len(nodes))
    relation_count = max(1, len(relations))
    rel = _relation_features(topology_target)

    support_count = _count_role(nodes, "support_region")
    divider_count = _count_role(nodes, "divider_region")
    insert_count = _count_role(nodes, "insert_object")
    group_count = _count_role(nodes, "insert_object_group")
    residual_count = _count_role(nodes, "residual_region")
    child_count = len(node.get("children", []) or [])
    parent_group_index = rel["parent_group"].get(node_id)
    inserted_container_index = rel["inserted_container_index"].get(node_id)
    divides_target_index = rel["divides_target_index"].get(node_id)
    sibling_count = int(rel["sibling_count"].get(node_id, 0))
    sibling_position = rel["sibling_position"].get(node_id)

    contains_parent_degree = int(rel["contains_parent"].get(node_id, 0))
    contains_child_degree = int(rel["contains_child"].get(node_id, 0))
    inserted_object_degree = int(rel["inserted_object"].get(node_id, 0))
    inserted_container_degree = int(rel["inserted_container"].get(node_id, 0))
    divides_divider_degree = int(rel["divides_divider"].get(node_id, 0))
    divides_target_degree = int(rel["divides_target"].get(node_id, 0))
    adjacent_degree = int(rel["adjacent"].get(node_id, 0))
    total_degree = (
        contains_parent_degree
        + contains_child_degree
        + inserted_object_degree
        + inserted_container_degree
        + divides_divider_degree
        + divides_target_degree
        + adjacent_degree
    )

    relation_counts = rel["relation_counts"]
    return [
        _safe_norm(int(node_index), node_count),
        float(node_count) / 128.0,
        float(relation_count) / 128.0,
        float(int(node.get("label", 0))) / 32.0,
        float(support_count) / float(node_count),
        float(divider_count) / float(node_count),
        float(insert_count) / float(node_count),
        float(group_count) / float(node_count),
        float(residual_count) / float(node_count),
        float(child_count) / float(node_count),
        float(sibling_count) / float(node_count),
        _safe_norm(sibling_position, max(1, sibling_count)),
        1.0 if parent_group_index is not None else 0.0,
        _safe_norm(parent_group_index, node_count),
        1.0 if inserted_container_index is not None else 0.0,
        _safe_norm(inserted_container_index, node_count),
        1.0 if divides_target_index is not None else 0.0,
        _safe_norm(divides_target_index, node_count),
        float(contains_parent_degree) / float(node_count),
        float(contains_child_degree) / float(node_count),
        float(inserted_object_degree) / float(node_count),
        float(inserted_container_degree) / float(node_count),
        float(divides_divider_degree) / float(node_count),
        float(divides_target_degree) / float(node_count),
        float(adjacent_degree) / float(node_count),
        float(total_degree) / float(max(1, node_count * 2)),
        float(relation_counts.get("contains", 0)) / float(relation_count),
        float(relation_counts.get("inserted_in", 0)) / float(relation_count),
        float(relation_counts.get("divides", 0)) / float(relation_count),
        float(relation_counts.get("adjacent_to", 0)) / float(relation_count),
        1.0 if bool(node.get("renderable", True)) else 0.0,
        1.0 if bool(node.get("is_reference_only", False)) else 0.0,
    ]


def frame_to_bins(frame: dict, *, config: ParseGraphTokenizerConfig) -> dict:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return {
        "origin_x": int(
            quantize(float(origin[0]), low=config.position_min, high=config.position_max, bins=config.position_bins)
        ),
        "origin_y": int(
            quantize(float(origin[1]), low=config.position_min, high=config.position_max, bins=config.position_bins)
        ),
        "scale": int(quantize(float(frame.get("scale", 1.0)), low=config.scale_min, high=config.scale_max, bins=config.scale_bins)),
        "orientation": int(
            quantize(
                float(frame.get("orientation", 0.0)),
                low=config.angle_min,
                high=config.angle_max,
                bins=config.angle_bins,
            )
        ),
    }


def bins_to_frame(bins: dict, *, config: ParseGraphTokenizerConfig) -> dict:
    return {
        "origin": [
            dequantize(int(bins["origin_x"]), low=config.position_min, high=config.position_max, bins=config.position_bins),
            dequantize(int(bins["origin_y"]), low=config.position_min, high=config.position_max, bins=config.position_bins),
        ],
        "scale": dequantize(int(bins["scale"]), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
        "orientation": dequantize(int(bins["orientation"]), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
    }


def build_layout_frame_example(
    topology_target: dict,
    *,
    node_index: int,
    frame: dict | None = None,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    node = nodes[int(node_index)]
    example = {
        "role_id": int(ROLE_TO_ID.get(str(node.get("role", "unknown")), ROLE_TO_ID["unknown"])),
        "label_id": max(0, min(int(node.get("label", 0)), 63)),
        "geometry_model_id": int(
            GEOMETRY_MODEL_TO_ID.get(str(node.get("geometry_model", "unknown")), GEOMETRY_MODEL_TO_ID["unknown"])
        ),
        "numeric": _numeric_features(topology_target, int(node_index)),
        "node_index": int(node_index),
        "node_id": str(node.get("id", "")),
        "role": str(node.get("role", "unknown")),
        "label": int(node.get("label", 0)),
        "geometry_model": str(node.get("geometry_model", "unknown")),
    }
    if frame is not None:
        example["target_bins"] = frame_to_bins(frame, config=config)
        example["target_frame"] = copy.deepcopy(frame)
    return example


def _geometry_by_source_node_id(geometry_targets: Sequence[dict]) -> Dict[str, dict]:
    return {str(target.get("source_node_id")): target for target in geometry_targets}


def iter_layout_frame_examples_from_split(
    split_root: Path,
    *,
    config: ParseGraphTokenizerConfig | None = None,
    max_samples: int | None = None,
) -> Iterable[dict]:
    config = config or ParseGraphTokenizerConfig()
    manifest_path = Path(split_root) / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    for row in rows:
        topology_path = _resolve_path(row["topology_path"], split_root=Path(split_root), manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
        node_index_by_id = _node_index_by_id(nodes)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=Path(split_root), manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        for source_node_id, geometry_target in _geometry_by_source_node_id(geometry_targets).items():
            if source_node_id not in node_index_by_id or "frame" not in geometry_target:
                continue
            yield {
                "source_topology": str(topology_path.as_posix()),
                "source_node_id": str(source_node_id),
                "stem": row.get("stem"),
                **build_layout_frame_example(
                    topology_target,
                    node_index=int(node_index_by_id[source_node_id]),
                    frame=geometry_target.get("frame", {}),
                    config=config,
                ),
            }


class ManualLayoutFrameDataset(Dataset):
    def __init__(
        self,
        split_root: str | Path,
        *,
        config: ParseGraphTokenizerConfig | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.split_root = Path(split_root)
        self.config = config or ParseGraphTokenizerConfig()
        self.rows = list(
            iter_layout_frame_examples_from_split(self.split_root, config=self.config, max_samples=max_samples)
        )
        if not self.rows:
            raise RuntimeError(f"No layout frame examples found in {self.split_root}")
        self.numeric_dim = len(self.rows[0]["numeric"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[int(index)]
        bins = row["target_bins"]
        return {
            "role_id": torch.tensor(int(row["role_id"]), dtype=torch.long),
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "geometry_model_id": torch.tensor(int(row["geometry_model_id"]), dtype=torch.long),
            "numeric": torch.tensor(row["numeric"], dtype=torch.float32),
            "origin_x": torch.tensor(int(bins["origin_x"]), dtype=torch.long),
            "origin_y": torch.tensor(int(bins["origin_y"]), dtype=torch.long),
            "scale": torch.tensor(int(bins["scale"]), dtype=torch.long),
            "orientation": torch.tensor(int(bins["orientation"]), dtype=torch.long),
            "metadata": {
                key: row.get(key)
                for key in ("source_topology", "source_node_id", "node_id", "node_index", "role", "label", "geometry_model")
            },
        }


def collate_layout_frame_examples(batch: Sequence[dict]) -> dict:
    return {
        "role_id": torch.stack([item["role_id"] for item in batch], dim=0),
        "label_id": torch.stack([item["label_id"] for item in batch], dim=0),
        "geometry_model_id": torch.stack([item["geometry_model_id"] for item in batch], dim=0),
        "numeric": torch.stack([item["numeric"] for item in batch], dim=0),
        "origin_x": torch.stack([item["origin_x"] for item in batch], dim=0),
        "origin_y": torch.stack([item["origin_y"] for item in batch], dim=0),
        "scale": torch.stack([item["scale"] for item in batch], dim=0),
        "orientation": torch.stack([item["orientation"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


@dataclass
class ManualLayoutFrameMLPConfig:
    numeric_dim: int
    role_count: int = 6
    label_count: int = 64
    geometry_model_count: int = 4
    role_emb_dim: int = 16
    label_emb_dim: int = 16
    geometry_emb_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    position_bins: int = 1024
    scale_bins: int = 1024
    angle_bins: int = 1024


class ManualLayoutFrameMLP(nn.Module):
    def __init__(self, config: ManualLayoutFrameMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(int(config.role_count), int(config.role_emb_dim))
        self.label_embedding = nn.Embedding(int(config.label_count), int(config.label_emb_dim))
        self.geometry_embedding = nn.Embedding(int(config.geometry_model_count), int(config.geometry_emb_dim))
        input_dim = int(config.numeric_dim + config.role_emb_dim + config.label_emb_dim + config.geometry_emb_dim)
        layers: List[nn.Module] = []
        for layer_index in range(int(config.num_layers)):
            layers.append(nn.Linear(input_dim if layer_index == 0 else int(config.hidden_dim), int(config.hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(config.dropout)))
        self.backbone = nn.Sequential(*layers)
        self.origin_x_head = nn.Linear(int(config.hidden_dim), int(config.position_bins))
        self.origin_y_head = nn.Linear(int(config.hidden_dim), int(config.position_bins))
        self.scale_head = nn.Linear(int(config.hidden_dim), int(config.scale_bins))
        self.orientation_head = nn.Linear(int(config.hidden_dim), int(config.angle_bins))

    def forward(self, batch: dict) -> dict:
        role = self.role_embedding(batch["role_id"].clamp(0, self.config.role_count - 1))
        label = self.label_embedding(batch["label_id"].clamp(0, self.config.label_count - 1))
        geometry = self.geometry_embedding(batch["geometry_model_id"].clamp(0, self.config.geometry_model_count - 1))
        x = torch.cat([role, label, geometry, batch["numeric"].float()], dim=-1)
        hidden = self.backbone(x)
        return {
            "origin_x": self.origin_x_head(hidden),
            "origin_y": self.origin_y_head(hidden),
            "scale": self.scale_head(hidden),
            "orientation": self.orientation_head(hidden),
        }


def layout_frame_loss(logits: dict, batch: dict) -> torch.Tensor:
    losses = [
        F.cross_entropy(logits["origin_x"], batch["origin_x"]),
        F.cross_entropy(logits["origin_y"], batch["origin_y"]),
        F.cross_entropy(logits["scale"], batch["scale"]),
        F.cross_entropy(logits["orientation"], batch["orientation"]),
    ]
    return sum(losses) / float(len(losses))


def move_layout_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    for key in ("role_id", "label_id", "geometry_model_id", "numeric", "origin_x", "origin_y", "scale", "orientation"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def predict_frame_bins(model: ManualLayoutFrameMLP, example: dict, *, device: torch.device) -> dict:
    batch = {
        "role_id": torch.tensor([int(example["role_id"])], dtype=torch.long, device=device),
        "label_id": torch.tensor([int(example["label_id"])], dtype=torch.long, device=device),
        "geometry_model_id": torch.tensor([int(example["geometry_model_id"])], dtype=torch.long, device=device),
        "numeric": torch.tensor([example["numeric"]], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        logits = model(batch)
    return {name: int(torch.argmax(value, dim=-1).item()) for name, value in logits.items()}


def _angle_abs_error(left: float, right: float) -> float:
    diff = abs(float(left) - float(right))
    return float(min(diff, 2.0 * math.pi - diff))


@torch.no_grad()
def evaluate_layout_frame_model(
    model: ManualLayoutFrameMLP,
    loader,
    *,
    device: torch.device,
    config: ParseGraphTokenizerConfig,
) -> dict:
    model.eval()
    losses: List[float] = []
    counts = Counter()
    correct = Counter()
    errors = defaultdict(list)
    role_errors = defaultdict(lambda: defaultdict(list))
    role_counts = Counter()

    for batch in loader:
        moved = move_layout_batch_to_device(batch, device)
        logits = model(moved)
        loss = layout_frame_loss(logits, moved)
        losses.append(float(loss.item()))
        predictions = {name: torch.argmax(logits[name], dim=-1).detach().cpu() for name in logits}
        targets = {name: batch[name].detach().cpu() for name in predictions}
        for name in predictions:
            counts[name] += int(predictions[name].numel())
            correct[name] += int((predictions[name] == targets[name]).sum().item())
        for row_index, metadata in enumerate(batch["metadata"]):
            role = str(metadata.get("role", "unknown"))
            role_counts[role] += 1
            pred_frame = bins_to_frame({name: int(predictions[name][row_index]) for name in predictions}, config=config)
            target_frame = bins_to_frame({name: int(targets[name][row_index]) for name in targets}, config=config)
            origin_error = math.dist(pred_frame["origin"], target_frame["origin"])
            scale_error = abs(float(pred_frame["scale"]) - float(target_frame["scale"]))
            orientation_error = _angle_abs_error(float(pred_frame["orientation"]), float(target_frame["orientation"]))
            for key, value in (
                ("origin_mae", origin_error),
                ("scale_mae", scale_error),
                ("orientation_mae", orientation_error),
            ):
                errors[key].append(float(value))
                role_errors[role][key].append(float(value))

    def avg(values: Sequence[float]) -> float | None:
        return float(mean(values)) if values else None

    return {
        "loss": avg(losses),
        "head_accuracy": {
            name: float(correct[name] / counts[name]) if counts[name] else 0.0 for name in ("origin_x", "origin_y", "scale", "orientation")
        },
        "origin_mae": avg(errors["origin_mae"]),
        "scale_mae": avg(errors["scale_mae"]),
        "orientation_mae": avg(errors["orientation_mae"]),
        "role_metrics": {
            role: {
                "count": int(role_counts[role]),
                "origin_mae": avg(values["origin_mae"]),
                "scale_mae": avg(values["scale_mae"]),
                "orientation_mae": avg(values["orientation_mae"]),
            }
            for role, values in sorted(role_errors.items())
        },
    }


def save_layout_frame_checkpoint(
    path: Path,
    *,
    model: ManualLayoutFrameMLP,
    optimizer: torch.optim.Optimizer | None,
    model_config: ManualLayoutFrameMLPConfig,
    tokenizer_config: ParseGraphTokenizerConfig,
    train_config: dict,
    metrics: dict,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "maskgen_manual_layout_frame_checkpoint_v1",
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


def load_layout_frame_checkpoint(path: Path, *, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_config = ManualLayoutFrameMLPConfig(**checkpoint["model_config"])
    tokenizer_config = ParseGraphTokenizerConfig(**checkpoint["tokenizer_config"])
    model = ManualLayoutFrameMLP(model_config)
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model, tokenizer_config


def _attach_geometry_payload(node: dict, geometry_target: dict, *, frame: dict) -> dict:
    output = copy.deepcopy(node)
    output.pop("geometry_ref", None)
    output["geometry_model"] = copy.deepcopy(geometry_target.get("geometry_model", output.get("geometry_model")))
    output["frame"] = copy.deepcopy(frame)
    if "geometry" in geometry_target:
        output["geometry"] = copy.deepcopy(geometry_target["geometry"])
    if "atoms" in geometry_target:
        output["atoms"] = copy.deepcopy(geometry_target["atoms"])
    output["layout_frame_source"] = "predicted"
    return output


def attach_predicted_frames_to_split_rows(
    split_root: Path,
    *,
    model: ManualLayoutFrameMLP,
    tokenizer_config: ParseGraphTokenizerConfig,
    device: torch.device,
    max_samples: int | None = None,
) -> List[dict]:
    manifest_path = Path(split_root) / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    outputs: List[dict] = []
    for row_index, row in enumerate(rows):
        topology_path = _resolve_path(row["topology_path"], split_root=Path(split_root), manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
        node_index = _node_index_by_id(nodes)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=Path(split_root), manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        geometry_by_id = _geometry_by_source_node_id(geometry_targets)
        output_nodes: List[dict] = []
        for index, node in enumerate(nodes):
            geometry_ref = node.get("geometry_ref")
            if geometry_ref and str(geometry_ref) in geometry_by_id:
                example = build_layout_frame_example(topology_target, node_index=int(index), config=tokenizer_config)
                bins = predict_frame_bins(model, example, device=device)
                frame = bins_to_frame(bins, config=tokenizer_config)
                output_nodes.append(_attach_geometry_payload(node, geometry_by_id[str(geometry_ref)], frame=frame))
            else:
                output_node = copy.deepcopy(node)
                output_node.pop("geometry_ref", None)
                output_nodes.append(output_node)
        outputs.append(
            {
                "format": "maskgen_generator_target_v1",
                "target_type": "parse_graph",
                "size": copy.deepcopy(topology_target.get("size", [256, 256])),
                "parse_graph": {
                    "nodes": output_nodes,
                    "relations": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("relations", []) or []),
                    "residuals": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("residuals", []) or []),
                },
                "metadata": {
                    "layout_frame_predicted": True,
                    "source_topology": str(topology_path.as_posix()),
                    "sample_index": int(row_index),
                },
            }
        )
    return outputs


def attach_predicted_frames_to_topology_sample_rows(
    rows: Sequence[dict],
    *,
    model: ManualLayoutFrameMLP,
    tokenizer_config: ParseGraphTokenizerConfig,
    device: torch.device,
    shape_library: GeometryPlaceholderLibrary,
    include_invalid: bool = False,
) -> List[dict]:
    outputs: List[dict] = []
    for fallback_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not include_invalid:
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
        output_nodes: List[dict] = []
        missing_shape_count = 0
        for index, node in enumerate(nodes):
            if node.get("geometry_ref"):
                shape_target, _mode = shape_library.choose(
                    role=str(node.get("role", "")),
                    label=int(node.get("label", 0)),
                    geometry_model=str(node.get("geometry_model", "polygon_code")),
                )
                if shape_target is None:
                    missing_shape_count += 1
                    output_node = copy.deepcopy(node)
                    output_node.pop("geometry_ref", None)
                    output_nodes.append(output_node)
                    continue
                example = build_layout_frame_example(topology_target, node_index=int(index), config=tokenizer_config)
                bins = predict_frame_bins(model, example, device=device)
                frame = bins_to_frame(bins, config=tokenizer_config)
                output_nodes.append(_attach_geometry_payload(node, shape_target, frame=frame))
            else:
                output_node = copy.deepcopy(node)
                output_node.pop("geometry_ref", None)
                output_nodes.append(output_node)
        outputs.append(
            {
                "format": "maskgen_generator_target_v1",
                "target_type": "parse_graph",
                "size": copy.deepcopy(topology_target.get("size", [256, 256])),
                "parse_graph": {
                    "nodes": output_nodes,
                    "relations": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("relations", []) or []),
                    "residuals": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("residuals", []) or []),
                },
                "metadata": {
                    "layout_frame_predicted": True,
                    "placeholder_shape": True,
                    "sample_index": int(row.get("sample_index", fallback_index)),
                    "semantic_valid": bool(validation["semantic_valid"]),
                    "missing_shape_count": int(missing_shape_count),
                },
            }
        )
    return outputs
