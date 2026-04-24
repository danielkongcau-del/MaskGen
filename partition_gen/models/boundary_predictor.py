from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


class BoundaryPredictor(nn.Module):
    def __init__(
        self,
        *,
        node_feature_vocab_sizes: List[int],
        edge_feature_vocab_size: int,
        max_faces: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        base_size: int = 64,
        raster_channels: int = 96,
        scene_channels: int = 32,
        decoder_hidden: int = 128,
        num_labels: int = 7,
    ) -> None:
        super().__init__()
        self.max_faces = max_faces
        self.d_model = d_model
        self.base_size = base_size
        self.num_labels = num_labels

        self.node_feature_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model) for vocab_size in node_feature_vocab_sizes
        )
        self.edge_embedding = nn.Embedding(edge_feature_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_faces, d_model)
        self.continuous_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.face_raster_proj = nn.Linear(d_model, raster_channels)
        self.scene_proj = nn.Linear(d_model, scene_channels)

        decoder_in = raster_channels + scene_channels + num_labels + 3
        self.stem = ConvBlock(decoder_in, decoder_hidden)
        self.mid = ConvBlock(decoder_hidden, decoder_hidden)
        self.up1 = UpBlock(decoder_hidden, decoder_hidden // 2)
        self.up2 = UpBlock(decoder_hidden // 2, decoder_hidden // 4)
        self.out_head = nn.Conv2d(decoder_hidden // 4, 1, kernel_size=1)

        coords = (torch.arange(base_size, dtype=torch.float32) + 0.5) / float(base_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", grid_x[None, None, :, :], persistent=False)
        self.register_buffer("grid_y", grid_y[None, None, :, :], persistent=False)

    def _embed_nodes(
        self,
        node_features: torch.Tensor,
        centroid_ratios: torch.Tensor,
        bbox_ratios: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.zeros(
            (*node_features.shape[:2], self.d_model),
            dtype=torch.float32,
            device=node_features.device,
        )
        for feature_index, embedding in enumerate(self.node_feature_embeddings):
            hidden = hidden + embedding(node_features[:, :, feature_index])
        continuous = torch.cat([centroid_ratios, bbox_ratios], dim=-1)
        hidden = hidden + self.continuous_mlp(continuous)
        return hidden

    def _neighbor_summary(
        self,
        node_hidden: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_tokens: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if neighbor_indices.shape[-1] == 0:
            return torch.zeros_like(node_hidden)

        batch_size, num_faces, max_neighbors = neighbor_indices.shape
        hidden_dim = node_hidden.shape[-1]

        safe_indices = neighbor_indices.clamp(min=0)
        gathered = torch.gather(
            node_hidden[:, None, :, :].expand(batch_size, num_faces, num_faces, hidden_dim),
            2,
            safe_indices[..., None].expand(batch_size, num_faces, max_neighbors, hidden_dim),
        )
        edge_embed = self.edge_embedding(neighbor_tokens)
        messages = (gathered + edge_embed) * neighbor_mask[..., None]
        denom = neighbor_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        return messages.sum(dim=2) / denom

    def _build_spatial_maps(
        self,
        centroid_ratios: torch.Tensor,
        bbox_ratios: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x0 = bbox_ratios[..., 0][:, :, None, None]
        y0 = bbox_ratios[..., 1][:, :, None, None]
        x1 = bbox_ratios[..., 2][:, :, None, None]
        y1 = bbox_ratios[..., 3][:, :, None, None]
        cx = centroid_ratios[..., 0][:, :, None, None]
        cy = centroid_ratios[..., 1][:, :, None, None]

        width = (x1 - x0).clamp(min=1.0 / self.base_size)
        height = (y1 - y0).clamp(min=1.0 / self.base_size)
        tau = 1.5 / float(self.base_size)

        soft_box = torch.sigmoid((self.grid_x - x0) / tau)
        soft_box = soft_box * torch.sigmoid((x1 - self.grid_x) / tau)
        soft_box = soft_box * torch.sigmoid((self.grid_y - y0) / tau)
        soft_box = soft_box * torch.sigmoid((y1 - self.grid_y) / tau)

        sigma_x = (width * 0.35).clamp(min=1.5 / self.base_size)
        sigma_y = (height * 0.35).clamp(min=1.5 / self.base_size)
        gaussian = torch.exp(
            -0.5
            * (
                ((self.grid_x - cx) / sigma_x) ** 2
                + ((self.grid_y - cy) / sigma_y) ** 2
            )
        )
        seed_sigma = 1.25 / float(self.base_size)
        seed_heat = torch.exp(
            -0.5
            * (
                ((self.grid_x - cx) / seed_sigma) ** 2
                + ((self.grid_y - cy) / seed_sigma) ** 2
            )
        )

        mask = face_mask[:, :, None, None].to(soft_box.dtype)
        return {
            "box": soft_box * mask,
            "weight": soft_box * gaussian * mask,
            "seed": seed_heat * mask,
        }

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        face_mask: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_tokens: torch.Tensor,
        neighbor_mask: torch.Tensor,
        centroid_ratios: torch.Tensor,
        bbox_ratios: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_hidden = self._embed_nodes(node_features, centroid_ratios, bbox_ratios)
        neighbor_summary = self._neighbor_summary(node_hidden, neighbor_indices, neighbor_tokens, neighbor_mask)
        positions = torch.arange(node_features.shape[1], device=node_features.device)
        hidden = node_hidden + neighbor_summary + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~face_mask)
        hidden = self.final_norm(hidden)

        spatial = self._build_spatial_maps(centroid_ratios, bbox_ratios, face_mask)
        weight = spatial["weight"]
        box = spatial["box"]
        seed = spatial["seed"]

        occupancy = weight.sum(dim=1, keepdim=True)
        norm = occupancy.clamp(min=1e-6)

        face_raster = self.face_raster_proj(hidden)
        raster_features = torch.einsum("bfhw,bfc->bchw", weight, face_raster) / norm
        raster_features = torch.nan_to_num(raster_features)

        label_one_hot = F.one_hot(labels.clamp(min=0), num_classes=self.num_labels).to(weight.dtype)
        label_one_hot = label_one_hot * face_mask[..., None].to(weight.dtype)
        label_maps = torch.einsum("bfhw,bfk->bkhw", weight, label_one_hot) / norm
        label_maps = torch.nan_to_num(label_maps)

        pooled = (hidden * face_mask[..., None].to(hidden.dtype)).sum(dim=1)
        pooled = pooled / face_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
        scene = self.scene_proj(pooled)[:, :, None, None].expand(-1, -1, self.base_size, self.base_size)

        density_maps = torch.cat(
            [
                torch.log1p(occupancy),
                torch.log1p(box.sum(dim=1, keepdim=True)),
                torch.log1p(seed.sum(dim=1, keepdim=True)),
            ],
            dim=1,
        )

        x = torch.cat([raster_features, scene, label_maps, density_maps], dim=1)
        x = self.stem(x)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        boundary_logits = self.out_head(x)

        return {
            "hidden": hidden,
            "boundary_logits": boundary_logits,
            "occupancy": occupancy,
        }


def build_boundary_model_from_metadata(
    *,
    binner_meta: Dict[str, object],
    max_faces: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    base_size: int = 64,
    raster_channels: int = 96,
    scene_channels: int = 32,
    decoder_hidden: int = 128,
    num_labels: int = 7,
) -> BoundaryPredictor:
    node_feature_vocab_sizes = [
        7,
        len(binner_meta["area_ratio"]) + 1,
        len(binner_meta["centroid_x"]) + 1,
        len(binner_meta["centroid_y"]) + 1,
        len(binner_meta["bbox_width_ratio"]) + 1,
        len(binner_meta["bbox_height_ratio"]) + 1,
        len(binner_meta["perimeter_ratio"]) + 1,
        len(binner_meta["border_ratio"]) + 1,
        len(binner_meta["outer_vertices"]) + 1,
        len(binner_meta["hole_count"]) + 1,
        len(binner_meta["degree"]) + 1,
        2,
    ]
    edge_feature_vocab_size = len(binner_meta["shared_length_ratio"]) + 1
    return BoundaryPredictor(
        node_feature_vocab_sizes=node_feature_vocab_sizes,
        edge_feature_vocab_size=edge_feature_vocab_size,
        max_faces=max_faces,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        base_size=base_size,
        raster_channels=raster_channels,
        scene_channels=scene_channels,
        decoder_hidden=decoder_hidden,
        num_labels=num_labels,
    )
