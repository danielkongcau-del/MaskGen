from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class PairConvBlock(nn.Module):
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


class PairBoundaryPredictor(nn.Module):
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
        target_size: int = 64,
        pair_hidden: int = 128,
        pair_token_channels: int = 32,
        decoder_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.max_faces = max_faces
        self.d_model = d_model
        self.target_size = target_size
        self.pair_token_channels = pair_token_channels

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

        self.border_hidden = nn.Parameter(torch.zeros(d_model))
        self.pair_feature_mlp = nn.Sequential(
            nn.Linear(13, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_model * 4, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, pair_hidden),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(pair_hidden, pair_token_channels)

        decoder_in = pair_token_channels + 10
        self.stem = PairConvBlock(decoder_in, decoder_hidden)
        self.mid = PairConvBlock(decoder_hidden, decoder_hidden)
        self.out_head = nn.Conv2d(decoder_hidden, 1, kernel_size=1)

        coords = (torch.arange(target_size, dtype=torch.float32) + 0.5) / float(target_size)
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

    def _soft_box(self, bbox_ratios: torch.Tensor, tau: float) -> torch.Tensor:
        x0 = bbox_ratios[..., 0][:, :, None, None]
        y0 = bbox_ratios[..., 1][:, :, None, None]
        x1 = bbox_ratios[..., 2][:, :, None, None]
        y1 = bbox_ratios[..., 3][:, :, None, None]
        box = torch.sigmoid((self.grid_x - x0) / tau)
        box = box * torch.sigmoid((x1 - self.grid_x) / tau)
        box = box * torch.sigmoid((self.grid_y - y0) / tau)
        box = box * torch.sigmoid((y1 - self.grid_y) / tau)
        return box

    def _seed_heat(self, centroids: torch.Tensor, sigma: float) -> torch.Tensor:
        cx = centroids[..., 0][:, :, None, None]
        cy = centroids[..., 1][:, :, None, None]
        return torch.exp(
            -0.5
            * (
                ((self.grid_x - cx) / sigma) ** 2
                + ((self.grid_y - cy) / sigma) ** 2
            )
        )

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
        pair_indices: torch.Tensor,
        pair_features: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_hidden = self._embed_nodes(node_features, centroid_ratios, bbox_ratios)
        neighbor_summary = self._neighbor_summary(node_hidden, neighbor_indices, neighbor_tokens, neighbor_mask)
        positions = torch.arange(node_features.shape[1], device=node_features.device)
        hidden = node_hidden + neighbor_summary + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~face_mask)
        hidden = self.final_norm(hidden)

        batch_size, max_pairs = pair_indices.shape[:2]
        hidden_dim = hidden.shape[-1]

        u_idx = pair_indices[..., 0].clamp(min=0)
        v_idx = pair_indices[..., 1].clamp(min=0)
        u_hidden = torch.gather(hidden, 1, u_idx[..., None].expand(batch_size, max_pairs, hidden_dim))
        v_hidden = torch.gather(hidden, 1, v_idx[..., None].expand(batch_size, max_pairs, hidden_dim))
        border_mask = pair_indices[..., 1] < 0
        v_hidden = torch.where(border_mask[..., None], self.border_hidden[None, None, :], v_hidden)

        u_centroids = torch.gather(
            centroid_ratios,
            1,
            u_idx[..., None].expand(batch_size, max_pairs, 2),
        )
        v_centroids = torch.gather(
            centroid_ratios,
            1,
            v_idx[..., None].expand(batch_size, max_pairs, 2),
        )
        v_centroids = torch.where(border_mask[..., None], u_centroids, v_centroids)

        u_bbox = torch.gather(
            bbox_ratios,
            1,
            u_idx[..., None].expand(batch_size, max_pairs, 4),
        )
        v_bbox = torch.gather(
            bbox_ratios,
            1,
            v_idx[..., None].expand(batch_size, max_pairs, 4),
        )
        v_bbox = torch.where(border_mask[..., None], u_bbox, v_bbox)

        pair_feature_hidden = self.pair_feature_mlp(pair_features)
        pair_hidden = self.pair_mlp(
            torch.cat([u_hidden, v_hidden, torch.abs(u_hidden - v_hidden), pair_feature_hidden], dim=-1)
        )
        token = self.token_proj(pair_hidden)[:, :, :, None, None].expand(
            batch_size,
            max_pairs,
            self.pair_token_channels,
            self.target_size,
            self.target_size,
        )

        tau = 1.5 / float(self.target_size)
        sigma = 1.25 / float(self.target_size)
        box_u = self._soft_box(u_bbox, tau=tau)
        box_v = self._soft_box(v_bbox, tau=tau) * (~border_mask)[..., None, None].to(u_bbox.dtype)
        union_bbox = pair_features[..., 4:8]
        union_box = self._soft_box(union_bbox, tau=tau)
        seed_u = self._seed_heat(u_centroids, sigma=sigma)
        seed_v = self._seed_heat(v_centroids, sigma=sigma) * (~border_mask)[..., None, None].to(u_bbox.dtype)

        border_channels = pair_features[..., 8:13]
        border_map = border_channels[:, :, :, None, None].expand(batch_size, max_pairs, 5, self.target_size, self.target_size)

        x = torch.cat(
            [
                token,
                box_u[:, :, None, :, :],
                box_v[:, :, None, :, :],
                union_box[:, :, None, :, :],
                seed_u[:, :, None, :, :],
                seed_v[:, :, None, :, :],
                border_map,
            ],
            dim=2,
        )
        x = x.view(batch_size * max_pairs, x.shape[2], self.target_size, self.target_size)
        x = self.stem(x)
        x = self.mid(x)
        pair_logits = self.out_head(x).view(batch_size, max_pairs, 1, self.target_size, self.target_size)
        pair_logits = pair_logits * pair_valid[:, :, None, None, None].to(pair_logits.dtype)

        return {
            "hidden": hidden,
            "pair_logits": pair_logits,
        }


def build_pair_boundary_model_from_metadata(
    *,
    binner_meta: Dict[str, object],
    max_faces: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    target_size: int = 64,
    pair_hidden: int = 128,
    pair_token_channels: int = 32,
    decoder_hidden: int = 64,
) -> PairBoundaryPredictor:
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
    return PairBoundaryPredictor(
        node_feature_vocab_sizes=node_feature_vocab_sizes,
        edge_feature_vocab_size=edge_feature_vocab_size,
        max_faces=max_faces,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        target_size=target_size,
        pair_hidden=pair_hidden,
        pair_token_channels=pair_token_channels,
        decoder_hidden=decoder_hidden,
    )
