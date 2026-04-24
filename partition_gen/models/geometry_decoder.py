from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class GeometryDecoder(nn.Module):
    def __init__(
        self,
        *,
        node_feature_vocab_sizes: List[int],
        edge_feature_vocab_size: int,
        max_faces: int,
        max_neighbors: int,
        max_vertices: int,
        max_holes: int = 0,
        max_hole_vertices: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_faces = max_faces
        self.max_neighbors = max_neighbors
        self.max_vertices = max_vertices
        self.max_holes = max_holes
        self.max_hole_vertices = max_hole_vertices
        self.d_model = d_model

        self.node_feature_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model) for vocab_size in node_feature_vocab_sizes
        )
        self.edge_embedding = nn.Embedding(edge_feature_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_faces, d_model)
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

        self.support_head = nn.Linear(d_model, 2)
        self.vertex_count_head = nn.Linear(d_model, max_vertices + 1)
        self.vertex_slot_embedding = nn.Embedding(max_vertices, d_model)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        if max_holes > 0 and max_hole_vertices > 0:
            self.hole_count_head = nn.Linear(d_model, max_holes + 1)
            self.hole_slot_embedding = nn.Embedding(max_holes, d_model)
            self.hole_vertex_count_head = nn.Linear(d_model, max_hole_vertices + 1)
            self.hole_vertex_slot_embedding = nn.Embedding(max_hole_vertices, d_model)
            self.hole_coord_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
        else:
            self.hole_count_head = None
            self.hole_slot_embedding = None
            self.hole_vertex_count_head = None
            self.hole_vertex_slot_embedding = None
            self.hole_coord_head = None

    def _embed_nodes(self, node_features: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            (*node_features.shape[:2], self.d_model),
            dtype=torch.float32,
            device=node_features.device,
        )
        for feature_index, embedding in enumerate(self.node_feature_embeddings):
            hidden = hidden + embedding(node_features[:, :, feature_index])
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
        messages = gathered + edge_embed
        messages = messages * neighbor_mask[..., None]
        denom = neighbor_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        summary = messages.sum(dim=2) / denom
        return summary

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        face_mask: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_tokens: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_hidden = self._embed_nodes(node_features)
        neighbor_summary = self._neighbor_summary(node_hidden, neighbor_indices, neighbor_tokens, neighbor_mask)
        positions = torch.arange(node_features.shape[1], device=node_features.device)
        hidden = node_hidden + neighbor_summary + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~face_mask)
        hidden = self.final_norm(hidden)

        support_logits = self.support_head(hidden)
        vertex_count_logits = self.vertex_count_head(hidden)

        slot_ids = torch.arange(self.max_vertices, device=node_features.device)
        slot_embed = self.vertex_slot_embedding(slot_ids)[None, None, :, :]
        coord_queries = hidden[:, :, None, :] + slot_embed
        vertex_coords = self.coord_head(coord_queries)

        if self.max_holes > 0 and self.max_hole_vertices > 0:
            hole_ids = torch.arange(self.max_holes, device=node_features.device)
            hole_embed = self.hole_slot_embedding(hole_ids)[None, None, :, :]
            hole_hidden = hidden[:, :, None, :] + hole_embed
            hole_count_logits = self.hole_count_head(hidden)
            hole_vertex_count_logits = self.hole_vertex_count_head(hole_hidden)

            hole_vertex_ids = torch.arange(self.max_hole_vertices, device=node_features.device)
            hole_vertex_embed = self.hole_vertex_slot_embedding(hole_vertex_ids)[None, None, None, :, :]
            hole_coord_queries = hole_hidden[:, :, :, None, :] + hole_vertex_embed
            hole_vertex_coords = self.hole_coord_head(hole_coord_queries)
        else:
            batch_size, num_faces = hidden.shape[:2]
            hole_count_logits = hidden.new_zeros((batch_size, num_faces, 0))
            hole_vertex_count_logits = hidden.new_zeros((batch_size, num_faces, 0, 0))
            hole_vertex_coords = hidden.new_zeros((batch_size, num_faces, 0, 0, 2))

        return {
            "hidden": hidden,
            "support_logits": support_logits,
            "vertex_count_logits": vertex_count_logits,
            "vertex_coords": vertex_coords,
            "hole_count_logits": hole_count_logits,
            "hole_vertex_count_logits": hole_vertex_count_logits,
            "hole_vertex_coords": hole_vertex_coords,
        }


def build_geometry_model_from_metadata(
    *,
    binner_meta: Dict[str, object],
    max_faces: int,
    max_neighbors: int,
    max_vertices: int,
    max_holes: int = 0,
    max_hole_vertices: int = 0,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
) -> GeometryDecoder:
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
    return GeometryDecoder(
        node_feature_vocab_sizes=node_feature_vocab_sizes,
        edge_feature_vocab_size=edge_feature_vocab_size,
        max_faces=max_faces,
        max_neighbors=max_neighbors,
        max_vertices=max_vertices,
        max_holes=max_holes,
        max_hole_vertices=max_hole_vertices,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
