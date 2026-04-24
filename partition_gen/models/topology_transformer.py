from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import nn


class TopologyARTransformer(nn.Module):
    def __init__(
        self,
        node_feature_vocab_sizes: List[int],
        edge_feature_vocab_size: int,
        max_faces: int,
        max_prev_neighbors: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_feature_vocab_sizes = list(node_feature_vocab_sizes)
        self.edge_feature_vocab_size = int(edge_feature_vocab_size)
        self.max_faces = int(max_faces)
        self.max_prev_neighbors = int(max_prev_neighbors)
        self.d_model = int(d_model)

        self.node_feature_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model) for vocab_size in self.node_feature_vocab_sizes
        )
        self.position_embedding = nn.Embedding(self.max_faces, d_model)
        self.bos_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
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

        self.node_feature_heads = nn.ModuleList(
            nn.Linear(d_model, vocab_size) for vocab_size in self.node_feature_vocab_sizes
        )
        self.face_exists_head = nn.Linear(d_model, 2)
        self.prev_count_head = nn.Linear(d_model, self.max_prev_neighbors + 1)

        self.prev_slot_embedding = nn.Embedding(self.max_prev_neighbors, d_model)
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)
        self.edge_token_head = nn.Linear(d_model, self.edge_feature_vocab_size)

    def _shift_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(node_features)
        shifted[:, 1:] = node_features[:, :-1]
        return shifted

    def _encode_inputs(self, node_features: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_faces, _ = node_features.shape
        shifted = self._shift_node_features(node_features)
        hidden = torch.zeros(
            (batch_size, num_faces, self.d_model),
            dtype=torch.float32,
            device=node_features.device,
        )
        for feature_index, embedding in enumerate(self.node_feature_embeddings):
            hidden = hidden + embedding(shifted[:, :, feature_index])

        positions = torch.arange(num_faces, device=node_features.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden[:, 0, :] = hidden[:, 0, :] + self.bos_embedding
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones((num_faces, num_faces), dtype=torch.bool, device=node_features.device),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~face_mask,
        )
        return self.final_norm(encoded)

    def _pointer_scores(self, hidden: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_faces, _ = hidden.shape
        slot_ids = torch.arange(self.max_prev_neighbors, device=hidden.device)
        slot_embeddings = self.prev_slot_embedding(slot_ids)[None, None, :, :]

        query = hidden[:, :, None, :] + slot_embeddings
        query = self.pointer_query(query)
        keys = self.pointer_key(hidden)
        scores = torch.einsum("btkd,bsd->bkts", query, keys) / math.sqrt(self.d_model)
        scores = scores.permute(0, 2, 1, 3).contiguous()

        target_positions = torch.arange(num_faces, device=hidden.device)
        valid_previous = target_positions[None, :, None] > target_positions[None, None, :]
        valid_previous = valid_previous[:, :, None, :].expand(batch_size, num_faces, self.max_prev_neighbors, num_faces)
        face_mask_expanded = face_mask[:, None, None, :].expand(batch_size, num_faces, self.max_prev_neighbors, num_faces)
        pointer_mask = valid_previous & face_mask_expanded
        scores = scores.masked_fill(~pointer_mask, float("-inf"))
        return scores

    def forward(self, node_features: torch.Tensor, face_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self._encode_inputs(node_features, face_mask)
        node_feature_logits = [
            head(hidden) for head in self.node_feature_heads
        ]
        face_exists_logits = self.face_exists_head(hidden)
        prev_count_logits = self.prev_count_head(hidden)
        prev_neighbor_logits = self._pointer_scores(hidden, face_mask)

        slot_ids = torch.arange(self.max_prev_neighbors, device=hidden.device)
        slot_embeddings = self.prev_slot_embedding(slot_ids)[None, None, :, :]
        edge_queries = hidden[:, :, None, :] + slot_embeddings
        edge_token_logits = self.edge_token_head(edge_queries)

        return {
            "hidden": hidden,
            "node_feature_logits": node_feature_logits,
            "face_exists_logits": face_exists_logits,
            "prev_count_logits": prev_count_logits,
            "prev_neighbor_logits": prev_neighbor_logits,
            "edge_token_logits": edge_token_logits,
        }


def build_model_from_metadata(
    binner_meta: Dict[str, object],
    max_faces: int,
    max_prev_neighbors: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
) -> TopologyARTransformer:
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
    return TopologyARTransformer(
        node_feature_vocab_sizes=node_feature_vocab_sizes,
        edge_feature_vocab_size=edge_feature_vocab_size,
        max_faces=max_faces,
        max_prev_neighbors=max_prev_neighbors,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
