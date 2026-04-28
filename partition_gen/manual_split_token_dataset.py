from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from partition_gen.parse_graph_tokenizer import load_vocabulary, tokens_to_ids


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ManualSplitTokenSequenceDataset(Dataset):
    """Autoregressive token dataset for manual topology/geometry split sequences."""

    def __init__(
        self,
        token_root: str | Path,
        *,
        sequence_kind: str = "topology",
        max_length: int | None = None,
    ) -> None:
        self.token_root = Path(token_root)
        if sequence_kind not in {"topology", "geometry", "conditioned_geometry"}:
            raise ValueError(f"sequence_kind must be topology, geometry, or conditioned_geometry, got {sequence_kind}")
        self.sequence_kind = sequence_kind
        self.vocab = load_vocabulary(self.token_root / "vocab.json")
        self.pad_id = int(self.vocab["<PAD>"])
        self.unk_id = int(self.vocab["<UNK>"])
        filename_by_kind = {
            "topology": "topology_sequences.jsonl",
            "geometry": "geometry_sequences.jsonl",
            "conditioned_geometry": "conditioned_geometry_sequences.jsonl",
        }
        filename = filename_by_kind[sequence_kind]
        rows = _read_jsonl(self.token_root / filename)
        if max_length is not None:
            rows = [row for row in rows if int(row.get("length", len(row.get("tokens", [])))) <= int(max_length)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[int(index)]
        ids = row.get("ids")
        if ids is None:
            ids = tokens_to_ids([str(token) for token in row.get("tokens", [])], self.vocab)
        ids = [int(value) for value in ids]
        return {
            "token_ids": torch.tensor(ids, dtype=torch.long),
            "length": int(len(ids)),
            "stem": row.get("stem"),
            "source_target": row.get("source_target"),
            "source_node_id": row.get("source_node_id"),
            "target_node_index": row.get("target_node_index"),
            "loss_start_index": row.get("loss_start_index"),
            "sequence_kind": self.sequence_kind,
        }


def collate_manual_split_token_sequences(
    batch: Sequence[Dict[str, object]],
    *,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, object]:
    batch_size = len(batch)
    max_input_length = max(max(0, int(item["length"]) - 1) for item in batch) if batch else 0
    input_ids = torch.full((batch_size, max_input_length), int(pad_id), dtype=torch.long)
    labels = torch.full((batch_size, max_input_length), int(ignore_index), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_length), dtype=torch.bool)
    lengths = torch.zeros((batch_size,), dtype=torch.long)
    for row_index, item in enumerate(batch):
        ids = item["token_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        input_length = max(0, int(ids.numel()) - 1)
        lengths[row_index] = int(ids.numel())
        if input_length <= 0:
            continue
        input_ids[row_index, :input_length] = ids[:-1]
        labels[row_index, :input_length] = ids[1:]
        loss_start_index = item.get("loss_start_index")
        if loss_start_index is not None:
            label_start = max(0, int(loss_start_index) - 1)
            labels[row_index, : min(label_start, input_length)] = int(ignore_index)
        attention_mask[row_index, :input_length] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "stems": [item.get("stem") for item in batch],
        "source_targets": [item.get("source_target") for item in batch],
        "source_node_ids": [item.get("source_node_id") for item in batch],
        "target_node_indices": [item.get("target_node_index") for item in batch],
        "loss_start_indices": [item.get("loss_start_index") for item in batch],
        "sequence_kind": batch[0].get("sequence_kind") if batch else None,
    }


def build_manual_split_token_dataloader(
    token_root: str | Path,
    *,
    sequence_kind: str = "topology",
    batch_size: int = 8,
    shuffle: bool = False,
    max_length: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind=sequence_kind, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=lambda batch: collate_manual_split_token_sequences(batch, pad_id=dataset.pad_id),
    )
