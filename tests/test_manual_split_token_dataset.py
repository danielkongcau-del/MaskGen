from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_split_token_audit import audit_manual_split_tokens
from partition_gen.manual_split_token_dataset import (
    ManualSplitTokenSequenceDataset,
    build_manual_split_token_dataloader,
    collate_manual_split_token_sequences,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary, save_vocabulary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class ManualSplitTokenDatasetTest(unittest.TestCase):
    def make_token_root(self, root: Path) -> Path:
        token_root = root / "tokens"
        token_root.mkdir()
        config = ParseGraphTokenizerConfig(max_int=64, coord_bins=32, area_bins=32)
        save_vocabulary(token_root / "vocab.json", build_token_vocabulary(config), config=config)
        _write_jsonl(
            token_root / "topology_sequences.jsonl",
            [
                {
                    "format": "maskgen_tokenized_parse_graph_v1",
                    "tokenizer": "manual_topology_v1",
                    "stem": "a",
                    "source_target": "topology/a.json",
                    "length": 4,
                    "tokens": ["<BOS>", "MANUAL_TOPOLOGY_V1", "SIZE", "<EOS>"],
                },
                {
                    "format": "maskgen_tokenized_parse_graph_v1",
                    "tokenizer": "manual_topology_v1",
                    "stem": "b",
                    "source_target": "topology/b.json",
                    "length": 3,
                    "tokens": ["<BOS>", "MANUAL_TOPOLOGY_V1", "<EOS>"],
                },
            ],
        )
        _write_jsonl(
            token_root / "geometry_sequences.jsonl",
            [
                {
                    "format": "maskgen_tokenized_parse_graph_v1",
                    "tokenizer": "manual_geometry_v1",
                    "stem": "a",
                    "source_target": "geometry/a/support_0.json",
                    "source_node_id": "support_0",
                    "length": 3,
                    "tokens": ["<BOS>", "MANUAL_GEOMETRY_V1", "<EOS>"],
                }
            ],
        )
        _write_jsonl(
            token_root / "coarse_scene_sequences.jsonl",
            [
                {
                    "format": "maskgen_tokenized_parse_graph_v1",
                    "tokenizer": "manual_coarse_scene_v1",
                    "stem": "a",
                    "source_target": "topology/a.json",
                    "length": 3,
                    "tokens": ["<BOS>", "MANUAL_COARSE_SCENE_V1", "<EOS>"],
                }
            ],
        )
        _write_jsonl(
            token_root / "manifest.jsonl",
            [
                {
                    "stem": "a",
                    "topology_written": True,
                    "geometry_written_count": 1,
                    "geometry_target_count": 1,
                }
            ],
        )
        (token_root / "summary.json").write_text(json.dumps({"sample_count": 1}), encoding="utf-8")
        return token_root

    def test_audit_manual_split_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_root = self.make_token_root(Path(tmp))
            audit = audit_manual_split_tokens(token_root, top_k=2)
            self.assertEqual(audit["topology"]["sequence_count"], 2)
            self.assertEqual(audit["geometry"]["sequence_count"], 1)
            self.assertEqual(audit["max_single_sequence_tokens"], 4)
            self.assertEqual(audit["topology"]["lengths"]["max"], 4)

    def test_dataset_maps_tokens_to_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_root = self.make_token_root(Path(tmp))
            dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind="topology")
            item = dataset[0]
            self.assertEqual(item["length"], 4)
            self.assertEqual(item["sequence_kind"], "topology")
            self.assertEqual(tuple(item["token_ids"].shape), (4,))

    def test_collate_builds_next_token_labels_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_root = self.make_token_root(Path(tmp))
            dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind="topology")
            batch = collate_manual_split_token_sequences([dataset[0], dataset[1]], pad_id=dataset.pad_id)
            self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
            self.assertEqual(tuple(batch["labels"].shape), (2, 3))
            self.assertEqual(int(batch["attention_mask"].sum().item()), 5)
            self.assertEqual(int(batch["labels"][1, 2].item()), -100)

    def test_dataloader_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_root = self.make_token_root(Path(tmp))
            loader = build_manual_split_token_dataloader(token_root, sequence_kind="geometry", batch_size=1)
            batch = next(iter(loader))
            self.assertEqual(batch["sequence_kind"], "geometry")
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 2))

    def test_coarse_scene_sequence_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_root = self.make_token_root(Path(tmp))
            dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind="coarse_scene")
            item = dataset[0]
            self.assertEqual(item["sequence_kind"], "coarse_scene")
            self.assertEqual(item["length"], 3)


if __name__ == "__main__":
    unittest.main()
