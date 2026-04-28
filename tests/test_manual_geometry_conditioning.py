from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from partition_gen.manual_geometry_conditioning import (
    CONDITION_TOKEN,
    build_conditioned_geometry_sequence_rows,
    conditioned_geometry_prefix_tokens,
    encode_conditioned_geometry_target,
    extract_geometry_tokens_from_conditioned,
    geometry_start_index,
)
from partition_gen.manual_geometry_constrained_sampling import (
    GeometryConstrainedSamplerConfig,
    sample_geometry_constrained,
)
from partition_gen.manual_split_token_dataset import (
    ManualSplitTokenSequenceDataset,
    collate_manual_split_token_sequences,
)
from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary, save_vocabulary


class _BadLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, invalid_id: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"block_size": 256})()
        self.vocab_size = int(vocab_size)
        self.invalid_id = int(invalid_id)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        batch, seq_len = input_ids.shape
        logits = self.anchor.new_zeros((batch, seq_len, self.vocab_size))
        logits[:, :, self.invalid_id] = 1000.0
        return {"logits": logits}


def _topology_target() -> dict:
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": [256, 256],
        "parse_graph": {
            "nodes": [
                {
                    "id": "support_0",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "support_0",
                }
            ],
            "relations": [],
            "residuals": [],
        },
    }


def _geometry_target() -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": "support_0",
        "role": "support_region",
        "label": 0,
        "geometry_model": "polygon_code",
        "frame": {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


class ManualGeometryConditioningTest(unittest.TestCase):
    def test_conditioned_geometry_tokens_extract_geometry_payload(self) -> None:
        tokens = encode_conditioned_geometry_target(_topology_target(), _geometry_target(), target_node_index=0)

        self.assertEqual(tokens[1], CONDITION_TOKEN)
        self.assertGreater(geometry_start_index(tokens), 0)
        self.assertEqual(extract_geometry_tokens_from_conditioned(tokens), encode_geometry_target(_geometry_target()))

    def test_conditioned_prefix_can_drive_constrained_sampler(self) -> None:
        vocab = build_token_vocabulary(ParseGraphTokenizerConfig())
        model = _BadLogitModel(len(vocab), invalid_id=vocab["<EOS>"])
        prefix = conditioned_geometry_prefix_tokens(_topology_target(), target_node_index=0)

        sample = sample_geometry_constrained(
            model,
            vocab,
            prefix_tokens=prefix,
            max_new_tokens=128,
            temperature=0.0,
            top_k=None,
            constraint_config=GeometryConstrainedSamplerConfig(max_polygons=1, max_points_per_ring=3, max_holes_per_polygon=0),
            device=torch.device("cpu"),
        )

        geometry_tokens = extract_geometry_tokens_from_conditioned(sample["tokens"])
        self.assertEqual(geometry_tokens[:7], encode_geometry_target(_geometry_target())[:7])
        self.assertTrue(sample["hit_eos"])

    def test_build_conditioned_rows_and_dataset_masks_condition_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_root = root / "targets" / "train"
            topology_path = split_root / "topology" / "graphs" / "0.json"
            geometry_path = split_root / "geometry" / "0" / "support_0.json"
            topology_path.parent.mkdir(parents=True)
            geometry_path.parent.mkdir(parents=True)
            topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
            geometry_path.write_text(json.dumps(_geometry_target()), encoding="utf-8")
            (split_root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "stem": "0",
                        "topology_path": str(topology_path.as_posix()),
                        "geometry_paths": [str(geometry_path.as_posix())],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = ParseGraphTokenizerConfig()
            vocab = build_token_vocabulary(config)
            rows, _summary = build_conditioned_geometry_sequence_rows(split_root, config=config, vocab=vocab)
            token_root = root / "tokens" / "train"
            token_root.mkdir(parents=True)
            save_vocabulary(token_root / "vocab.json", vocab, config=config)
            with (token_root / "conditioned_geometry_sequences.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind="conditioned_geometry")
            batch = collate_manual_split_token_sequences([dataset[0]], pad_id=dataset.pad_id)
            loss_start = int(rows[0]["loss_start_index"])

            self.assertEqual(len(dataset), 1)
            self.assertTrue(torch.all(batch["labels"][0, : loss_start - 1] == -100))
            self.assertNotEqual(int(batch["labels"][0, loss_start - 1]), -100)


if __name__ == "__main__":
    unittest.main()
