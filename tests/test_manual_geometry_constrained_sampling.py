from __future__ import annotations

import unittest

import torch

from partition_gen.manual_geometry_constrained_sampling import (
    GeometryConstrainedSamplerConfig,
    GeometryGrammarState,
    sample_geometry_constrained,
)
from partition_gen.manual_geometry_sample_validation import validate_geometry_tokens
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary


class _BadLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, invalid_id: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"block_size": 128})()
        self.vocab_size = int(vocab_size)
        self.invalid_id = int(invalid_id)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, past_kv=None, use_cache: bool = False):
        batch, seq_len = input_ids.shape
        logits = self.anchor.new_zeros((batch, seq_len, self.vocab_size))
        logits[:, :, self.invalid_id] = 1000.0
        return {"logits": logits, "past_kv": None}


class ManualGeometryConstrainedSamplingTest(unittest.TestCase):
    def test_state_prefix_and_frame(self) -> None:
        state = GeometryGrammarState()
        self.assertEqual(state.allowed_token_strings(), ["MANUAL_GEOMETRY_V1"])
        for token in ["MANUAL_GEOMETRY_V1", "GEOMETRY_BLOCK", "ROLE_SUPPORT", "LABEL", "I_0", "GEOM_POLYGON_CODE"]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.phase, "frame_token")
        self.assertEqual(state.allowed_token_strings(), ["FRAME"])

    def test_constrained_sampler_masks_invalid_logits_and_generates_valid_geometry(self) -> None:
        config = ParseGraphTokenizerConfig(max_int=128, coord_bins=16, position_bins=16, scale_bins=16, angle_bins=16)
        vocab = build_token_vocabulary(config)
        model = _BadLogitModel(len(vocab), invalid_id=vocab["<EOS>"])
        sample = sample_geometry_constrained(
            model,
            vocab,
            max_new_tokens=128,
            temperature=0.0,
            top_k=None,
            constraint_config=GeometryConstrainedSamplerConfig(
                tokenizer_config=config,
                max_polygons=1,
                max_points_per_ring=3,
                max_holes_per_polygon=0,
            ),
            device=torch.device("cpu"),
        )

        result = validate_geometry_tokens(sample["tokens"], config=config)

        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(sample["hit_eos"])
        self.assertEqual(result["structure"]["polygon_count"], 1)
        self.assertEqual(result["structure"]["point_total"], 3)

    def test_illegal_step_records_error(self) -> None:
        state = GeometryGrammarState()
        self.assertFalse(state.step("<EOS>"))
        self.assertTrue(state.errors)


if __name__ == "__main__":
    unittest.main()
