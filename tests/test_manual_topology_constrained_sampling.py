from __future__ import annotations

import unittest

import torch

from partition_gen.manual_topology_constrained_sampling import (
    TopologyConstrainedSamplerConfig,
    TopologyGrammarState,
    sample_topology_constrained,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary


class _BadLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, invalid_id: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"block_size": 128})()
        self.vocab_size = int(vocab_size)
        self.invalid_id = int(invalid_id)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        logits = self.anchor.new_zeros((batch, seq_len, self.vocab_size))
        logits[:, :, self.invalid_id] = 1000.0
        return {"logits": logits}


class ManualTopologyConstrainedSamplingTest(unittest.TestCase):
    def test_state_prefix_and_node_count(self) -> None:
        state = TopologyGrammarState(config=TopologyConstrainedSamplerConfig(max_nodes=4))
        self.assertEqual(state.allowed_token_strings(), ["MANUAL_TOPOLOGY_V1"])
        for token in ["MANUAL_TOPOLOGY_V1", "SIZE", "I_256", "I_256", "NODE_BLOCK"]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.phase, "node_count")
        self.assertEqual(state.allowed_token_strings(), ["I_1", "I_2", "I_3", "I_4"])

    def test_group_children_indices_are_bounded_by_declared_nodes(self) -> None:
        state = TopologyGrammarState(
            config=TopologyConstrainedSamplerConfig(
                max_nodes=4,
                max_children_per_group=8,
                max_label=6,
                enforce_semantics=False,
            )
        )
        for token in [
            "MANUAL_TOPOLOGY_V1",
            "SIZE",
            "I_256",
            "I_256",
            "NODE_BLOCK",
            "I_3",
            "NODE",
            "ROLE_SUPPORT",
            "I_0",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "NODE",
            "ROLE_INSERT_GROUP",
            "I_1",
            "I_0",
            "I_0",
            "GEOM_NONE",
            "I_0",
            "CHILDREN",
        ]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.phase, "child_count")
        self.assertEqual(state.allowed_token_strings(), ["I_0", "I_1", "I_2", "I_3"])
        self.assertTrue(state.step("I_2"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_0", "I_1", "I_2"])

    def test_relation_endpoint_indices_are_bounded_by_declared_nodes(self) -> None:
        state = TopologyGrammarState(
            config=TopologyConstrainedSamplerConfig(max_nodes=2, max_relation_pairs=3, enforce_semantics=False)
        )
        for token in [
            "MANUAL_TOPOLOGY_V1",
            "SIZE",
            "I_256",
            "I_256",
            "NODE_BLOCK",
            "I_1",
            "NODE",
            "ROLE_SUPPORT",
            "I_0",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "REL_BLOCK_INSERTED_IN",
            "I_1",
        ]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.phase, "pair_endpoint")
        self.assertEqual(state.allowed_token_strings(), ["I_0"])

    def test_group_children_reserve_future_insert_roles(self) -> None:
        state = TopologyGrammarState(
            config=TopologyConstrainedSamplerConfig(max_nodes=4, max_children_per_group=8, max_label=6)
        )
        for token in [
            "MANUAL_TOPOLOGY_V1",
            "SIZE",
            "I_256",
            "I_256",
            "NODE_BLOCK",
            "I_3",
            "NODE",
            "ROLE_SUPPORT",
            "I_0",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "NODE",
            "ROLE_INSERT_GROUP",
            "I_1",
            "I_0",
            "I_0",
            "GEOM_NONE",
            "I_0",
            "CHILDREN",
        ]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.phase, "child_count")
        self.assertEqual(state.allowed_token_strings(), ["I_1"])
        self.assertTrue(state.step("I_1"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_2"])
        self.assertTrue(state.step("I_2"), state.errors)
        for token in ["END_NODE", "NODE"]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["ROLE_INSERT"])

    def test_semantic_relation_endpoint_roles_and_duplicate_edges_are_constrained(self) -> None:
        state = TopologyGrammarState(config=TopologyConstrainedSamplerConfig(max_nodes=4, max_relation_pairs=8))
        for token in [
            "MANUAL_TOPOLOGY_V1",
            "SIZE",
            "I_256",
            "I_256",
            "NODE_BLOCK",
            "I_4",
            "NODE",
            "ROLE_SUPPORT",
            "I_0",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "NODE",
            "ROLE_INSERT_GROUP",
            "I_1",
            "I_0",
            "I_0",
            "GEOM_NONE",
            "I_0",
            "CHILDREN",
            "I_1",
            "I_2",
            "END_NODE",
            "NODE",
            "ROLE_INSERT",
            "I_1",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "NODE",
            "ROLE_DIVIDER",
            "I_2",
            "I_1",
            "I_0",
            "GEOM_POLYGON_CODE",
            "I_1",
            "END_NODE",
            "REL_BLOCK_INSERTED_IN",
        ]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_1"])
        self.assertTrue(state.step("I_1"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_1"])
        self.assertTrue(state.step("I_1"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_0"])
        for token in ["I_0", "END_BLOCK", "REL_BLOCK_DIVIDES", "I_1"]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_3"])
        self.assertTrue(state.step("I_3"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_0", "I_1"])
        for token in ["I_0", "END_BLOCK", "REL_BLOCK_ADJACENT_TO", "I_2", "I_0", "I_2"]:
            self.assertTrue(state.step(token), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_0", "I_1", "I_2"])
        self.assertTrue(state.step("I_0"), state.errors)
        self.assertEqual(state.allowed_token_strings(), ["I_1"])

    def test_constrained_sampler_masks_invalid_logits_and_generates_valid_tokens(self) -> None:
        vocab = build_token_vocabulary(ParseGraphTokenizerConfig())
        model = _BadLogitModel(len(vocab), invalid_id=vocab["<EOS>"])
        sample = sample_topology_constrained(
            model,
            vocab,
            max_new_tokens=128,
            temperature=0.0,
            top_k=None,
            constraint_config=TopologyConstrainedSamplerConfig(max_nodes=1, max_label=1, max_relation_pairs=0),
            device=torch.device("cpu"),
        )
        result = validate_topology_tokens(sample["tokens"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["semantic_valid"], result["semantic_errors"])
        self.assertTrue(sample["hit_eos"])
        self.assertEqual(result["node_count_declared"], 1)
        self.assertEqual(result["node_count_actual"], 1)

    def test_count_prior_biases_allowed_count_tokens(self) -> None:
        vocab = build_token_vocabulary(ParseGraphTokenizerConfig())
        model = _BadLogitModel(len(vocab), invalid_id=vocab["<EOS>"])
        sample = sample_topology_constrained(
            model,
            vocab,
            max_new_tokens=128,
            temperature=0.0,
            top_k=None,
            constraint_config=TopologyConstrainedSamplerConfig(max_nodes=2, max_label=1, max_relation_pairs=0),
            device=torch.device("cpu"),
            count_priors={"node_count": {1: 0.01, 2: 1.0}},
            count_prior_weight=10.0,
        )
        result = validate_topology_tokens(sample["tokens"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["semantic_valid"], result["semantic_errors"])
        self.assertEqual(result["node_count_declared"], 2)

    def test_complexity_level_biases_allowed_count_tokens(self) -> None:
        vocab = build_token_vocabulary(ParseGraphTokenizerConfig())
        model = _BadLogitModel(len(vocab), invalid_id=vocab["<EOS>"])
        sample = sample_topology_constrained(
            model,
            vocab,
            max_new_tokens=128,
            temperature=0.0,
            top_k=None,
            constraint_config=TopologyConstrainedSamplerConfig(max_nodes=2, max_label=1, max_relation_pairs=0),
            device=torch.device("cpu"),
            complexity_level=10.0,
        )
        result = validate_topology_tokens(sample["tokens"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["semantic_valid"], result["semantic_errors"])
        self.assertEqual(result["node_count_declared"], 2)
        self.assertTrue(sample["complexity_diagnostics"]["enabled"])

    def test_illegal_step_records_error(self) -> None:
        state = TopologyGrammarState()
        self.assertFalse(state.step("<EOS>"))
        self.assertFalse(state.diagnostics()["done"] is False)
        self.assertTrue(state.errors)


if __name__ == "__main__":
    unittest.main()
