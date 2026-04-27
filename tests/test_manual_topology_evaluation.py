from __future__ import annotations

import unittest

from partition_gen.manual_topology_evaluation import (
    evaluate_topology_sample_rows,
    parse_topology_structure,
    score_topology_structure,
    topology_structure_targets_from_summary,
)
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


class ManualTopologyEvaluationTest(unittest.TestCase):
    def make_topology_tokens(self) -> list[str]:
        target = {
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
                    },
                    {
                        "id": "insert_group_0",
                        "role": "insert_object_group",
                        "label": 1,
                        "renderable": False,
                        "is_reference_only": False,
                        "geometry_model": "none",
                        "children": ["insert_0"],
                    },
                    {
                        "id": "insert_0",
                        "role": "insert_object",
                        "label": 1,
                        "renderable": True,
                        "is_reference_only": False,
                        "geometry_model": "polygon_code",
                        "geometry_ref": "insert_0",
                    },
                ],
                "relations": [
                    {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                    {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                    {"type": "adjacent_to", "faces": ["support_0", "insert_0"]},
                ],
                "residuals": [],
            },
        }
        return encode_topology_target(target, config=ParseGraphTokenizerConfig(max_int=128))

    def test_parse_topology_structure_counts_nodes_and_relations(self) -> None:
        parsed = parse_topology_structure(self.make_topology_tokens())
        self.assertEqual(parsed["node_count"], 3)
        self.assertEqual(parsed["roles"]["ROLE_INSERT_GROUP"], 1)
        self.assertEqual(parsed["labels"]["1"], 2)
        self.assertEqual(parsed["insert_group_child_counts"], [1])
        self.assertEqual(parsed["relation_counts"]["REL_BLOCK_INSERTED_IN"], 1)
        self.assertEqual(parsed["relation_counts"]["REL_BLOCK_ADJACENT_TO"], 1)
        self.assertEqual(parsed["relation_counts"]["REL_BLOCK_OTHER"], 0)

    def test_evaluate_topology_sample_rows_summarizes_valid_and_invalid(self) -> None:
        valid_tokens = self.make_topology_tokens()
        invalid_tokens = list(valid_tokens)
        invalid_tokens[invalid_tokens.index("NODE_BLOCK") + 1] = "I_4"
        summary = evaluate_topology_sample_rows(
            [
                {"sample_index": 0, "tokens": valid_tokens, "length": len(valid_tokens)},
                {"sample_index": 1, "tokens": invalid_tokens, "length": len(invalid_tokens)},
            ]
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(summary["valid_rate"], 0.5)
        self.assertEqual(summary["semantic_valid_count"], 1)
        self.assertEqual(summary["semantic_valid_rate"], 0.5)
        self.assertEqual(summary["node_counts"]["mean"], 3.0)
        self.assertEqual(summary["relation_mean_per_valid_sample"]["REL_BLOCK_INSERTED_IN"], 1.0)
        self.assertTrue(summary["failure_reason_histogram"])

    def test_evaluate_topology_sample_rows_summarizes_semantic_invalid(self) -> None:
        semantic_invalid = self.make_topology_tokens()
        child_count_index = semantic_invalid.index("CHILDREN") + 1
        semantic_invalid[child_count_index + 1] = "I_0"
        summary = evaluate_topology_sample_rows(
            [{"sample_index": 0, "tokens": semantic_invalid, "length": len(semantic_invalid)}]
        )

        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(summary["valid_rate"], 1.0)
        self.assertEqual(summary["semantic_valid_count"], 0)
        self.assertEqual(summary["semantic_valid_rate"], 0.0)
        self.assertTrue(summary["semantic_failure_reason_histogram"])

    def test_score_topology_structure_penalizes_relative_distribution_error(self) -> None:
        summary = evaluate_topology_sample_rows(
            [{"sample_index": 0, "tokens": self.make_topology_tokens(), "length": len(self.make_topology_tokens())}]
        )
        exact_targets = topology_structure_targets_from_summary(summary)
        exact_score = score_topology_structure(summary, exact_targets)
        shifted_score = score_topology_structure(summary, {**exact_targets, "node_count_mean": 6.0})

        self.assertEqual(exact_score["score"], 1.0)
        self.assertLess(shifted_score["score"], exact_score["score"])
        self.assertIn("node_count_mean", shifted_score["metric_relative_errors"])


if __name__ == "__main__":
    unittest.main()
