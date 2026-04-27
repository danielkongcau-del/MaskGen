from __future__ import annotations

import unittest

from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


class ManualTopologySampleValidationTest(unittest.TestCase):
    def make_topology_target(self) -> dict:
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
                    {
                        "type": "inserted_in",
                        "object": "insert_group_0",
                        "container": "support_0",
                        "support": "support_0",
                    },
                    {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                    {"type": "adjacent_to", "faces": ["support_0", "insert_0"]},
                ],
                "residuals": [],
            },
        }

    def test_valid_topology_tokens(self) -> None:
        tokens = encode_topology_target(self.make_topology_target(), config=ParseGraphTokenizerConfig(max_int=128))
        result = validate_topology_tokens(tokens)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["node_count_declared"], 3)
        self.assertEqual(result["node_count_actual"], 3)

    def test_detects_node_count_mismatch(self) -> None:
        tokens = encode_topology_target(self.make_topology_target(), config=ParseGraphTokenizerConfig(max_int=128))
        node_block_index = tokens.index("NODE_BLOCK")
        tokens[node_block_index + 1] = "I_4"
        result = validate_topology_tokens(tokens)
        self.assertFalse(result["valid"])
        self.assertTrue(any("expected_NODE" in error for error in result["errors"]))

    def test_detects_endpoint_out_of_range(self) -> None:
        tokens = encode_topology_target(self.make_topology_target(), config=ParseGraphTokenizerConfig(max_int=128))
        block_index = tokens.index("REL_BLOCK_INSERTED_IN")
        tokens[block_index + 2] = "I_99"
        result = validate_topology_tokens(tokens)
        self.assertFalse(result["valid"])
        self.assertTrue(any("endpoint_out_of_range" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
