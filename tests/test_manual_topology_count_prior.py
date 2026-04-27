from __future__ import annotations

import unittest

from partition_gen.manual_topology_count_prior import topology_count_prior_from_rows
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


class ManualTopologyCountPriorTest(unittest.TestCase):
    def make_tokens(self) -> list[str]:
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

    def test_topology_count_prior_from_rows_counts_target_distributions(self) -> None:
        payload = topology_count_prior_from_rows([{"tokens": self.make_tokens()}], source="unit")

        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["valid_count"], 1)
        self.assertEqual(payload["semantic_valid_count"], 1)
        self.assertEqual(payload["histograms"]["node_count"], {"3": 1})
        self.assertEqual(payload["histograms"]["child_count"], {"1": 1})
        self.assertEqual(payload["histograms"]["REL_BLOCK_DIVIDES"], {"0": 1})
        self.assertEqual(payload["histograms"]["REL_BLOCK_ADJACENT_TO"], {"1": 1})
        self.assertEqual(payload["priors"]["node_count"][3], 1.0)


if __name__ == "__main__":
    unittest.main()
