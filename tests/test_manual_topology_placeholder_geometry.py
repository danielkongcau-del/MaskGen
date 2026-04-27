from __future__ import annotations

import unittest

from partition_gen.manual_topology_placeholder_geometry import (
    GeometryPlaceholderLibrary,
    attach_placeholder_geometry,
    build_placeholder_targets_from_sample_rows,
    decode_topology_tokens_to_target,
)
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target


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


def _geometry_target(source_node_id: str, role: str, label: int) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id,
        "role": role,
        "label": label,
        "geometry_model": "polygon_code",
        "frame": {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


class ManualTopologyPlaceholderGeometryTest(unittest.TestCase):
    def test_decode_topology_tokens_to_target_roundtrips_tokens(self) -> None:
        target = _topology_target()
        tokens = encode_topology_target(target)

        decoded = decode_topology_tokens_to_target(tokens)

        self.assertEqual(decoded["target_type"], "manual_parse_graph_topology_v1")
        self.assertEqual(encode_topology_target(decoded), tokens)
        nodes_by_id = {node["id"]: node for node in decoded["parse_graph"]["nodes"]}
        self.assertEqual(nodes_by_id["insert_group_0"]["children"], ["insert_0"])

    def test_attach_placeholder_geometry_uses_exact_matches(self) -> None:
        target = decode_topology_tokens_to_target(encode_topology_target(_topology_target()))
        library = GeometryPlaceholderLibrary(
            [
                _geometry_target("support_src", "support_region", 0),
                _geometry_target("insert_src", "insert_object", 1),
            ],
            seed=1,
        )

        merged, diagnostics = attach_placeholder_geometry(target, library)

        nodes_by_id = {node["id"]: node for node in merged["parse_graph"]["nodes"]}
        self.assertIn("geometry", nodes_by_id["support_0"])
        self.assertIn("geometry", nodes_by_id["insert_0"])
        self.assertEqual(diagnostics["attached_geometry_count"], 2)
        self.assertEqual(diagnostics["attach_modes"], {"exact": 2})

    def test_build_placeholder_targets_skips_invalid_rows_by_default(self) -> None:
        valid_tokens = encode_topology_target(_topology_target())
        invalid_tokens = valid_tokens[:-1]
        library = GeometryPlaceholderLibrary([_geometry_target("support_src", "support_region", 0)], seed=1)

        targets, summary = build_placeholder_targets_from_sample_rows(
            [
                {"sample_index": 0, "tokens": valid_tokens},
                {"sample_index": 1, "tokens": invalid_tokens},
            ],
            library,
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(summary["skipped_invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
