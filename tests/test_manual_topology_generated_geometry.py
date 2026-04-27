from __future__ import annotations

import unittest

from partition_gen.manual_topology_generated_geometry import (
    attach_generated_geometry,
    build_generated_geometry_targets_from_sample_rows,
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


def _sampler(node: dict, _node_index: int) -> tuple[dict, dict]:
    target = _geometry_target(str(node["id"]), str(node["role"]), int(node["label"]))
    return target, {"valid": True, "length": 32, "hit_eos": True, "errors": []}


class ManualTopologyGeneratedGeometryTest(unittest.TestCase):
    def test_attach_generated_geometry_uses_sampler_outputs(self) -> None:
        merged, diagnostics = attach_generated_geometry(_topology_target(), _sampler)

        nodes_by_id = {node["id"]: node for node in merged["parse_graph"]["nodes"]}
        self.assertIn("geometry", nodes_by_id["support_0"])
        self.assertIn("geometry", nodes_by_id["insert_0"])
        self.assertEqual(diagnostics["geometry_request_count"], 2)
        self.assertEqual(diagnostics["geometry_valid_count"], 2)
        self.assertEqual(diagnostics["attached_geometry_count"], 2)
        self.assertEqual(diagnostics["attach_modes"], {"generated": 2})

    def test_build_generated_geometry_targets_skips_invalid_rows_by_default(self) -> None:
        valid_tokens = encode_topology_target(_topology_target())
        invalid_tokens = valid_tokens[:-1]

        targets, summary = build_generated_geometry_targets_from_sample_rows(
            [
                {"sample_index": 0, "tokens": valid_tokens},
                {"sample_index": 1, "tokens": invalid_tokens},
            ],
            _sampler,
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(summary["skipped_invalid_count"], 1)
        self.assertEqual(summary["attached_geometry_count"], 2)


if __name__ == "__main__":
    unittest.main()
