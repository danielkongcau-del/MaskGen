from __future__ import annotations

import unittest

from partition_gen.manual_parse_graph_relation_spatial_audit import (
    audit_manual_parse_graph_target_relation_spatial,
)


def _square_node(node_id: str, role: str, label: int, origin: list[float], scale: float) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "id": node_id,
        "role": role,
        "label": label,
        "renderable": True,
        "is_reference_only": False,
        "geometry_model": "polygon_code",
        "frame": {"origin": origin, "scale": scale, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _target(nodes: list[dict], relations: list[dict]) -> dict:
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": [256, 256],
        "parse_graph": {"nodes": nodes, "relations": relations, "residuals": []},
        "metadata": {"sample_index": 7},
    }


class ManualParseGraphRelationSpatialAuditTest(unittest.TestCase):
    def test_inserted_group_uses_contains_children_bbox(self) -> None:
        target = _target(
            [
                _square_node("support_0", "support_region", 0, [128.0, 128.0], 200.0),
                {
                    "id": "insert_group_0",
                    "role": "insert_object_group",
                    "label": 1,
                    "geometry_model": "none",
                    "renderable": False,
                },
                _square_node("insert_0", "insert_object", 1, [128.0, 128.0], 40.0),
            ],
            [
                {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
            ],
        )

        audit = audit_manual_parse_graph_target_relation_spatial(target)
        inserted = [row for row in audit["relations"] if row["relation_type"] == "inserted_in"][0]

        self.assertEqual(audit["failed_relation_pair_count"], 0)
        self.assertEqual(audit["passed_relation_pair_count"], 2)
        self.assertEqual(inserted["left_bbox_source"], "contains_children")
        self.assertTrue(inserted["left_center_in_right"])

    def test_adjacent_large_overlap_fails(self) -> None:
        target = _target(
            [
                _square_node("support_0", "support_region", 0, [80.0, 128.0], 100.0),
                _square_node("support_1", "support_region", 1, [100.0, 128.0], 100.0),
            ],
            [{"type": "adjacent_to", "faces": ["support_0", "support_1"]}],
        )

        audit = audit_manual_parse_graph_target_relation_spatial(target)
        relation = audit["relations"][0]

        self.assertEqual(audit["failed_relation_pair_count"], 1)
        self.assertFalse(relation["passed"])
        self.assertIn("adjacent_overlap_too_large", relation["failure_reasons"])
        self.assertGreater(relation["smaller_intersection_ratio"], 0.1)


if __name__ == "__main__":
    unittest.main()
