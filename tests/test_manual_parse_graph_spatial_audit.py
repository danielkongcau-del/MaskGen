from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_parse_graph_spatial_audit import (
    audit_manual_parse_graph_target_spatial,
    audit_manual_parse_graph_targets_spatial,
)


def _polygon_node(node_id: str, *, origin: list[float], scale: float = 16.0) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "id": node_id,
        "role": "support_region",
        "label": 0,
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


def _target(nodes: list[dict]) -> dict:
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": [256, 256],
        "parse_graph": {"nodes": nodes, "relations": [], "residuals": []},
        "metadata": {},
    }


class ManualParseGraphSpatialAuditTest(unittest.TestCase):
    def test_spatial_audit_counts_visible_and_corner_nodes(self) -> None:
        target = _target(
            [
                _polygon_node("centered", origin=[128.0, 128.0], scale=16.0),
                _polygon_node("corner", origin=[0.0, 0.0], scale=8.0),
                _polygon_node("outside", origin=[320.0, 320.0], scale=8.0),
            ]
        )

        row = audit_manual_parse_graph_target_spatial(target, edge_margin=16.0)

        self.assertEqual(row["renderable_polygon_node_count"], 3)
        self.assertEqual(row["visible_polygon_node_count"], 2)
        self.assertEqual(row["bbox_intersects_canvas_count"], 2)
        self.assertEqual(row["origin_corner_count"], 1)
        self.assertEqual(row["origin_corner_histogram"], {"top_left": 1})
        self.assertEqual(row["invisible_polygon_node_count"], 1)

    def test_spatial_audit_summarizes_multiple_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            path.write_text(json.dumps(_target([_polygon_node("centered", origin=[128.0, 128.0])])), encoding="utf-8")

            payload = audit_manual_parse_graph_targets_spatial([path])

            self.assertEqual(payload["loaded_count"], 1)
            self.assertEqual(payload["renderable_polygon_node_count"], 1)
            self.assertEqual(payload["visible_polygon_node_count"], 1)
            self.assertEqual(payload["origin_x_stats"]["median"], 128.0)


if __name__ == "__main__":
    unittest.main()
