from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_parse_graph_target_audit import (
    audit_manual_parse_graph_target,
    audit_manual_parse_graph_targets,
    iter_manual_parse_graph_target_paths,
)
from partition_gen.manual_parse_graph_visualization import local_to_world, polygon_world_rings


def _parse_graph_target() -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
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
                    "frame": {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
                    "geometry": {
                        "outer_local": points,
                        "holes_local": [],
                        "polygons_local": [{"outer_local": points, "holes_local": []}],
                    },
                },
                {
                    "id": "insert_group_0",
                    "role": "insert_object_group",
                    "label": 1,
                    "renderable": False,
                    "is_reference_only": False,
                    "geometry_model": "none",
                    "children": [],
                },
            ],
            "relations": [{"type": "inserted_in", "object": "insert_group_0", "container": "support_0"}],
            "residuals": [],
        },
        "metadata": {"placeholder_geometry": True},
    }


class ManualParseGraphTargetAuditTest(unittest.TestCase):
    def test_audit_manual_parse_graph_target_checks_geometry_and_tokenizers(self) -> None:
        row = audit_manual_parse_graph_target(_parse_graph_target())

        self.assertTrue(row["encodable"], row["errors"])
        self.assertEqual(row["missing_geometry_payload_count"], 0)
        self.assertGreater(row["old_token_count"], 0)
        self.assertGreater(row["compact_token_count"], 0)

    def test_audit_manual_parse_graph_targets_summarizes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            path.write_text(json.dumps(_parse_graph_target()), encoding="utf-8")

            payload = audit_manual_parse_graph_targets([path])

            self.assertEqual(payload["loaded_count"], 1)
            self.assertEqual(payload["encodable_count"], 1)
            self.assertEqual(payload["node_counts"]["max"], 2)

    def test_iter_manual_parse_graph_target_paths_reads_placeholder_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graphs" / "sample_000000.json"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text(json.dumps(_parse_graph_target()), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"output_path": str(graph_path.as_posix())}) + "\n", encoding="utf-8")

            self.assertEqual(iter_manual_parse_graph_target_paths(root), [graph_path])

    def test_visualization_geometry_helpers_transform_local_rings(self) -> None:
        node = _parse_graph_target()["parse_graph"]["nodes"][0]
        self.assertEqual(local_to_world([0.0, 0.0], node["frame"]), [128.0, 128.0])
        rings = list(polygon_world_rings(node))

        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0][0][0], [120.0, 120.0])


if __name__ == "__main__":
    unittest.main()
