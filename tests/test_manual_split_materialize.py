from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_split_materialize import attach_true_geometry_to_topology, materialize_manual_split_targets


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
                    "id": "insert_0",
                    "role": "insert_object",
                    "label": 1,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "insert_0",
                },
            ],
            "relations": [{"type": "adjacent_to", "faces": ["support_0", "insert_0"]}],
            "residuals": [],
        },
    }


def _geometry_target(source_node_id: str, origin: list[float]) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id,
        "role": "support_region",
        "label": 0,
        "geometry_model": "polygon_code",
        "frame": {"origin": origin, "scale": 16.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _write_split(tmpdir: str) -> Path:
    root = Path(tmpdir) / "split" / "val"
    topology_path = root / "topology" / "graphs" / "0.json"
    geometry_dir = root / "geometry" / "0"
    topology_path.parent.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
    geometry_paths = [geometry_dir / "support_0.json", geometry_dir / "insert_0.json"]
    for path, target in zip(
        geometry_paths,
        [_geometry_target("support_0", [128.0, 128.0]), _geometry_target("insert_0", [96.0, 96.0])],
    ):
        path.write_text(json.dumps(target), encoding="utf-8")
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "stem": "0",
                "topology_path": str(topology_path.as_posix()),
                "geometry_paths": [str(path.as_posix()) for path in geometry_paths],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


class ManualSplitMaterializeTest(unittest.TestCase):
    def test_attach_true_geometry_preserves_relations_and_removes_refs(self) -> None:
        target, diagnostics = attach_true_geometry_to_topology(
            _topology_target(),
            [_geometry_target("support_0", [128.0, 128.0]), _geometry_target("insert_0", [96.0, 96.0])],
        )

        nodes_by_id = {node["id"]: node for node in target["parse_graph"]["nodes"]}
        self.assertEqual(diagnostics["attached_geometry_count"], 2)
        self.assertEqual(diagnostics["missing_geometry_count"], 0)
        self.assertIn("geometry", nodes_by_id["support_0"])
        self.assertEqual(nodes_by_id["insert_0"]["frame"]["origin"], [96.0, 96.0])
        self.assertNotIn("geometry_ref", nodes_by_id["support_0"])
        self.assertEqual(target["parse_graph"]["relations"], _topology_target()["parse_graph"]["relations"])

    def test_materialize_manual_split_targets_loads_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)

            targets = materialize_manual_split_targets(split_root)

            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0]["metadata"]["attached_geometry_count"], 2)
            self.assertTrue(targets[0]["metadata"]["materialized_from_manual_split"])


if __name__ == "__main__":
    unittest.main()
