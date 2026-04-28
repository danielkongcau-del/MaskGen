from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_relative_layout_ar import (
    attach_relative_layout_frames_to_topology,
    build_relative_layout_sequence_rows,
    decode_relative_layout_tokens_to_target,
    encode_conditioned_relative_layout_target,
    encode_relative_layout_target,
    extract_relative_layout_tokens_from_conditioned,
    relative_layout_anchor_for_node,
    relative_layout_start_index,
    relative_layout_to_absolute_layout_target,
    validate_relative_layout_tokens,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary


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
                    "id": "adjacent_support_0",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "adjacent_support_0",
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
                {
                    "id": "divider_0",
                    "role": "divider_region",
                    "label": 2,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "divider_0",
                },
            ],
            "relations": [
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                {"type": "divides", "divider": "divider_0", "target": "insert_group_0"},
                {"type": "adjacent_to", "faces": ["support_0", "adjacent_support_0"]},
            ],
            "residuals": [],
        },
    }


def _geometry_target(source_node_id: str, role: str, label: int, origin: list[float], scale: float = 16.0) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id,
        "role": role,
        "label": label,
        "geometry_model": "polygon_code",
        "frame": {"origin": origin, "scale": scale, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _geometry_targets() -> list[dict]:
    return [
        _geometry_target("support_0", "support_region", 0, [128.0, 128.0], 64.0),
        _geometry_target("adjacent_support_0", "support_region", 0, [192.0, 128.0], 32.0),
        _geometry_target("insert_0", "insert_object", 1, [144.0, 112.0], 16.0),
        _geometry_target("divider_0", "divider_region", 2, [160.0, 128.0], 8.0),
    ]


def _write_split(tmpdir: str) -> Path:
    root = Path(tmpdir) / "split" / "train"
    topology_path = root / "topology" / "graphs" / "0.json"
    geometry_dir = root / "geometry" / "0"
    topology_path.parent.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
    geometry_paths = []
    for target in _geometry_targets():
        path = geometry_dir / f"{target['source_node_id']}.json"
        path.write_text(json.dumps(target), encoding="utf-8")
        geometry_paths.append(path)
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


class ManualRelativeLayoutARTest(unittest.TestCase):
    def test_anchor_rules_recurse_groups_and_ignore_adjacency(self) -> None:
        geometry_by_id = {target["source_node_id"]: target for target in _geometry_targets()}

        support_anchor = relative_layout_anchor_for_node(_topology_target(), 1, geometry_by_id)
        insert_anchor = relative_layout_anchor_for_node(_topology_target(), 3, geometry_by_id)
        divider_anchor = relative_layout_anchor_for_node(_topology_target(), 4, geometry_by_id)

        self.assertEqual(support_anchor["anchor_mode"], "global")
        self.assertEqual(insert_anchor["anchor_mode"], "node")
        self.assertEqual(insert_anchor["anchor_node_index"], 0)
        self.assertEqual(divider_anchor["anchor_mode"], "node")
        self.assertEqual(divider_anchor["anchor_node_index"], 0)

    def test_relative_layout_roundtrip_reconstructs_absolute_frames(self) -> None:
        tokens = encode_relative_layout_target(_topology_target(), _geometry_targets())
        decoded = decode_relative_layout_tokens_to_target(tokens)
        absolute = relative_layout_to_absolute_layout_target(decoded)
        frames_by_index = {item["node_index"]: item["frame"] for item in absolute["nodes"]}

        self.assertEqual(tokens[1], "MANUAL_REL_LAYOUT_V1")
        self.assertEqual([row["node_index"] for row in decoded["nodes"]], [0, 1, 3, 4])
        self.assertAlmostEqual(frames_by_index[3]["origin"][0], 144.0, delta=1.0)
        self.assertAlmostEqual(frames_by_index[3]["origin"][1], 112.0, delta=1.0)
        self.assertAlmostEqual(frames_by_index[4]["scale"], 8.0, delta=1.0)

    def test_conditioned_relative_layout_loss_starts_at_layout(self) -> None:
        tokens = encode_conditioned_relative_layout_target(_topology_target(), _geometry_targets())

        start = relative_layout_start_index(tokens)
        layout_tokens = extract_relative_layout_tokens_from_conditioned(tokens)

        self.assertEqual(tokens[start], "MANUAL_REL_LAYOUT_V1")
        self.assertEqual(layout_tokens[0], "<BOS>")
        self.assertEqual(layout_tokens[1], "MANUAL_REL_LAYOUT_V1")

    def test_validator_rejects_duplicate_and_invalid_anchor(self) -> None:
        valid = encode_relative_layout_target(_topology_target(), _geometry_targets())
        duplicate = list(valid)
        second_node_pos = valid.index("NODE", valid.index("NODE") + 1)
        duplicate[second_node_pos + 1] = "I_0"
        invalid_anchor = list(valid)
        anchor_pos = invalid_anchor.index("ANCHOR_NODE")
        invalid_anchor[anchor_pos + 1] = "I_99"

        self.assertTrue(validate_relative_layout_tokens(valid, topology_target=_topology_target())["valid"])
        self.assertFalse(validate_relative_layout_tokens(duplicate, topology_target=_topology_target())["valid"])
        self.assertFalse(validate_relative_layout_tokens(invalid_anchor, topology_target=_topology_target())["valid"])

    def test_build_rows_and_attach_preserves_shape_and_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            config = ParseGraphTokenizerConfig()
            rows, summary = build_relative_layout_sequence_rows(split_root, config=config, vocab=build_token_vocabulary(config))
            layout = decode_relative_layout_tokens_to_target(extract_relative_layout_tokens_from_conditioned(rows[0]["tokens"]))
            target, diagnostics = attach_relative_layout_frames_to_topology(
                _topology_target(),
                layout,
                geometry_by_node_id={target["source_node_id"]: target for target in _geometry_targets()},
            )
            nodes_by_id = {node["id"]: node for node in target["parse_graph"]["nodes"]}

            self.assertEqual(summary["written_layout"], 1)
            self.assertEqual(rows[0]["tokens"][rows[0]["loss_start_index"]], "MANUAL_REL_LAYOUT_V1")
            self.assertGreater(summary["anchor_mode_histogram"]["node"], 0)
            self.assertIn("geometry", nodes_by_id["insert_0"])
            self.assertEqual(nodes_by_id["insert_0"]["geometry"]["outer_local"][0], [-0.5, -0.5])
            self.assertEqual(target["parse_graph"]["relations"], _topology_target()["parse_graph"]["relations"])
            self.assertEqual(diagnostics["attached_geometry_count"], 4)


if __name__ == "__main__":
    unittest.main()
