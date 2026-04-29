from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_coarse_scene_ar import (
    CoarseSceneGrammarState,
    build_coarse_scene_actions,
    build_coarse_scene_sequence_rows,
    decode_coarse_scene_tokens_to_target,
    encode_coarse_scene_target,
    parent_first_node_order,
    validate_coarse_scene_tokens,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary, int_token


def _topology_target() -> dict:
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": [256, 256],
        "parse_graph": {
            "nodes": [
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
                    "id": "insert_group_0",
                    "role": "insert_object_group",
                    "label": 1,
                    "renderable": False,
                    "is_reference_only": False,
                    "geometry_model": "none",
                    "children": ["insert_0"],
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
                    "id": "support_1",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "support_1",
                },
            ],
            "relations": [
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                {"type": "divides", "divider": "divider_0", "target": "insert_group_0"},
                {"type": "divides", "divider": "divider_0", "target": "support_0"},
                {"type": "adjacent_to", "faces": ["support_0", "support_1"]},
            ],
            "residuals": [],
        },
    }


def _geometry_target(source_node_id: str, role: str, label: int, origin: list[float], scale: float) -> dict:
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
        _geometry_target("insert_0", "insert_object", 1, [128.0, 128.0], 24.0),
        _geometry_target("divider_0", "divider_region", 2, [128.0, 128.0], 8.0),
        _geometry_target("support_0", "support_region", 0, [128.0, 128.0], 96.0),
        _geometry_target("support_1", "support_region", 0, [192.0, 128.0], 48.0),
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


class ManualCoarseSceneARTest(unittest.TestCase):
    def test_parent_first_order_places_container_before_children(self) -> None:
        order, diagnostics = parent_first_node_order(_topology_target())
        nodes = _topology_target()["parse_graph"]["nodes"]
        ordered_ids = [nodes[index]["id"] for index in order]

        self.assertLess(ordered_ids.index("support_0"), ordered_ids.index("insert_group_0"))
        self.assertLess(ordered_ids.index("insert_group_0"), ordered_ids.index("insert_0"))
        self.assertLess(ordered_ids.index("insert_group_0"), ordered_ids.index("divider_0"))
        self.assertEqual(diagnostics["forward_reference_count"], 0)

    def test_encode_decode_roundtrip_preserves_parent_first_relations(self) -> None:
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets())
        decoded = decode_coarse_scene_tokens_to_target(tokens)
        nodes = decoded["parse_graph"]["nodes"]
        relations = decoded["parse_graph"]["relations"]
        role_by_id = {node["id"]: node["role"] for node in nodes}

        self.assertEqual(tokens[1], "MANUAL_COARSE_SCENE_V1")
        self.assertEqual(nodes[0]["role"], "support_region")
        self.assertTrue(validate_coarse_scene_tokens(tokens)["valid"])
        self.assertIn({"type": "inserted_in", "object": "insert_group_0", "container": "support_0"}, relations)
        self.assertIn({"type": "contains", "parent": "insert_group_0", "child": "insert_0"}, relations)
        self.assertIn({"type": "divides", "divider": "divider_0", "target": "insert_group_0"}, relations)
        self.assertIn({"type": "divides", "divider": "divider_0", "target": "support_0"}, relations)
        self.assertIn({"type": "adjacent_to", "faces": ["support_0", "support_1"]}, relations)
        self.assertEqual(role_by_id["support_1"], "support_region")

    def test_actions_report_adjacent_and_node_anchors(self) -> None:
        actions, diagnostics = build_coarse_scene_actions(_topology_target(), _geometry_targets())
        by_source = {action["source_node_id"]: action for action in actions}

        self.assertEqual(by_source["support_1"]["action_token"], "ACTION_ADJACENT_SUPPORT")
        self.assertEqual(by_source["insert_0"]["anchor_mode"], "node")
        self.assertEqual(diagnostics["relation_histogram"]["REL_ADJACENT_TO"], 1)
        self.assertEqual(diagnostics["relation_histogram"]["REL_DIVIDES"], 2)

    def test_validator_rejects_forward_anchor(self) -> None:
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets())
        invalid = list(tokens)
        first_rel = invalid.index("REL_INSERTED_IN")
        invalid[first_rel + 1] = int_token(99, config=ParseGraphTokenizerConfig())

        self.assertFalse(validate_coarse_scene_tokens(invalid)["valid"])

    def test_validator_rejects_insert_anchored_to_support(self) -> None:
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets())
        invalid = list(tokens)
        contains = invalid.index("REL_CONTAINS")
        invalid[contains + 1] = int_token(0, config=ParseGraphTokenizerConfig())
        anchor = invalid.index("ANCHOR_NODE", contains)
        invalid[anchor + 1] = int_token(0, config=ParseGraphTokenizerConfig())

        validation = validate_coarse_scene_tokens(invalid)

        self.assertFalse(validation["valid"])
        self.assertIn("Insert object anchor", validation["errors"][0])

    def test_coarse_bbox_roundtrip_is_within_one_bin(self) -> None:
        config = ParseGraphTokenizerConfig(coarse_grid_bins=8, coarse_size_bins=8)
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets(), config=config)
        decoded = decode_coarse_scene_tokens_to_target(tokens, config=config)
        support = decoded["parse_graph"]["nodes"][0]

        self.assertAlmostEqual(support["frame"]["origin"][0], 128.0, delta=256.0 / 7.0)
        self.assertAlmostEqual(support["frame"]["origin"][1], 128.0, delta=256.0 / 7.0)

    def test_build_rows_writes_summary_and_dataset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            config = ParseGraphTokenizerConfig()
            rows, summary = build_coarse_scene_sequence_rows(split_root, config=config, vocab=build_token_vocabulary(config))

            self.assertEqual(summary["written_coarse_scene"], 1)
            self.assertEqual(summary["forward_reference_count"], 0)
            self.assertEqual(rows[0]["tokens"][rows[0]["loss_start_index"]], "MANUAL_COARSE_SCENE_V1")

    def test_grammar_masks_future_and_wrong_role_anchors(self) -> None:
        state = CoarseSceneGrammarState()

        self.assertEqual(state.allowed_token_strings(), ["MANUAL_COARSE_SCENE_V1"])
        state.step("MANUAL_COARSE_SCENE_V1")
        state.step("SIZE")
        state.step("I_256")
        state.step("I_256")
        state.step("COUNT")
        state.step("I_2")
        self.assertEqual(state.allowed_token_strings(), ["ACTION_SUPPORT"])
        state.step("ACTION_SUPPORT")
        state.step("ROLE_SUPPORT")
        state.step("LABEL")
        state.step("I_0")
        state.step("GEOM_POLYGON_CODE")
        state.step("ANCHOR_GLOBAL")
        state.step("FRAME_ABS_COARSE")
        for _ in range(6):
            state.step("Q_0")
        state.step("END_ACTION")

        self.assertIn("ACTION_INSERT_GROUP", state.allowed_token_strings())
        self.assertNotIn("ACTION_INSERT", state.allowed_token_strings())


if __name__ == "__main__":
    unittest.main()
