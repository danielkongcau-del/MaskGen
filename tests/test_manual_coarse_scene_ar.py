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
from scripts.attach_coarse_scene_true_shape_to_samples import _fit_frame_to_shape_bbox, _repair_adjacent_true_shape_frames


def _bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_intersection(left: list[float], right: list[float]) -> list[float] | None:
    min_x = max(float(left[0]), float(right[0]))
    min_y = max(float(left[1]), float(right[1]))
    max_x = min(float(left[2]), float(right[2]))
    max_y = min(float(left[3]), float(right[3]))
    if max_x <= min_x or max_y <= min_y:
        return None
    return [min_x, min_y, max_x, max_y]


def _bbox_gap(left: list[float], right: list[float]) -> float:
    import math

    dx = max(float(right[0]) - float(left[2]), float(left[0]) - float(right[2]), 0.0)
    dy = max(float(right[1]) - float(left[3]), float(left[1]) - float(right[3]), 0.0)
    return float(math.hypot(dx, dy))


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


def _two_adjacent_support_target() -> tuple[dict, list[dict]]:
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
                    "id": "support_1",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "support_1",
                },
                {
                    "id": "support_2",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "support_2",
                },
            ],
            "relations": [
                {"type": "adjacent_to", "faces": ["support_0", "support_1"]},
                {"type": "adjacent_to", "faces": ["support_0", "support_2"]},
            ],
            "residuals": [],
        },
    }
    geometry = [
        _geometry_target("support_0", "support_region", 0, [128.0, 128.0], 96.0),
        _geometry_target("support_1", "support_region", 0, [192.0, 128.0], 48.0),
        _geometry_target("support_2", "support_region", 0, [192.0, 128.0], 48.0),
    ]
    return target, geometry


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
        self.assertEqual(sum(1 for relation in relations if relation["type"] == "divides"), 1)
        self.assertIn({"type": "adjacent_to", "faces": ["support_0", "support_1"]}, relations)
        self.assertEqual(role_by_id["support_1"], "support_region")

    def test_actions_report_adjacent_and_node_anchors(self) -> None:
        actions, diagnostics = build_coarse_scene_actions(_topology_target(), _geometry_targets())
        by_source = {action["source_node_id"]: action for action in actions}

        self.assertEqual(by_source["support_1"]["action_token"], "ACTION_ADJACENT_SUPPORT")
        self.assertEqual(by_source["insert_0"]["anchor_mode"], "node")
        self.assertEqual(diagnostics["relation_histogram"]["REL_ADJACENT_TO"], 1)
        self.assertEqual(diagnostics["relation_histogram"]["REL_DIVIDES"], 1)

    def test_divider_and_adjacent_actions_use_single_anchor_tokens(self) -> None:
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets())
        for relation_token in ("REL_DIVIDES", "REL_ADJACENT_TO"):
            index = tokens.index(relation_token)
            self.assertNotEqual(tokens[index + 1], "COUNT")

    def test_relation_aware_decode_places_adjacent_and_divider_bboxes(self) -> None:
        tokens = encode_coarse_scene_target(_topology_target(), _geometry_targets())
        decoded = decode_coarse_scene_tokens_to_target(tokens)
        nodes = {node["id"]: node for node in decoded["parse_graph"]["nodes"]}

        support_0 = nodes["support_0"]["coarse_bbox"]
        support_1 = nodes["support_1"]["coarse_bbox"]
        divider = nodes["divider_0"]["coarse_bbox"]
        divide_target_id = next(relation["target"] for relation in decoded["parse_graph"]["relations"] if relation["type"] == "divides")
        divide_target = nodes[divide_target_id]["coarse_bbox"]

        adjacent_intersection = _bbox_intersection(support_0, support_1)
        adjacent_overlap_ratio = _bbox_area(adjacent_intersection) / max(1e-6, min(_bbox_area(support_0), _bbox_area(support_1)))
        divider_intersection = _bbox_intersection(divider, divide_target)
        divider_coverage = _bbox_area(divider_intersection) / max(1e-6, _bbox_area(divider))

        self.assertLessEqual(_bbox_gap(support_0, support_1), 4.0)
        self.assertLessEqual(adjacent_overlap_ratio, 0.1)
        self.assertGreaterEqual(divider_coverage, 0.3)
        self.assertLessEqual(_bbox_area(divider) / max(1e-6, _bbox_area(divide_target)), 1.25)

    def test_adjacent_decode_avoids_existing_support_collision(self) -> None:
        target, geometry = _two_adjacent_support_target()
        tokens = encode_coarse_scene_target(target, geometry)
        decoded = decode_coarse_scene_tokens_to_target(tokens)
        nodes = {node["id"]: node for node in decoded["parse_graph"]["nodes"]}

        support_0 = nodes["support_0"]["coarse_bbox"]
        support_1 = nodes["support_1"]["coarse_bbox"]
        support_2 = nodes["support_2"]["coarse_bbox"]
        sibling_intersection = _bbox_intersection(support_1, support_2)
        sibling_overlap_ratio = _bbox_area(sibling_intersection) / max(1e-6, min(_bbox_area(support_1), _bbox_area(support_2)))

        self.assertLessEqual(_bbox_gap(support_0, support_1), 4.0)
        self.assertLessEqual(_bbox_gap(support_0, support_2), 4.0)
        self.assertLessEqual(sibling_overlap_ratio, 0.1)

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

    def test_true_shape_fit_offsets_non_centered_local_bbox(self) -> None:
        frame = {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0}
        coarse_bbox = [10.0, 20.0, 50.0, 60.0]
        local_bbox = {"min_x": 1.0, "min_y": -1.0, "width": 2.0, "height": 4.0}

        fitted = _fit_frame_to_shape_bbox(frame, coarse_bbox, local_bbox, mode="contain")

        self.assertEqual(fitted["scale"], 10.0)
        self.assertEqual(fitted["origin"], [10.0, 30.0])

    def test_true_shape_adjacent_repair_shifts_child_subtree(self) -> None:
        nodes = [
            {
                "id": "support_0",
                "role": "support_region",
                "frame": {"origin": [5.0, 5.0], "scale": 10.0, "orientation": 0.0},
                "coarse_bbox": [0.0, 0.0, 10.0, 10.0],
                "true_shape_local_bbox": {"min_x": -0.5, "min_y": -0.5, "width": 1.0, "height": 1.0},
            },
            {
                "id": "support_1",
                "role": "support_region",
                "frame": {"origin": [105.0, 5.0], "scale": 10.0, "orientation": 0.0},
                "coarse_bbox": [100.0, 0.0, 110.0, 10.0],
                "true_shape_local_bbox": {"min_x": -0.5, "min_y": -0.5, "width": 1.0, "height": 1.0},
            },
            {
                "id": "insert_group_0",
                "role": "insert_object_group",
                "frame": {"origin": [105.0, 5.0], "scale": 4.0, "orientation": 0.0},
                "coarse_bbox": [103.0, 3.0, 107.0, 7.0],
            },
        ]
        relations = [
            {"type": "adjacent_to", "faces": ["support_0", "support_1"]},
            {"type": "inserted_in", "object": "insert_group_0", "container": "support_1"},
        ]

        summary = _repair_adjacent_true_shape_frames(nodes, relations)

        self.assertEqual(summary["repair_count"], 1)
        self.assertLessEqual(_bbox_gap(nodes[0]["coarse_bbox"], nodes[1]["coarse_bbox"]), 4.0)
        self.assertEqual(nodes[1]["frame"]["origin"], [15.0, 5.0])
        self.assertEqual(nodes[2]["coarse_bbox"], [13.0, 3.0, 17.0, 7.0])

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
