import unittest
import tempfile
import json
from pathlib import Path

from partition_gen.manual_split_validator import validate_topology_geometry_split
from partition_gen.manual_target_split import build_topology_geometry_split_targets, merge_topology_geometry_targets
from partition_gen.parse_graph_compact_tokenizer import (
    compact_tokenizer_diagnostics,
    decode_topology_tokens_to_target,
    encode_generator_target_compact,
    encode_geometry_target,
    encode_topology_target,
)
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    build_token_vocabulary,
    encode_generator_target,
    tokens_to_ids,
)
from scripts.build_manual_split_dataset import build_split_dataset
from scripts.tokenize_manual_split_dataset import _tokenize_target


def _polygon_node(node_id, role="support_region", label=0, *, renderable=True, reference_only=False):
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "id": node_id,
        "role": role,
        "label": label,
        "renderable": renderable,
        "is_reference_only": reference_only,
        "frame": {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
        "geometry_model": "polygon_code",
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _target(nodes, relations=None):
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": [256, 256],
        "parse_graph": {
            "nodes": nodes,
            "relations": relations or [],
            "residuals": [],
        },
        "metadata": {},
    }


class ManualCompactTokenizerTests(unittest.TestCase):
    def test_compact_tokenizer_shorter_for_many_contains(self):
        support = _polygon_node("support_0", label=0)
        inserts = [_polygon_node(f"insert_{index}", role="insert_object", label=1) for index in range(20)]
        group = {
            "id": "insert_group_0",
            "role": "insert_object_group",
            "label": 1,
            "geometry_model": "none",
            "renderable": False,
            "children": [node["id"] for node in inserts],
            "count": len(inserts),
        }
        relations = [{"type": "contains", "parent": "insert_group_0", "child": node["id"]} for node in inserts]
        relations.append({"type": "inserted_in", "object": "insert_group_0", "container": "support_0"})
        target = _target([support, group, *inserts], relations)

        old_tokens = encode_generator_target(target)
        compact_tokens = encode_generator_target_compact(target)
        diagnostics = compact_tokenizer_diagnostics(target)

        self.assertLess(len(compact_tokens), len(old_tokens))
        self.assertEqual(diagnostics["contains_relation_count"], 20)
        self.assertEqual(diagnostics["skipped_contains_relation_count"], 20)
        self.assertNotIn("REL_CONTAINS", compact_tokens)
        self.assertIn("CHILDREN", compact_tokens)

    def test_compact_tokenizer_supports_container_target(self):
        target = _target(
            [
                {"id": "support_0", "role": "support_region", "label": 0, "geometry_model": "none"},
                {"id": "legacy_support_ignored", "role": "support_region", "label": 9, "geometry_model": "none"},
                {"id": "divider_0", "role": "divider_region", "label": 2, "geometry_model": "none"},
                {"id": "insert_group_0", "role": "insert_object_group", "label": 1, "geometry_model": "none", "children": []},
            ],
            [
                {
                    "type": "inserted_in",
                    "object": "insert_group_0",
                    "container": "support_0",
                    "support": "legacy_support_ignored",
                },
                {
                    "type": "divides",
                    "divider": "divider_0",
                    "target": "insert_group_0",
                    "support": "legacy_support_ignored",
                },
            ],
        )
        config = ParseGraphTokenizerConfig(max_int=512)
        tokens = encode_generator_target_compact(target, config=config)
        vocab = build_token_vocabulary(config)
        ids = tokens_to_ids(tokens, vocab)

        self.assertIn("REL_BLOCK_INSERTED_IN", tokens)
        self.assertIn("REL_BLOCK_DIVIDES", tokens)
        self.assertNotIn(vocab["<UNK>"], ids)

    def test_topology_geometry_split_removes_geometry_from_topology(self):
        target = _target([_polygon_node("support_0", label=0)])

        topology_target, geometry_targets, diagnostics = build_topology_geometry_split_targets(target, source_target="synthetic.json")
        topology_node = topology_target["parse_graph"]["nodes"][0]
        geometry_target = geometry_targets[0]

        self.assertNotIn("frame", topology_node)
        self.assertNotIn("geometry", topology_node)
        self.assertNotIn("atoms", topology_node)
        self.assertEqual(topology_node["geometry_ref"], "support_0")
        self.assertEqual(geometry_target["source_node_id"], "support_0")
        self.assertIn("frame", geometry_target)
        self.assertIn("geometry", geometry_target)
        self.assertEqual(diagnostics["geometry_target_count"], 1)

    def test_split_validator_accepts_valid_split(self):
        target = _target([_polygon_node("support_0", label=0)])
        topology_target, geometry_targets, _diagnostics = build_topology_geometry_split_targets(target)

        validation = validate_topology_geometry_split(topology_target, geometry_targets)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["missing_geometry_ref_ids"], [])
        self.assertEqual(validation["invalid_relation_refs"], [])

    def test_topology_geometry_split_skips_non_renderable_nodes(self):
        support = _polygon_node("support_0")
        reference = _polygon_node("support_ref_0", renderable=False, reference_only=True)
        group = {
            "id": "insert_group_0",
            "role": "insert_object_group",
            "label": 1,
            "geometry_model": "none",
            "renderable": False,
            "children": [],
        }
        topology_target, geometry_targets, diagnostics = build_topology_geometry_split_targets(_target([support, reference, group]))
        topology_by_id = {node["id"]: node for node in topology_target["parse_graph"]["nodes"]}

        self.assertEqual(len(geometry_targets), 1)
        self.assertEqual(geometry_targets[0]["source_node_id"], "support_0")
        self.assertEqual(topology_by_id["support_ref_0"]["geometry_model"], "none")
        self.assertNotIn("geometry_ref", topology_by_id["support_ref_0"])
        self.assertEqual(diagnostics["reference_only_count"], 1)

    def test_split_token_lengths_reduce_max_sequence(self):
        nodes = [_polygon_node(f"support_{index}", label=index % 3) for index in range(8)]
        target = _target(nodes)

        old_total = len(encode_generator_target(target))
        topology_target, geometry_targets, _diagnostics = build_topology_geometry_split_targets(target)
        topology_tokens = len(encode_topology_target(topology_target))
        geometry_max = max(len(encode_geometry_target(target)) for target in geometry_targets)

        self.assertGreater(old_total, max(topology_tokens, geometry_max))

    def test_decode_topology_tokens_to_target_roundtrips_structure(self):
        support = _polygon_node("support_0", label=0)
        insert = _polygon_node("insert_0", role="insert_object", label=1)
        group = {
            "id": "insert_group_0",
            "role": "insert_object_group",
            "label": 1,
            "geometry_model": "none",
            "renderable": False,
            "children": ["insert_0"],
        }
        target = _target(
            [support, group, insert],
            [
                {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                {"type": "adjacent_to", "faces": ["support_0", "insert_0"]},
            ],
        )
        topology_target, _geometry_targets, _diagnostics = build_topology_geometry_split_targets(target)
        tokens = encode_topology_target(topology_target)

        decoded = decode_topology_tokens_to_target(tokens)

        self.assertEqual(decoded["target_type"], "manual_parse_graph_topology_v1")
        self.assertEqual(encode_topology_target(decoded), tokens)
        nodes_by_id = {node["id"]: node for node in decoded["parse_graph"]["nodes"]}
        self.assertEqual(nodes_by_id["support_0"]["geometry_ref"], "support_0")
        self.assertEqual(nodes_by_id["insert_group_0"]["children"], ["insert_0"])
        self.assertTrue(any(relation["type"] == "contains" for relation in decoded["parse_graph"]["relations"]))

    def test_merge_topology_geometry_targets_restores_geometry_payload(self):
        target = _target([_polygon_node("support_0", label=0)])
        topology_target, geometry_targets, _diagnostics = build_topology_geometry_split_targets(target)

        merged = merge_topology_geometry_targets(topology_target, geometry_targets)

        node = merged["parse_graph"]["nodes"][0]
        self.assertEqual(merged["target_type"], "parse_graph")
        self.assertNotIn("geometry_ref", node)
        self.assertIn("frame", node)
        self.assertIn("geometry", node)
        self.assertEqual(merged["metadata"]["attached_geometry_count"], 1)

    def test_old_tokenizer_still_available(self):
        target = _target([_polygon_node("support_0")])
        tokens = encode_generator_target(target)

        self.assertEqual(tokens[1], "MANUAL_PARSE_GRAPH_V1")
        self.assertIn("POLYS", tokens)

    def test_build_split_dataset_writes_manifest(self):
        target = _target([_polygon_node("support_0")])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "manual" / "val" / "graphs" / "0.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(target), encoding="utf-8")
            output_root = root / "split"

            rows = build_split_dataset(
                [source],
                output_split_root=output_root / "val",
                target_root=root / "manual",
                split="val",
            )

            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(rows[0]["topology_path"]).exists())
            self.assertEqual(rows[0]["geometry_target_count"], 1)
            self.assertTrue(Path(rows[0]["geometry_paths"][0]).exists())
            self.assertTrue((output_root / "val" / "manifest.jsonl").exists())
            self.assertTrue((output_root / "val" / "summary.json").exists())

    def test_split_tokenize_helpers_encode_written_targets(self):
        target = _target([_polygon_node("support_0")])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "manual" / "val" / "graphs" / "0.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(target), encoding="utf-8")
            output_root = root / "split"
            rows = build_split_dataset(
                [source],
                output_split_root=output_root / "val",
                target_root=root / "manual",
                split="val",
            )

            topology_tokens = _tokenize_target(Path(rows[0]["topology_path"]), target_kind="topology", config=ParseGraphTokenizerConfig())
            geometry_tokens = _tokenize_target(Path(rows[0]["geometry_paths"][0]), target_kind="geometry", config=ParseGraphTokenizerConfig())

            self.assertEqual(topology_tokens[1], "MANUAL_TOPOLOGY_V1")
            self.assertEqual(geometry_tokens[1], "MANUAL_GEOMETRY_V1")


if __name__ == "__main__":
    unittest.main()
