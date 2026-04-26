import json
import math
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon

from partition_gen.manual_target_geometry_simplify import (
    ManualTargetSimplifyConfig,
    count_polygon_vertices,
    remove_near_collinear_points,
    simplify_manual_generator_target,
)
from partition_gen.manual_target_token_stats import analyze_manual_target_token_stats
from partition_gen.parse_graph_tokenizer import encode_generator_target
from scripts.benchmark_manual_target_simplification import run_benchmark


def _dense_rectangle(width=10.0, height=5.0, per_edge=25):
    points = []
    for index in range(per_edge):
        points.append([width * index / per_edge, 0.0])
    for index in range(per_edge):
        points.append([width, height * index / per_edge])
    for index in range(per_edge):
        points.append([width - width * index / per_edge, height])
    for index in range(per_edge):
        points.append([0.0, height - height * index / per_edge])
    return points


def _polygon_node(
    node_id,
    *,
    role="support_region",
    label=0,
    points=None,
    renderable=True,
    is_reference_only=False,
):
    points = points or _dense_rectangle()
    return {
        "id": node_id,
        "role": role,
        "label": label,
        "geometry_model": "polygon_code",
        "renderable": renderable,
        "is_reference_only": is_reference_only,
        "frame": {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
        "evidence": {"owned_face_ids": [1], "referenced_face_ids": []},
    }


def _target(nodes, relations=None):
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": [64, 64],
        "parse_graph": {
            "nodes": nodes,
            "relations": relations or [],
            "residuals": [],
        },
        "metadata": {},
    }


class ManualTargetGeometrySimplifyTests(unittest.TestCase):
    def test_remove_collinear_points_reduces_ring(self):
        ring = _dense_rectangle(per_edge=20)
        before_area = Polygon(ring).area

        reduced = remove_near_collinear_points(ring, eps=1e-6)

        self.assertLess(len(reduced), len(ring))
        self.assertGreaterEqual(len(reduced), 3)
        self.assertTrue(math.isclose(Polygon(reduced).area, before_area, rel_tol=1e-9, abs_tol=1e-9))

    def test_simplify_polygon_node_reduces_vertices(self):
        node = _polygon_node("support_0", role="support_region", label=7, points=_dense_rectangle(per_edge=30))
        original_vertices = count_polygon_vertices(node)

        simplified, diagnostics = simplify_manual_generator_target(
            _target([node]),
            config=ManualTargetSimplifyConfig(profile="light"),
        )
        simplified_node = simplified["parse_graph"]["nodes"][0]

        self.assertEqual(simplified_node["id"], "support_0")
        self.assertEqual(simplified_node["role"], "support_region")
        self.assertEqual(simplified_node["label"], 7)
        self.assertEqual(simplified_node["geometry_model"], "polygon_code")
        self.assertLess(count_polygon_vertices(simplified_node), original_vertices)
        self.assertGreater(diagnostics["simplified_node_count"], 0)

    def test_simplify_does_not_touch_non_renderable_nodes(self):
        support = _polygon_node("support_0", points=_dense_rectangle(per_edge=30))
        reference = _polygon_node(
            "context_0",
            points=_dense_rectangle(per_edge=30),
            renderable=False,
            is_reference_only=True,
        )
        insert_group = {
            "id": "insert_group_0",
            "role": "insert_object_group",
            "label": 1,
            "geometry_model": "none",
            "renderable": False,
            "children": [],
            "count": 0,
        }
        atoms_node = {
            "id": "residual_0",
            "role": "residual_region",
            "label": 2,
            "geometry_model": "convex_atoms",
            "renderable": True,
            "frame": {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0},
            "atoms": [
                {
                    "type": "triangle",
                    "vertex_count": 3,
                    "area": 0.5,
                    "outer_local": [[0, 0], [1, 0], [0, 1]],
                }
            ],
        }
        original_reference_geometry = json.loads(json.dumps(reference["geometry"]))
        original_atoms = json.loads(json.dumps(atoms_node["atoms"]))

        simplified, diagnostics = simplify_manual_generator_target(
            _target([support, reference, insert_group, atoms_node]),
            config=ManualTargetSimplifyConfig(profile="light"),
        )
        nodes = {node["id"]: node for node in simplified["parse_graph"]["nodes"]}

        self.assertEqual(nodes["context_0"]["geometry"], original_reference_geometry)
        self.assertEqual(nodes["residual_0"]["atoms"], original_atoms)
        self.assertEqual(nodes["insert_group_0"]["geometry_model"], "none")
        self.assertEqual(diagnostics["simplified_node_count"], 1)

    def test_simplify_invalid_fallback(self):
        tiny = [[0.0, 0.0], [1e-6, 0.0], [0.0, 1e-6]]
        node = _polygon_node("support_0", points=tiny)
        original_geometry = json.loads(json.dumps(node["geometry"]))

        simplified, diagnostics = simplify_manual_generator_target(
            _target([node]),
            config=ManualTargetSimplifyConfig(profile="light"),
        )

        self.assertEqual(simplified["parse_graph"]["nodes"][0]["geometry"], original_geometry)
        self.assertEqual(diagnostics["failed_node_count"], 1)
        self.assertEqual(diagnostics["invalid_geometry_count"], 1)

    def test_token_stats_matches_encoder_length(self):
        support = _polygon_node("support_0", label=0)
        group = {
            "id": "insert_group_0",
            "role": "insert_object_group",
            "label": 1,
            "geometry_model": "none",
            "renderable": False,
            "children": ["insert_0"],
            "count": 1,
        }
        insert = _polygon_node("insert_0", role="insert_object", label=1, points=_dense_rectangle(width=1, height=1, per_edge=4))
        target = _target(
            [support, group, insert],
            relations=[
                {"type": "inserted_in", "object": "insert_0", "container": "support_0"},
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
            ],
        )

        stats = analyze_manual_target_token_stats(target)

        self.assertEqual(stats["total_tokens"], len(encode_generator_target(target)))
        self.assertEqual(stats["attribution_gap"], 0)

    def test_benchmark_smoke(self):
        target = _target([_polygon_node("support_0", points=_dense_rectangle(per_edge=20))])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graphs" / "0.json"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text(json.dumps(target), encoding="utf-8")

            rows = run_benchmark([graph_path], profiles=["light"], output_root=root / "out")

        self.assertEqual(len(rows), 1)
        self.assertIn("original_total_tokens", rows[0])
        self.assertIn("simplified_total_tokens", rows[0])
        self.assertIn("token_reduction_ratio", rows[0])


if __name__ == "__main__":
    unittest.main()
