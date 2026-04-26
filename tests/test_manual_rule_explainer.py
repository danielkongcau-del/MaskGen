from __future__ import annotations

import math
import unittest

from shapely.geometry import Polygon

from partition_gen.manual_rule_explainer import ManualRuleExplainerConfig, build_manual_rule_explanation_payload


def _compactness(polygon: Polygon) -> float:
    if polygon.length <= 1e-8:
        return 0.0
    return float(4.0 * math.pi * polygon.area / (polygon.length * polygon.length))


def _face(face_id: int, label: int, outer, holes=None, *, is_thin: bool = False, degree: int = 0) -> dict:
    holes = holes or []
    polygon = Polygon(outer, holes)
    minx, miny, maxx, maxy = polygon.bounds
    aspect = max(maxx - minx, maxy - miny) / max(min(maxx - minx, maxy - miny), 1e-8)
    if is_thin:
        aspect = max(aspect, 8.0)
    centroid = polygon.centroid
    return {
        "id": int(face_id),
        "label": int(label),
        "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
        "outer_arc_refs": [],
        "hole_arc_refs": [],
        "geometry": {
            "outer": [[float(x), float(y)] for x, y in outer],
            "holes": [[[float(x), float(y)] for x, y in ring] for ring in holes],
        },
        "features": {
            "area": float(polygon.area),
            "area_ratio": float(polygon.area / 400.0),
            "centroid": [float(centroid.x), float(centroid.y)],
            "bbox_width": float(maxx - minx),
            "bbox_height": float(maxy - miny),
            "perimeter": float(polygon.length),
            "compactness": _compactness(polygon),
            "oriented_aspect_ratio": float(aspect),
            "degree": int(degree),
            "is_thin": bool(is_thin),
            "is_compact": bool(_compactness(polygon) >= 0.45),
        },
        "convex_partition": {
            "backend": "synthetic",
            "valid": True,
            "piece_count": 1,
            "atoms": [
                {
                    "id": 0,
                    "type": "quad" if len(outer) == 4 else "convex",
                    "outer": [[float(x), float(y)] for x, y in outer],
                    "holes": [],
                    "vertex_count": len(outer),
                    "area": float(polygon.area),
                    "centroid": [float(centroid.x), float(centroid.y)],
                }
            ],
            "validation": {"is_valid": True},
        },
    }


def _adj(left: int, right: int, label_left: int, label_right: int, length: float = 10.0) -> dict:
    return {
        "faces": [int(left), int(right)],
        "labels": [int(label_left), int(label_right)],
        "arc_ids": [],
        "shared_length": float(length),
        "arc_count": 0,
    }


def _evidence(faces, adjacency=None) -> dict:
    adjacency = adjacency or []
    total_area = sum(float(face["features"]["area"]) for face in faces)
    return {
        "format": "maskgen_explanation_evidence_v1",
        "source_global_approx": "synthetic_global.json",
        "source_partition_graph": "synthetic_partition.json",
        "source_mask": "synthetic.png",
        "size": [20, 20],
        "faces": faces,
        "arcs": [],
        "adjacency": adjacency,
        "global_validation": {"is_valid": True},
        "evidence_validation": {
            "is_valid": True,
            "usable_for_explainer": True,
            "face_count": len(faces),
            "arc_count": 0,
            "adjacency_count": len(adjacency),
        },
        "statistics": {
            "image_area": 400.0,
            "total_face_area": float(total_area),
            "label_histogram": {},
        },
    }


def _role_spec(relations) -> dict:
    return {
        "format": "maskgen_role_spec_v1",
        "name": "synthetic_manual",
        "relations": relations,
        "defaults": {"unspecified_pair": "RESIDUAL"},
    }


class ManualRuleExplainerTests(unittest.TestCase):
    def test_manual_inserted_in_builds_parse_graph_without_selection(self) -> None:
        insert_ring = [[4, 4], [6, 4], [6, 6], [4, 6]]
        support = _face(0, 0, [[0, 0], [20, 0], [20, 20], [0, 20]], [insert_ring], degree=1)
        insert = _face(1, 1, insert_ring, degree=1)
        payload = build_manual_rule_explanation_payload(
            _evidence([support, insert], [_adj(0, 1, 0, 1, 8.0)]),
            _role_spec([{"subject_label": 1, "object_label": 0, "relation": "INSERTED_IN", "hard": True}]),
        )
        self.assertEqual(payload["format"], "maskgen_manual_rule_explanation_v1")
        self.assertEqual(payload["diagnostics"]["selection_method"], "manual_role_spec_direct")
        self.assertFalse(payload["diagnostics"]["uses_ortools"])
        graph = payload["generator_target"]["parse_graph"]
        roles = {node["role"] for node in graph["nodes"]}
        relation_types = {relation["type"] for relation in graph["relations"]}
        self.assertIn("support_region", roles)
        self.assertIn("insert_object_group", roles)
        self.assertIn("insert_object", roles)
        self.assertIn("inserted_in", relation_types)
        self.assertIn("contains", relation_types)
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 0)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])
        group_nodes = [node for node in graph["nodes"] if node["role"] == "insert_object_group"]
        self.assertTrue(group_nodes)
        self.assertEqual(group_nodes[0]["evidence"]["owned_face_ids"], [])
        self.assertEqual(group_nodes[0]["evidence"]["referenced_face_ids"], [1])
        self.assertIsNone(payload["generator_target"]["metadata"]["render_iou"])

    def test_manual_divides_builds_divider_and_support(self) -> None:
        left = _face(0, 0, [[0, 0], [9, 0], [9, 20], [0, 20]], degree=1)
        divider = _face(1, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2)
        right = _face(2, 0, [[11, 0], [20, 0], [20, 20], [11, 20]], degree=1)
        payload = build_manual_rule_explanation_payload(
            _evidence([left, divider, right], [_adj(0, 1, 0, 2, 20.0), _adj(1, 2, 2, 0, 20.0)]),
            _role_spec([{"subject_label": 2, "object_label": 0, "relation": "DIVIDES", "hard": True}]),
        )
        graph = payload["generator_target"]["parse_graph"]
        roles = {node["role"] for node in graph["nodes"]}
        relation_types = {relation["type"] for relation in graph["relations"]}
        self.assertIn("support_region", roles)
        self.assertIn("divider_region", roles)
        self.assertIn("divides", relation_types)
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 0)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])
        self.assertTrue(payload["validation"]["relation_reference_valid"])

    def test_unspecified_faces_become_residual(self) -> None:
        support = _face(0, 0, [[0, 0], [10, 0], [10, 10], [0, 10]])
        isolated = _face(1, 5, [[12, 12], [14, 12], [14, 14], [12, 14]])
        payload = build_manual_rule_explanation_payload(
            _evidence([support, isolated]),
            _role_spec([{"subject_label": 1, "object_label": 0, "relation": "INSERTED_IN", "hard": True}]),
        )
        graph = payload["generator_target"]["parse_graph"]
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 2)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])
        self.assertTrue(all(node["role"] == "residual_region" for node in graph["nodes"]))
        self.assertEqual(len(graph["residuals"]), 2)

    def test_manual_parallel_builds_adjacent_relation(self) -> None:
        left = _face(0, 0, [[0, 0], [10, 0], [10, 20], [0, 20]], degree=1)
        right = _face(1, 6, [[10, 0], [20, 0], [20, 20], [10, 20]], degree=1)
        payload = build_manual_rule_explanation_payload(
            _evidence([left, right], [_adj(0, 1, 0, 6, 20.0)]),
            _role_spec([{"subject_label": 0, "object_label": 6, "relation": "PARALLEL", "hard": True}]),
        )
        graph = payload["generator_target"]["parse_graph"]
        self.assertEqual({node["role"] for node in graph["nodes"]}, {"support_region"})
        self.assertIn("adjacent_to", {relation["type"] for relation in graph["relations"]})
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 0)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])

    def test_insert_group_can_be_relation_endpoint(self) -> None:
        insert_a = [[4, 4], [6, 4], [6, 6], [4, 6]]
        insert_b = [[14, 4], [16, 4], [16, 6], [14, 6]]
        support = _face(0, 0, [[0, 0], [20, 0], [20, 20], [0, 20]], [insert_a, insert_b], degree=2)
        building_a = _face(1, 1, insert_a, degree=2)
        building_b = _face(2, 1, insert_b, degree=2)
        road = _face(3, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2)
        evidence = _evidence(
            [support, building_a, building_b, road],
            [
                _adj(0, 1, 0, 1, 8.0),
                _adj(0, 2, 0, 1, 8.0),
                _adj(1, 3, 1, 2, 4.0),
                _adj(2, 3, 1, 2, 4.0),
            ],
        )
        payload = build_manual_rule_explanation_payload(
            evidence,
            _role_spec(
                [
                    {"subject_label": 1, "object_label": 0, "relation": "INSERTED_IN", "hard": True},
                    {"subject_label": 2, "object_label": 1, "relation": "DIVIDES", "hard": True},
                ]
            ),
        )
        graph = payload["generator_target"]["parse_graph"]
        group_ids = {node["id"] for node in graph["nodes"] if node["role"] == "insert_object_group"}
        self.assertTrue(group_ids)
        divides = [relation for relation in graph["relations"] if relation["type"] == "divides"]
        self.assertTrue(divides)
        self.assertIn(divides[0]["support"], group_ids)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])

    def test_nested_insert_can_use_reference_only_support(self) -> None:
        field = _face(0, 6, [[0, 0], [20, 0], [20, 20], [0, 20]], degree=1)
        woodland = _face(1, 5, [[2, 2], [12, 2], [12, 12], [2, 12]], degree=2)
        building = _face(2, 1, [[4, 4], [6, 4], [6, 6], [4, 6]], degree=1)
        evidence = _evidence(
            [field, woodland, building],
            [
                _adj(0, 1, 6, 5, 20.0),
                _adj(1, 2, 5, 1, 8.0),
            ],
        )
        payload = build_manual_rule_explanation_payload(
            evidence,
            _role_spec(
                [
                    {"subject_label": 5, "object_label": 6, "relation": "INSERTED_IN", "hard": True},
                    {"subject_label": 1, "object_label": 5, "relation": "INSERTED_IN", "hard": True},
                ]
            ),
        )
        graph = payload["generator_target"]["parse_graph"]
        reference_supports = [node for node in graph["nodes"] if node.get("is_reference_only")]
        self.assertTrue(reference_supports)
        self.assertEqual(reference_supports[0]["evidence"]["owned_face_ids"], [])
        self.assertEqual(reference_supports[0]["evidence"]["referenced_face_ids"], [1])
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 0)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])

    def test_single_label_image_defaults_to_support(self) -> None:
        woodland = _face(0, 5, [[0, 0], [20, 0], [20, 20], [0, 20]])
        payload = build_manual_rule_explanation_payload(_evidence([woodland]), _role_spec([]))
        graph = payload["generator_target"]["parse_graph"]
        self.assertEqual([node["role"] for node in graph["nodes"]], ["support_region"])
        self.assertEqual(payload["diagnostics"]["residual_face_count"], 0)
        self.assertEqual(payload["diagnostics"]["duplicate_owned_face_count"], 0)
        self.assertTrue(payload["validation"]["all_faces_owned_exactly_once"])


if __name__ == "__main__":
    unittest.main()
