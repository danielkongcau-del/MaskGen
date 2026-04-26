from __future__ import annotations

import math
import unittest

from shapely.geometry import Polygon

from partition_gen.operation_explainer import build_operation_explanation_payload
from partition_gen.operation_selector import select_operations_with_ortools
from partition_gen.operation_types import (
    OVERLAY_INSERT,
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
)


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
        "id": face_id,
        "label": label,
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
            "bbox_area": float((maxx - minx) * (maxy - miny)),
            "bbox_aspect_ratio": float(aspect),
            "perimeter": float(polygon.length),
            "compactness": _compactness(polygon),
            "convex_hull_area": float(polygon.convex_hull.area),
            "solidity": float(polygon.area / polygon.convex_hull.area) if polygon.convex_hull.area > 0 else 0.0,
            "oriented_bbox_width": float(min(maxx - minx, maxy - miny)),
            "oriented_bbox_height": float(max(maxx - minx, maxy - miny)),
            "oriented_aspect_ratio": float(aspect),
            "degree": int(degree),
            "shared_boundary_length": 0.0,
            "touches_border": False,
            "hole_count": len(holes),
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


def _adj(left: int, right: int, label_left: int, label_right: int, length: float = 10.0) -> dict:
    return {
        "faces": [left, right],
        "labels": [label_left, label_right],
        "arc_ids": [],
        "shared_length": float(length),
        "arc_count": 0,
    }


class OperationExplainerTests(unittest.TestCase):
    def test_residual_fallback_covers_single_face(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        payload = build_operation_explanation_payload(_evidence([face]), config=OperationExplainerConfig())

        self.assertEqual(payload["format"], "maskgen_operation_explanation_v1")
        self.assertTrue(payload["validation"]["all_faces_covered_exactly_once"])
        graph = payload["generator_target"]["parse_graph"]
        self.assertTrue(any(node["role"] == "residual_region" for node in graph["nodes"]))
        self.assertNotIn("program_sequence", payload["generator_target"])

    def test_overlay_insert_synthetic(self) -> None:
        insert_a_ring = [[4, 4], [6, 4], [6, 6], [4, 6]]
        insert_b_ring = [[12, 12], [14, 12], [14, 14], [12, 14]]
        support = _face(0, 0, [[0, 0], [20, 0], [20, 20], [0, 20]], [insert_a_ring, insert_b_ring], degree=2)
        insert_a = _face(1, 1, insert_a_ring, degree=1)
        insert_b = _face(2, 1, insert_b_ring, degree=1)
        evidence = _evidence(
            [support, insert_a, insert_b],
            [_adj(0, 1, 0, 1, 8.0), _adj(0, 2, 0, 1, 8.0)],
        )
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig())
        selected_types = {item["operation_type"] for item in payload["selected_operations"]}
        self.assertIn("OVERLAY_INSERT", selected_types)
        self.assertTrue(any(candidate["operation_type"] == "OVERLAY_INSERT" and "compression_gain" in candidate for candidate in payload["candidate_summary"]["top_candidates"]))
        graph = payload["generator_target"]["parse_graph"]
        roles = {node["role"] for node in graph["nodes"]}
        relation_types = {relation["type"] for relation in graph["relations"]}
        self.assertIn("support_region", roles)
        self.assertIn("insert_object_group", roles)
        self.assertIn("insert_object", roles)
        self.assertIn("inserted_in", relation_types)

    def test_divide_by_region_synthetic(self) -> None:
        left = _face(0, 0, [[0, 0], [9, 0], [9, 20], [0, 20]], degree=1)
        divider = _face(1, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2)
        right = _face(2, 0, [[11, 0], [20, 0], [20, 20], [11, 20]], degree=1)
        evidence = _evidence([left, divider, right], [_adj(0, 1, 0, 2, 20.0), _adj(1, 2, 2, 0, 20.0)])
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig())

        selected_types = {item["operation_type"] for item in payload["selected_operations"]}
        self.assertIn("DIVIDE_BY_REGION", selected_types)
        graph = payload["generator_target"]["parse_graph"]
        divider_nodes = [node for node in graph["nodes"] if node["role"] == "divider_region"]
        self.assertTrue(divider_nodes)
        self.assertNotEqual(divider_nodes[0]["geometry_model"], "skeleton_width_graph")

    def test_parallel_supports_synthetic(self) -> None:
        left = _face(0, 0, [[0, 0], [10, 0], [10, 20], [0, 20]], degree=1)
        right = _face(1, 1, [[10, 0], [20, 0], [20, 20], [10, 20]], degree=1)
        evidence = _evidence([left, right], [_adj(0, 1, 0, 1, 20.0)])
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig())
        relation_types = {relation["type"] for relation in payload["generator_target"]["parse_graph"]["relations"]}
        self.assertIn("adjacent_to", relation_types)

    def test_ortools_conflict_selection_covers_face_once(self) -> None:
        candidates = [
            OperationCandidate("a", OVERLAY_INSERT, "p", (1,), (), [], [], [], 10.0, 0.0, 10.0, {}, True),
            OperationCandidate("b", OVERLAY_INSERT, "p", (1,), (), [], [], [], 8.0, 0.0, 8.0, {}, True),
            OperationCandidate("r", RESIDUAL, "p", (1,), (), [], [], [], 5.0, 5.0, 0.0, {}, True),
        ]
        result = select_operations_with_ortools(candidates, [1], OperationExplainerConfig())
        self.assertEqual(result.selected_candidate_ids, ["a"])
        self.assertTrue(result.global_optimal)

    def test_same_label_can_have_different_local_roles(self) -> None:
        insert_ring = [[4, 4], [6, 4], [6, 6], [4, 6]]
        support = _face(0, 0, [[0, 0], [10, 0], [10, 10], [0, 10]], [insert_ring], degree=1)
        insert = _face(1, 1, insert_ring, degree=1)
        isolated_same_label = _face(2, 1, [[14, 14], [16, 14], [16, 16], [14, 16]], degree=0)
        evidence = _evidence([support, insert, isolated_same_label], [_adj(0, 1, 0, 1, 8.0)])
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig())
        graph = payload["generator_target"]["parse_graph"]
        label_one_roles = {node["role"] for node in graph["nodes"] if int(node.get("label", -1)) == 1}
        self.assertIn("insert_object", label_one_roles)
        self.assertIn("residual_region", label_one_roles)

    def test_nested_parse_graph_and_relation_refs_valid(self) -> None:
        left = _face(0, 0, [[0, 0], [10, 0], [10, 20], [0, 20]], degree=1)
        right = _face(1, 1, [[10, 0], [20, 0], [20, 20], [10, 20]], degree=1)
        payload = build_operation_explanation_payload(_evidence([left, right], [_adj(0, 1, 0, 1, 20.0)]), config=OperationExplainerConfig())
        target = payload["generator_target"]
        self.assertEqual(target["target_type"], "parse_graph")
        self.assertIn("parse_graph", target)
        self.assertNotIn("program_sequence", target)
        self.assertTrue(payload["validation"]["relation_reference_valid"])


if __name__ == "__main__":
    unittest.main()
