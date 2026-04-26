from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon

from partition_gen.operation_candidates import propose_operation_candidates_with_diagnostics
from partition_gen.operation_code_length import convex_atoms_code_length, polygon_code_length_from_payload
from partition_gen.operation_explainer import build_operation_explanation_payload
from partition_gen.operation_patches import build_operation_patches
from partition_gen.operation_selector import select_operations_with_ortools
from partition_gen.operation_types import (
    DIVIDE_BY_REGION,
    OVERLAY_INSERT,
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
)
from scripts.benchmark_operation_explainer import run_benchmark


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
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig(cost_profile="heuristic_v1"))
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
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig(cost_profile="heuristic_v1"))

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
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig(cost_profile="heuristic_v1"))
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
        insert_ring_b = [[7, 7], [9, 7], [9, 9], [7, 9]]
        support = _face(0, 0, [[0, 0], [12, 0], [12, 12], [0, 12]], [insert_ring, insert_ring_b], degree=2)
        insert = _face(1, 1, insert_ring, degree=1)
        insert_b = _face(2, 1, insert_ring_b, degree=1)
        isolated_same_label = _face(3, 1, [[14, 14], [16, 14], [16, 16], [14, 16]], degree=0)
        evidence = _evidence([support, insert, insert_b, isolated_same_label], [_adj(0, 1, 0, 1, 8.0), _adj(0, 2, 0, 1, 8.0)])
        payload = build_operation_explanation_payload(evidence, config=OperationExplainerConfig(cost_profile="heuristic_v1"))
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

    def test_operation_cost_includes_object_nodes(self) -> None:
        insert_ring = [[4, 4], [6, 4], [6, 6], [4, 6]]
        support = _face(0, 0, [[0, 0], [20, 0], [20, 20], [0, 20]], [insert_ring], degree=1)
        insert = _face(1, 1, insert_ring, degree=1)
        payload = build_operation_explanation_payload(
            _evidence([support, insert], [_adj(0, 1, 0, 1, 8.0)]),
            config=OperationExplainerConfig(cost_profile="heuristic_v1"),
        )
        overlay = next(item for item in payload["candidate_summary"]["top_candidates"] if item["operation_type"] == "OVERLAY_INSERT")
        operation = overlay["cost_breakdown"]["operation"]
        self.assertIn("node_object", operation)
        self.assertGreater(operation["node_object"], 0.0)
        self.assertGreaterEqual(operation["node_count"], 3)
        self.assertGreaterEqual(operation["group_node_count"], 1)

    def test_false_cover_penalty_detects_external_overlap(self) -> None:
        left = _face(0, 0, [[0, 0], [2, 0], [2, 2], [0, 2]], degree=1)
        right = _face(1, 0, [[18, 18], [20, 18], [20, 20], [18, 20]], degree=1)
        divider = _face(2, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2)
        external = _face(3, 3, [[8, 8], [12, 8], [12, 12], [8, 12]], degree=0)
        evidence = _evidence([left, right, divider, external], [_adj(0, 2, 0, 2, 2.0), _adj(1, 2, 0, 2, 2.0)])
        payload = build_operation_explanation_payload(
            evidence,
            config=OperationExplainerConfig(
                cost_profile="heuristic_v1",
                enable_visible_union=False,
                enable_union_with_divider=False,
                enable_convex_hull_fill=True,
                false_cover_ratio_invalid=1.0,
                hard_invalid_false_cover_ratio=1.0,
            ),
        )
        divider_candidates = [item for item in payload["candidate_summary"]["top_candidates"] if item["operation_type"] == "DIVIDE_BY_REGION"]
        self.assertTrue(divider_candidates)
        false_cover = divider_candidates[0]["false_cover"]
        self.assertGreater(false_cover["area"], 0.0)
        self.assertGreater(false_cover["cost"], 0.0)
        self.assertIn(3, false_cover["overlapped_face_ids"])

    def test_patch_trim_preserves_seed(self) -> None:
        seed = _face(99, 2, [[0, 0], [1, 0], [1, 20], [0, 20]], is_thin=True, degree=5)
        faces = [seed]
        adjacency = []
        for index in range(8):
            face = _face(index, 0, [[index + 2, 0], [index + 2.5, 0], [index + 2.5, 1], [index + 2, 1]], degree=1)
            faces.append(face)
            adjacency.append(_adj(99, index, 2, 0, length=10.0 - index))
        patches = build_operation_patches(_evidence(faces, adjacency), None, None, OperationExplainerConfig(max_patch_size=3))
        seed_patch = next(patch for patch in patches if patch.seed_face_id == 99)
        self.assertIn(99, seed_patch.face_ids)
        self.assertTrue(seed_patch.metadata["trimmed"])
        self.assertGreater(len(seed_patch.metadata["dropped_face_ids"]), 0)

    def test_candidate_dedupe_keeps_different_roles(self) -> None:
        faces = [
            _face(0, 0, [[0, 0], [9, 0], [9, 20], [0, 20]], degree=1),
            _face(1, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2),
            _face(2, 0, [[11, 0], [20, 0], [20, 20], [11, 20]], degree=1),
            _face(3, 2, [[0, 9], [20, 9], [20, 11], [0, 11]], is_thin=True, degree=2),
        ]
        evidence = _evidence(
            faces,
            [_adj(0, 1, 0, 2, 20.0), _adj(1, 2, 2, 0, 20.0), _adj(0, 3, 0, 2, 20.0), _adj(3, 2, 2, 0, 20.0)],
        )
        patches = build_operation_patches(evidence, None, None, OperationExplainerConfig())
        candidates, diagnostics = propose_operation_candidates_with_diagnostics(evidence, patches, None, None, OperationExplainerConfig())
        divider_candidate_keys = {
            (
                tuple(candidate.metadata.get("divider_face_ids", [])),
                tuple(candidate.metadata.get("support_face_ids", [])),
                candidate.metadata.get("latent_geometry", {}).get("policy"),
            )
            for candidate in candidates
            if candidate.operation_type == DIVIDE_BY_REGION
        }
        self.assertGreaterEqual(len(divider_candidate_keys), 2)
        self.assertGreaterEqual(diagnostics["raw_candidate_count"], diagnostics["deduplicated_candidate_count"])

    def test_benchmark_smoke_test(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "evidence.json"
            path.write_text(__import__("json").dumps(_evidence([face])), encoding="utf-8")
            rows = run_benchmark(root, config=OperationExplainerConfig())
        self.assertEqual(len(rows), 1)
        self.assertIn("residual_area_ratio", rows[0])
        self.assertIn("false_cover_ratio_max", rows[0])

    def test_render_iou_remains_null(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        payload = build_operation_explanation_payload(_evidence([face]), config=OperationExplainerConfig())
        metadata = payload["generator_target"]["metadata"]
        self.assertIsNone(metadata["render_iou"])
        self.assertEqual(metadata["render_validation"]["status"], "not_implemented")
        self.assertEqual(payload["validation"]["proxy_validation"]["status"], "proxy_only")

    def test_token_length_polygon_code(self) -> None:
        length = polygon_code_length_from_payload(
            {"outer_local": [[0, 0], [1, 0], [1, 1], [0, 1]], "holes_local": []},
            OperationExplainerConfig(),
        )
        self.assertIsInstance(length["total"], int)
        self.assertEqual(length["outer_vertex_count"], 4)
        self.assertEqual(length["hole_count"], 0)

    def test_token_length_convex_atoms(self) -> None:
        node = {
            "geometry_model": "convex_atoms",
            "atoms": [
                {"vertex_count": 3, "outer_local": [[0, 0], [1, 0], [0, 1]]},
                {"vertex_count": 4, "outer_local": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            ],
        }
        length = convex_atoms_code_length(node, OperationExplainerConfig())
        self.assertEqual(length["atom_count"], 2)
        self.assertEqual(length["atom_vertex_count"], 7)
        self.assertIsInstance(length["total"], int)

    def test_token_length_residual_gain_zero_or_non_positive(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        payload = build_operation_explanation_payload(_evidence([face]), config=OperationExplainerConfig(cost_profile="token_length_v1"))
        residual = next(item for item in payload["selected_operations"] if item["operation_type"] == "RESIDUAL")
        self.assertLessEqual(residual["cost"]["compression_gain"], 0)

    def test_false_cover_hard_invalid(self) -> None:
        left = _face(0, 0, [[0, 0], [2, 0], [2, 2], [0, 2]], degree=1)
        right = _face(1, 0, [[18, 18], [20, 18], [20, 20], [18, 20]], degree=1)
        divider = _face(2, 2, [[9, 0], [11, 0], [11, 20], [9, 20]], is_thin=True, degree=2)
        external = _face(3, 3, [[8, 8], [12, 8], [12, 12], [8, 12]], degree=0)
        evidence = _evidence([left, right, divider, external], [_adj(0, 2, 0, 2, 2.0), _adj(1, 2, 0, 2, 2.0)])
        payload = build_operation_explanation_payload(
            evidence,
            config=OperationExplainerConfig(
                cost_profile="token_length_v1",
                enable_visible_union=False,
                enable_union_with_divider=False,
                enable_convex_hull_fill=True,
                false_cover_ratio_invalid=0.01,
            ),
        )
        self.assertIn("false_cover", payload["diagnostics"]["failure_reasons"])

    def test_token_profile_selected_output(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        payload = build_operation_explanation_payload(_evidence([face]), config=OperationExplainerConfig(cost_profile="token_length_v1"))
        self.assertEqual(payload["diagnostics"]["cost_profile"], "token_length_v1")
        operation = payload["selected_operations"][0]["cost"]["breakdown"]
        self.assertEqual(operation["cost_profile"], "token_length_v1")

    def test_heuristic_profile_still_available(self) -> None:
        face = _face(0, 1, [[0, 0], [10, 0], [10, 10], [0, 10]])
        payload = build_operation_explanation_payload(_evidence([face]), config=OperationExplainerConfig(cost_profile="heuristic_v1"))
        self.assertEqual(payload["diagnostics"]["cost_profile"], "heuristic_v1")
        operation = payload["selected_operations"][0]["cost"]["breakdown"]
        self.assertEqual(operation["cost_profile"], "heuristic_v1")


if __name__ == "__main__":
    unittest.main()
