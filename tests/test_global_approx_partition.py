from __future__ import annotations

import unittest

from partition_gen.global_approx_partition import (
    GlobalApproxConfig,
    build_global_approx_partition_payload,
)
from partition_gen.global_arc_regularizer import (
    GlobalArcRegularizationConfig,
    regularize_global_arc_payload,
)


def two_face_graph_with_split_shared_chain() -> dict:
    return {
        "source_mask": "synthetic/masks_id/two_faces.png",
        "size": [10, 10],
        "vertices": [
            [0, 0],
            [5, 0],
            [10, 0],
            [0, 10],
            [5, 10],
            [10, 10],
            [5, 5],
        ],
        "edges": [
            {"id": 0, "vertices": [0, 1], "length": 5, "faces": [0]},
            {"id": 1, "vertices": [1, 2], "length": 5, "faces": [1]},
            {"id": 2, "vertices": [3, 4], "length": 5, "faces": [0]},
            {"id": 3, "vertices": [4, 5], "length": 5, "faces": [1]},
            {"id": 4, "vertices": [0, 3], "length": 10, "faces": [0]},
            {"id": 5, "vertices": [2, 5], "length": 10, "faces": [1]},
            {"id": 6, "vertices": [1, 6], "length": 5, "faces": [0, 1]},
            {"id": 7, "vertices": [6, 4], "length": 5, "faces": [0, 1]},
        ],
        "faces": [
            {"id": 0, "label": 1, "area": 50, "bbox": [0, 0, 5, 10], "outer": [0, 1, 6, 4, 3], "holes": []},
            {"id": 1, "label": 2, "area": 50, "bbox": [5, 0, 10, 10], "outer": [1, 2, 5, 4, 6], "holes": []},
        ],
        "adjacency": [{"faces": [0, 1], "shared_length": 10}],
        "stats": {"num_edges": 8, "num_vertices": 7, "num_faces": 2},
    }


def two_face_graph_with_zigzag_shared_chain() -> dict:
    return {
        "source_mask": "synthetic/masks_id/two_faces_zigzag.png",
        "size": [10, 10],
        "vertices": [
            [0, 0],
            [5, 0],
            [4.5, 2],
            [5.5, 4],
            [4.5, 6],
            [5.5, 8],
            [5, 10],
            [0, 10],
            [10, 0],
            [10, 10],
        ],
        "edges": [
            {"id": 0, "vertices": [0, 1], "length": 5, "faces": [0]},
            {"id": 1, "vertices": [1, 2], "length": 2, "faces": [0, 1]},
            {"id": 2, "vertices": [2, 3], "length": 2, "faces": [0, 1]},
            {"id": 3, "vertices": [3, 4], "length": 2, "faces": [0, 1]},
            {"id": 4, "vertices": [4, 5], "length": 2, "faces": [0, 1]},
            {"id": 5, "vertices": [5, 6], "length": 2, "faces": [0, 1]},
            {"id": 6, "vertices": [6, 7], "length": 5, "faces": [0]},
            {"id": 7, "vertices": [7, 0], "length": 10, "faces": [0]},
            {"id": 8, "vertices": [1, 8], "length": 5, "faces": [1]},
            {"id": 9, "vertices": [8, 9], "length": 10, "faces": [1]},
            {"id": 10, "vertices": [9, 6], "length": 5, "faces": [1]},
        ],
        "faces": [
            {"id": 0, "label": 1, "area": 50, "bbox": [0, 0, 5, 10], "outer": [0, 1, 2, 3, 4, 5, 6, 7], "holes": []},
            {"id": 1, "label": 2, "area": 50, "bbox": [5, 0, 10, 10], "outer": [1, 8, 9, 6, 5, 4, 3, 2], "holes": []},
        ],
        "adjacency": [{"faces": [0, 1], "shared_length": 10}],
        "stats": {"num_edges": 11, "num_vertices": 10, "num_faces": 2},
    }


class GlobalApproxPartitionTests(unittest.TestCase):
    def test_shared_small_edges_become_one_maximal_arc(self) -> None:
        payload = build_global_approx_partition_payload(
            two_face_graph_with_split_shared_chain(),
            config=GlobalApproxConfig(simplify_tolerance=0.0),
            source_tag="synthetic.json",
        )
        self.assertTrue(payload["validation"]["is_valid"])
        self.assertEqual(payload["validation"]["face_count"], 2)
        self.assertEqual(payload["validation"]["shared_arc_count"], 1)
        self.assertEqual(payload["validation"]["missing_adjacency"], [])
        self.assertEqual(payload["validation"]["extra_adjacency"], [])

        shared_arcs = [
            arc
            for arc in payload["arcs"]
            if sorted(face for face in arc["incident_faces"] if face >= 0) == [0, 1]
        ]
        self.assertEqual(len(shared_arcs), 1)
        self.assertEqual(shared_arcs[0]["source_edge_ids"], [6, 7])

        face0_shared_refs = [
            ref for ref in payload["faces"][0]["outer_arc_refs"] if ref["arc_id"] == shared_arcs[0]["id"]
        ]
        face1_shared_refs = [
            ref for ref in payload["faces"][1]["outer_arc_refs"] if ref["arc_id"] == shared_arcs[0]["id"]
        ]
        self.assertEqual(len(face0_shared_refs), 1)
        self.assertEqual(len(face1_shared_refs), 1)
        self.assertNotEqual(face0_shared_refs[0]["reversed"], face1_shared_refs[0]["reversed"])

    def test_arc_regularizer_smooths_zigzag_shared_arc(self) -> None:
        graph = two_face_graph_with_zigzag_shared_chain()
        payload = build_global_approx_partition_payload(
            graph,
            config=GlobalApproxConfig(face_simplify_tolerance=0.0, simplify_tolerance=0.0),
            source_tag="synthetic_zigzag.json",
        )
        regularized = regularize_global_arc_payload(
            payload,
            graph_data=graph,
            config=GlobalArcRegularizationConfig(simplify_tolerance=1.25, max_distance=1.25),
        )

        self.assertTrue(regularized["validation"]["is_valid"])
        self.assertGreaterEqual(regularized["arc_regularization"]["accepted_count"], 1)
        shared_arcs = [
            arc
            for arc in regularized["arcs"]
            if sorted(face for face in arc["incident_faces"] if face >= 0) == [0, 1]
        ]
        self.assertEqual(len(shared_arcs), 1)
        self.assertTrue(shared_arcs[0].get("regularized"))
        self.assertEqual(shared_arcs[0]["vertex_count"], 2)


if __name__ == "__main__":
    unittest.main()
