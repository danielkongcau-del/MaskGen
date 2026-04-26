from __future__ import annotations

import unittest

from partition_gen.explainer import ExplainerConfig, build_explanation_payload
from partition_gen.explanation_evidence import ExplanationEvidenceConfig, build_explanation_evidence_payload
from partition_gen.global_approx_partition import GlobalApproxConfig, build_global_approx_partition_payload
from partition_gen.pairwise_relation_explainer import PairwiseRelationConfig, build_pairwise_relation_payload
from partition_gen.weak_parse_graph_renderer import WeakRenderConfig, render_weak_explanation_payload
from partition_gen.weak_explainer import WeakExplainerConfig, build_weak_explanation_payload
from tests.test_global_approx_partition import two_face_graph_with_split_shared_chain


class ExplanationEvidenceAndExplainerTests(unittest.TestCase):
    def test_evidence_builder_packs_faces_arcs_adjacency_and_convex_atoms(self) -> None:
        global_payload = build_global_approx_partition_payload(
            two_face_graph_with_split_shared_chain(),
            config=GlobalApproxConfig(face_simplify_tolerance=0.0, simplify_tolerance=0.0),
            source_tag="synthetic.json",
        )
        evidence = build_explanation_evidence_payload(
            global_payload,
            config=ExplanationEvidenceConfig(convex_backend="fallback_cdt_greedy"),
            source_tag="global.json",
        )

        self.assertEqual(evidence["format"], "maskgen_explanation_evidence_v1")
        self.assertEqual(evidence["evidence_validation"]["face_count"], 2)
        self.assertEqual(evidence["evidence_validation"]["adjacency_count"], 1)
        self.assertEqual(evidence["evidence_validation"]["convex_failure_count"], 0)
        self.assertIn("features", evidence["faces"][0])
        self.assertIn("convex_partition", evidence["faces"][0])
        self.assertGreaterEqual(evidence["faces"][0]["convex_partition"]["piece_count"], 1)
        self.assertTrue(evidence["arcs"][0]["features"]["length"] >= 0.0)

    def test_explainer_outputs_nested_generator_target_parse_graph(self) -> None:
        global_payload = build_global_approx_partition_payload(
            two_face_graph_with_split_shared_chain(),
            config=GlobalApproxConfig(face_simplify_tolerance=0.0, simplify_tolerance=0.0),
            source_tag="synthetic.json",
        )
        evidence = build_explanation_evidence_payload(
            global_payload,
            config=ExplanationEvidenceConfig(convex_backend="fallback_cdt_greedy"),
            source_tag="global.json",
        )
        explanation = build_explanation_payload(
            evidence,
            config=ExplainerConfig(),
            source_tag="evidence.json",
        )

        self.assertEqual(explanation["format"], "maskgen_explanation_v1")
        target = explanation["generator_target"]
        self.assertEqual(target["format"], "maskgen_generator_target_v1")
        self.assertEqual(target["target_type"], "parse_graph")
        self.assertIn("parse_graph", target)
        self.assertIn("nodes", target["parse_graph"])
        self.assertIn("relations", target["parse_graph"])
        self.assertIn("residuals", target["parse_graph"])
        self.assertIn("metadata", target)
        self.assertTrue(explanation["validation"]["is_valid"])
        self.assertIn("label_role_summary", explanation["diagnostics"])
        self.assertTrue(explanation["diagnostics"]["label_role_summary"]["enabled"])
        for node in target["parse_graph"]["nodes"]:
            self.assertIn("role", node)
            self.assertIn("label", node)

    def test_pairwise_relation_payload_records_binary_scene_candidates(self) -> None:
        evidence = {
            "size": [20, 20],
            "faces": [
                {
                    "id": 0,
                    "label": 0,
                    "geometry": {
                        "outer": [[0, 0], [20, 0], [20, 20], [0, 20]],
                        "holes": [],
                    },
                },
                {
                    "id": 1,
                    "label": 1,
                    "geometry": {
                        "outer": [[4, 4], [8, 4], [8, 8], [4, 8]],
                        "holes": [],
                    },
                },
                {
                    "id": 2,
                    "label": 2,
                    "geometry": {
                        "outer": [[9, 0], [11, 0], [11, 20], [9, 20]],
                        "holes": [],
                    },
                },
            ],
            "adjacency": [
                {"faces": [0, 1], "labels": [0, 1], "shared_length": 16.0},
                {"faces": [0, 2], "labels": [0, 2], "shared_length": 40.0},
            ],
        }

        payload = build_pairwise_relation_payload(
            evidence,
            config=PairwiseRelationConfig(convex_backend="fallback_cdt_greedy"),
        )

        self.assertEqual(payload["format"], "maskgen_pairwise_relation_analysis_v1")
        self.assertEqual(payload["statistics"]["pair_count"], 2)
        self.assertIn("preferred_role_by_label", payload)
        for pair in payload["pairs"]:
            self.assertIn("selected", pair)
            self.assertGreaterEqual(len(pair["candidates"]), 1)
            self.assertIn("roles", pair["selected"])

    def test_weak_explainer_outputs_face_atom_parse_graph(self) -> None:
        global_payload = build_global_approx_partition_payload(
            two_face_graph_with_split_shared_chain(),
            config=GlobalApproxConfig(face_simplify_tolerance=0.0, simplify_tolerance=0.0),
            source_tag="synthetic.json",
        )
        evidence = build_explanation_evidence_payload(
            global_payload,
            config=ExplanationEvidenceConfig(convex_backend="fallback_cdt_greedy"),
            source_tag="global.json",
        )
        weak = build_weak_explanation_payload(
            evidence,
            config=WeakExplainerConfig(),
            source_tag="evidence.json",
        )

        self.assertEqual(weak["format"], "maskgen_explanation_v1")
        self.assertEqual(weak["explainer_profile"], "weak_convex_face_atoms_v1")
        target = weak["generator_target"]
        self.assertEqual(target["format"], "maskgen_generator_target_v1")
        self.assertEqual(target["metadata"]["target_profile"], "weak_convex_face_atoms_v1")
        graph = target["parse_graph"]
        roles = {node["role"] for node in graph["nodes"]}
        self.assertIn("semantic_face", roles)
        self.assertIn("convex_atom", roles)
        relation_types = {relation["type"] for relation in graph["relations"]}
        self.assertIn("atom_part_of", relation_types)
        self.assertTrue(weak["validation"]["is_valid"])

    def test_weak_renderer_reconstructs_atom_faces(self) -> None:
        global_payload = build_global_approx_partition_payload(
            two_face_graph_with_split_shared_chain(),
            config=GlobalApproxConfig(face_simplify_tolerance=0.0, simplify_tolerance=0.0),
            source_tag="synthetic.json",
        )
        evidence = build_explanation_evidence_payload(
            global_payload,
            config=ExplanationEvidenceConfig(convex_backend="fallback_cdt_greedy"),
            source_tag="global.json",
        )
        weak = build_weak_explanation_payload(
            evidence,
            config=WeakExplainerConfig(),
            source_tag="evidence.json",
        )
        rendered = render_weak_explanation_payload(
            weak,
            evidence_payload=evidence,
            config=WeakRenderConfig(),
        )

        validation = rendered["validation"]
        self.assertEqual(rendered["format"], "maskgen_weak_rendered_partition_v1")
        self.assertTrue(validation["is_valid"])
        self.assertAlmostEqual(validation["full_iou"], 1.0, places=6)
        self.assertEqual(validation["low_iou_face_ids"], [])


if __name__ == "__main__":
    unittest.main()
