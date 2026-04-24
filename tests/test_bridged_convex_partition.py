from __future__ import annotations

from pathlib import Path
import unittest

from shapely.geometry import Polygon

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    _post_snap_cleanup_and_convex_merge,
    bridged_optimal_convex_partition,
    generate_bridge_candidates,
    validate_bridged_partition,
)
from partition_gen.convex_partition import _is_convex_polygon


class BridgedConvexPartitionTests(unittest.TestCase):
    CGAL_CLI = Path("build/cgal_tools/Release/optimal_convex_partition_cli.exe")

    def test_convex_polygon_no_holes(self) -> None:
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        payload = bridged_optimal_convex_partition(
            polygon,
            config=BridgedPartitionConfig(backend="fallback_cdt_greedy"),
        )
        self.assertEqual(payload["hole_count"] if "hole_count" in payload else 0, 0)
        self.assertFalse(payload["backend_info"]["simple_polygon_optimal"])
        self.assertFalse(payload["backend_info"]["global_optimal"])
        self.assertEqual(payload["backend_info"]["optimal_scope"], "fallback_cdt_greedy")
        self.assertEqual(payload["selected_bridge_set"]["bridge_ids"], [])
        self.assertEqual(payload["validation"]["piece_count"], 1)
        self.assertTrue(payload["validation"]["all_convex"])
        self.assertAlmostEqual(payload["validation"]["iou"], 1.0, places=6)

    def test_concave_polygon_no_holes(self) -> None:
        polygon = Polygon([(0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)])
        payload = bridged_optimal_convex_partition(
            polygon,
            config=BridgedPartitionConfig(backend="fallback_cdt_greedy"),
        )
        self.assertGreaterEqual(payload["validation"]["piece_count"], 2)
        self.assertTrue(payload["validation"]["all_convex"])
        self.assertAlmostEqual(payload["validation"]["iou"], 1.0, places=6)

    def test_concave_polygon_uses_cgal_when_cli_is_available(self) -> None:
        if not self.CGAL_CLI.exists():
            self.skipTest("CGAL CLI is not built")
        polygon = Polygon([(0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)])
        payload = bridged_optimal_convex_partition(
            polygon,
            config=BridgedPartitionConfig(
                backend="cgal",
                cgal_cli=str(self.CGAL_CLI),
            ),
        )
        self.assertEqual(payload["backend_info"]["backend"], "cgal")
        self.assertTrue(payload["backend_info"]["optimal"])
        self.assertTrue(payload["backend_info"]["simple_polygon_optimal"])
        self.assertTrue(payload["backend_info"]["global_optimal"])
        self.assertEqual(payload["backend_info"]["optimal_scope"], "simple_polygon")
        self.assertEqual(payload["validation"]["piece_count"], 2)
        self.assertTrue(payload["validation"]["all_convex"])
        self.assertAlmostEqual(payload["validation"]["iou"], 1.0, places=6)

    def test_one_rectangular_hole_has_bridge_and_valid_partition(self) -> None:
        polygon = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [[(3, 3), (7, 3), (7, 7), (3, 7)]],
        )
        config = BridgedPartitionConfig(backend="fallback_cdt_greedy")
        candidates = generate_bridge_candidates(polygon, config=config)
        payload = bridged_optimal_convex_partition(polygon, config=config)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(len(payload["selected_bridge_set"]["bridge_ids"]), 1)
        self.assertTrue(payload["validation"]["all_convex"])
        self.assertAlmostEqual(payload["validation"]["iou"], 1.0, places=6)

    def test_one_rectangular_hole_uses_cgal_bridge_cut_when_cli_is_available(self) -> None:
        if not self.CGAL_CLI.exists():
            self.skipTest("CGAL CLI is not built")
        polygon = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [[(3, 3), (7, 3), (7, 7), (3, 7)]],
        )
        payload = bridged_optimal_convex_partition(
            polygon,
            config=BridgedPartitionConfig(
                backend="cgal",
                cgal_cli=str(self.CGAL_CLI),
            ),
        )
        self.assertEqual(payload["backend_info"]["backend"], "cgal_bridge_cut")
        self.assertTrue(payload["backend_info"]["optimal"])
        self.assertTrue(payload["backend_info"]["simple_polygon_optimal"])
        self.assertFalse(payload["backend_info"]["global_optimal"])
        self.assertEqual(payload["backend_info"]["optimal_scope"], "selected_bridge_cut_simple_polygon")
        self.assertEqual(len(payload["selected_bridge_set"]["bridge_ids"]), 1)
        self.assertEqual(payload["validation"]["piece_count"], 4)
        self.assertTrue(payload["validation"]["all_convex"])
        self.assertAlmostEqual(payload["validation"]["iou"], 1.0, places=6)

    def test_validation_rejects_non_convex_piece(self) -> None:
        original = Polygon([(0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)])
        validation = validate_bridged_partition(
            original,
            [original],
            config=BridgedPartitionConfig(),
        )
        self.assertFalse(
            _is_convex_polygon(original, rel_eps=1e-7, abs_eps=1e-8)
        )
        self.assertFalse(validation["all_convex"])

    def test_post_snap_cleanup_merges_near_duplicate_slit_pieces(self) -> None:
        pieces = [
            Polygon([(198.00024835648006, 71.99795105903969), (198.0, 72.0), (193.0, 0.0), (199.0, 2.0)]),
            Polygon([(198.00074506944011, 71.9980752372797), (198.00024835648006, 71.99795105903969), (199.0, 2.0), (206.0, 70.0)]),
        ]
        merged, info = _post_snap_cleanup_and_convex_merge(
            pieces,
            cleanup_eps=0.004096,
            config=BridgedPartitionConfig(),
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(info["post_snap_merge_count"], 1)
        self.assertTrue(_is_convex_polygon(merged[0], rel_eps=1e-7, abs_eps=1e-8))
        self.assertAlmostEqual(merged[0].area, 490.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
