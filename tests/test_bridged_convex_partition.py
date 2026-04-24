from __future__ import annotations

import unittest

from shapely.geometry import Polygon

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    bridged_optimal_convex_partition,
    generate_bridge_candidates,
    validate_bridged_partition,
)
from partition_gen.convex_partition import _is_convex_polygon


class BridgedConvexPartitionTests(unittest.TestCase):
    def test_convex_polygon_no_holes(self) -> None:
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        payload = bridged_optimal_convex_partition(
            polygon,
            config=BridgedPartitionConfig(backend="fallback_cdt_greedy"),
        )
        self.assertEqual(payload["hole_count"] if "hole_count" in payload else 0, 0)
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


if __name__ == "__main__":
    unittest.main()
