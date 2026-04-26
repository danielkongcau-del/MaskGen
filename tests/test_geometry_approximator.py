from __future__ import annotations

import unittest

from shapely.geometry import MultiPolygon, Polygon

from partition_gen.geometry_approximator import _normalize_approx_geometry


class GeometryApproximatorTest(unittest.TestCase):
    def test_non_polygon_approx_geometry_falls_back_to_original_polygon(self) -> None:
        original = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        disjoint_approx = MultiPolygon(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            ]
        )

        geometry, reason = _normalize_approx_geometry(disjoint_approx, original)

        self.assertIsInstance(geometry, Polygon)
        self.assertEqual(reason, "non_polygon_approx_geometry:MultiPolygon")
        self.assertAlmostEqual(geometry.area, original.area)


if __name__ == "__main__":
    unittest.main()
