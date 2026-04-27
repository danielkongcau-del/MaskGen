from __future__ import annotations

import unittest

from partition_gen.manual_geometry_sample_validation import (
    decode_geometry_tokens_to_target,
    geometry_prefix_from_tokens,
    geometry_prefix_tokens,
    validate_geometry_tokens,
)
from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


def _geometry_target() -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": "support_0",
        "role": "support_region",
        "label": 0,
        "geometry_model": "polygon_code",
        "frame": {"origin": [128.0, 128.0], "scale": 16.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


class ManualGeometrySampleValidationTest(unittest.TestCase):
    def test_decode_geometry_tokens_to_target_roundtrips_tokens(self) -> None:
        config = ParseGraphTokenizerConfig(coord_bins=32, position_bins=32, scale_bins=32, angle_bins=32)
        tokens = encode_geometry_target(_geometry_target(), config=config)

        decoded = decode_geometry_tokens_to_target(tokens, config=config, source_node_id="generated_0")

        self.assertEqual(decoded["target_type"], "manual_parse_graph_geometry_v1")
        self.assertEqual(decoded["source_node_id"], "generated_0")
        self.assertEqual(encode_geometry_target(decoded, config=config), tokens)

    def test_validate_geometry_tokens_detects_truncation(self) -> None:
        tokens = encode_geometry_target(_geometry_target())[:-1]

        result = validate_geometry_tokens(tokens)

        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])

    def test_geometry_prefix_helpers(self) -> None:
        prefix = geometry_prefix_tokens(role="support_region", label=0, geometry_model="polygon_code")

        self.assertEqual(prefix, ["<BOS>", "MANUAL_GEOMETRY_V1", "GEOMETRY_BLOCK", "ROLE_SUPPORT", "LABEL", "I_0", "GEOM_POLYGON_CODE"])
        self.assertEqual(geometry_prefix_from_tokens(prefix + ["FRAME"]), prefix)


if __name__ == "__main__":
    unittest.main()
