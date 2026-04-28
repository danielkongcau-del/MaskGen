from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_geometry_oracle_frame_conditioning import (
    build_oracle_frame_geometry_sequence_rows,
    encode_oracle_frame_conditioned_geometry_target,
    extract_geometry_tokens_from_oracle_frame_conditioned,
    geometry_shape_start_index,
    oracle_frame_geometry_prefix_from_tokens,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, build_token_vocabulary


def _topology_target() -> dict:
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": [256, 256],
        "parse_graph": {
            "nodes": [
                {
                    "id": "support_0",
                    "role": "support_region",
                    "label": 0,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "support_0",
                }
            ],
            "relations": [],
            "residuals": [],
        },
    }


def _geometry_target() -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": "support_0",
        "role": "support_region",
        "label": 0,
        "geometry_model": "polygon_code",
        "frame": {"origin": [128.0, 96.0], "scale": 24.0, "orientation": 0.25},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _write_split(tmpdir: str) -> Path:
    root = Path(tmpdir) / "split" / "train"
    topology_path = root / "topology" / "graphs" / "0.json"
    geometry_path = root / "geometry" / "0" / "support_0.json"
    topology_path.parent.mkdir(parents=True)
    geometry_path.parent.mkdir(parents=True)
    topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
    geometry_path.write_text(json.dumps(_geometry_target()), encoding="utf-8")
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "stem": "0",
                "topology_path": str(topology_path.as_posix()),
                "geometry_paths": [str(geometry_path.as_posix())],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


class ManualOracleFrameGeometryTest(unittest.TestCase):
    def test_oracle_frame_sequence_loss_starts_at_shape(self) -> None:
        config = ParseGraphTokenizerConfig()
        tokens = encode_oracle_frame_conditioned_geometry_target(
            _topology_target(),
            _geometry_target(),
            target_node_index=0,
            config=config,
        )

        shape_start = geometry_shape_start_index(tokens)
        prefix = oracle_frame_geometry_prefix_from_tokens(tokens)

        self.assertEqual(tokens[shape_start], "POLYS")
        self.assertEqual(prefix[-5], "FRAME")
        self.assertEqual(len(prefix), shape_start)

    def test_oracle_frame_extract_preserves_frame_in_geometry_tokens(self) -> None:
        config = ParseGraphTokenizerConfig()
        tokens = encode_oracle_frame_conditioned_geometry_target(
            _topology_target(),
            _geometry_target(),
            target_node_index=0,
            config=config,
        )

        geometry_tokens = extract_geometry_tokens_from_oracle_frame_conditioned(tokens)
        decoded = decode_geometry_tokens_to_target(geometry_tokens, config=config, source_node_id="support_0")

        self.assertAlmostEqual(decoded["frame"]["origin"][0], 128.0, delta=0.3)
        self.assertEqual(decoded["geometry"]["outer_local"][0], [-0.5, -0.5])

    def test_build_oracle_frame_rows_write_loss_start_at_polys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            config = ParseGraphTokenizerConfig()
            rows, summary = build_oracle_frame_geometry_sequence_rows(
                split_root,
                config=config,
                vocab=build_token_vocabulary(config),
            )

            self.assertEqual(summary["written_oracle_frame_geometry"], 1)
            self.assertEqual(rows[0]["tokens"][rows[0]["loss_start_index"]], "POLYS")


if __name__ == "__main__":
    unittest.main()
