from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_layout_ar import (
    attach_layout_frames_to_topology,
    build_layout_sequence_rows,
    decode_layout_tokens_to_target,
    encode_conditioned_layout_target,
    encode_layout_target,
    extract_layout_tokens_from_conditioned,
    layout_start_index,
    validate_layout_tokens,
)
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
                },
                {
                    "id": "insert_0",
                    "role": "insert_object",
                    "label": 1,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "insert_0",
                },
            ],
            "relations": [{"type": "adjacent_to", "faces": ["support_0", "insert_0"]}],
            "residuals": [],
        },
    }


def _geometry_target(source_node_id: str, role: str, label: int, origin: list[float]) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id,
        "role": role,
        "label": label,
        "geometry_model": "polygon_code",
        "frame": {"origin": origin, "scale": 16.0, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _geometry_targets() -> list[dict]:
    return [
        _geometry_target("support_0", "support_region", 0, [128.0, 128.0]),
        _geometry_target("insert_0", "insert_object", 1, [96.0, 96.0]),
    ]


def _write_split(tmpdir: str) -> Path:
    root = Path(tmpdir) / "split" / "train"
    topology_path = root / "topology" / "graphs" / "0.json"
    geometry_dir = root / "geometry" / "0"
    topology_path.parent.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
    geometry_paths = []
    for target in _geometry_targets():
        path = geometry_dir / f"{target['source_node_id']}.json"
        path.write_text(json.dumps(target), encoding="utf-8")
        geometry_paths.append(path)
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "stem": "0",
                "topology_path": str(topology_path.as_posix()),
                "geometry_paths": [str(path.as_posix()) for path in geometry_paths],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


class ManualLayoutARTest(unittest.TestCase):
    def test_layout_sequence_roundtrips_node_order(self) -> None:
        tokens = encode_layout_target(_topology_target(), _geometry_targets())

        decoded = decode_layout_tokens_to_target(tokens)

        self.assertEqual([row["node_index"] for row in decoded["nodes"]], [0, 1])
        self.assertEqual(tokens[1], "MANUAL_LAYOUT_V1")

    def test_conditioned_layout_loss_starts_at_layout(self) -> None:
        tokens = encode_conditioned_layout_target(_topology_target(), _geometry_targets())

        start = layout_start_index(tokens)
        layout_tokens = extract_layout_tokens_from_conditioned(tokens)

        self.assertEqual(tokens[start], "MANUAL_LAYOUT_V1")
        self.assertEqual(layout_tokens[0], "<BOS>")
        self.assertEqual(layout_tokens[1], "MANUAL_LAYOUT_V1")

    def test_layout_validator_rejects_missing_and_duplicate_nodes(self) -> None:
        valid = encode_layout_target(_topology_target(), _geometry_targets())
        missing = valid[: valid.index("NODE", valid.index("NODE") + 1)] + ["<EOS>"]
        duplicate = list(valid)
        duplicate[duplicate.index("I_1")] = "I_0"

        self.assertTrue(validate_layout_tokens(valid, topology_target=_topology_target())["valid"])
        self.assertFalse(validate_layout_tokens(missing, topology_target=_topology_target())["valid"])
        self.assertFalse(validate_layout_tokens(duplicate, topology_target=_topology_target())["valid"])

    def test_build_layout_rows_and_attach_preserves_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            config = ParseGraphTokenizerConfig()
            rows, summary = build_layout_sequence_rows(split_root, config=config, vocab=build_token_vocabulary(config))
            layout = decode_layout_tokens_to_target(extract_layout_tokens_from_conditioned(rows[0]["tokens"]))
            target, diagnostics = attach_layout_frames_to_topology(
                _topology_target(),
                layout,
                geometry_by_node_id={target["source_node_id"]: target for target in _geometry_targets()},
            )
            nodes_by_id = {node["id"]: node for node in target["parse_graph"]["nodes"]}

            self.assertEqual(summary["written_layout"], 1)
            self.assertEqual(rows[0]["tokens"][rows[0]["loss_start_index"]], "MANUAL_LAYOUT_V1")
            self.assertIn("geometry", nodes_by_id["support_0"])
            self.assertEqual(nodes_by_id["insert_0"]["geometry"]["outer_local"][0], [-0.5, -0.5])
            self.assertEqual(diagnostics["attached_geometry_count"], 2)


if __name__ == "__main__":
    unittest.main()
