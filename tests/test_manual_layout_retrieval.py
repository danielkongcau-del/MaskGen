from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from partition_gen.manual_geometry_oracle_frame_conditioning import (
    encode_oracle_frame_conditioned_geometry_target,
    extract_geometry_tokens_from_oracle_frame_conditioned,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target
from partition_gen.manual_layout_retrieval import (
    attach_retrieved_layout_to_split_targets,
    build_layout_retrieval_fallbacks,
    build_layout_retrieval_library,
    geometry_condition_target_from_topology_node,
    map_retrieved_layout_frames,
    retrieve_layout_entry,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


def _topology_target(*, stem: str, insert_label: int = 1, include_divider: bool = False) -> dict:
    nodes = [
        {
            "id": f"{stem}_support",
            "role": "support_region",
            "label": 0,
            "renderable": True,
            "is_reference_only": False,
            "geometry_model": "polygon_code",
            "geometry_ref": f"{stem}_support",
        },
        {
            "id": f"{stem}_insert",
            "role": "insert_object",
            "label": insert_label,
            "renderable": True,
            "is_reference_only": False,
            "geometry_model": "polygon_code",
            "geometry_ref": f"{stem}_insert",
        },
    ]
    relations = [{"type": "adjacent_to", "faces": [nodes[0]["id"], nodes[1]["id"]]}]
    if include_divider:
        nodes.append(
            {
                "id": f"{stem}_divider",
                "role": "divider_region",
                "label": 2,
                "renderable": True,
                "is_reference_only": False,
                "geometry_model": "polygon_code",
                "geometry_ref": f"{stem}_divider",
            }
        )
        relations.append({"type": "divides", "divider": nodes[-1]["id"], "target": nodes[0]["id"]})
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": [256, 256],
        "parse_graph": {"nodes": nodes, "relations": relations, "residuals": []},
    }


def _geometry_target(source_node_id: str, role: str, label: int, origin: list[float], scale: float = 16.0) -> dict:
    points = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id,
        "role": role,
        "label": label,
        "geometry_model": "polygon_code",
        "frame": {"origin": origin, "scale": scale, "orientation": 0.0},
        "geometry": {
            "outer_local": points,
            "holes_local": [],
            "polygons_local": [{"outer_local": points, "holes_local": []}],
        },
    }


def _write_split_row(root: Path, *, stem: str, topology: dict, origins: list[list[float]]) -> None:
    topology_path = root / "topology" / "graphs" / f"{stem}.json"
    geometry_dir = root / "geometry" / stem
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(json.dumps(topology), encoding="utf-8")
    geometry_paths = []
    for node, origin in zip(topology["parse_graph"]["nodes"], origins):
        if not node.get("geometry_ref"):
            continue
        path = geometry_dir / f"{node['geometry_ref']}.json"
        target = _geometry_target(str(node["geometry_ref"]), str(node["role"]), int(node["label"]), origin)
        path.write_text(json.dumps(target), encoding="utf-8")
        geometry_paths.append(path)
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stem": stem,
                    "topology_path": str(topology_path.as_posix()),
                    "geometry_paths": [str(path.as_posix()) for path in geometry_paths],
                }
            )
            + "\n"
        )


class ManualLayoutRetrievalTest(unittest.TestCase):
    def test_retrieves_nearest_signature_and_maps_by_role_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            train.mkdir(parents=True)
            close_topology = _topology_target(stem="close", insert_label=1)
            far_topology = _topology_target(stem="far", insert_label=9, include_divider=True)
            _write_split_row(train, stem="close", topology=close_topology, origins=[[128.0, 128.0], [96.0, 96.0]])
            _write_split_row(train, stem="far", topology=far_topology, origins=[[16.0, 16.0], [32.0, 32.0], [48.0, 48.0]])
            library, _summary = build_layout_retrieval_library(train)
            fallback = build_layout_retrieval_fallbacks(library)
            query = _topology_target(stem="query", insert_label=1)

            retrieved, score = retrieve_layout_entry(query, library)
            frame_by_index, diagnostics = map_retrieved_layout_frames(query, retrieved, fallback_frames=fallback)

            self.assertEqual(retrieved["stem"], "close")
            self.assertEqual(score, 0.0)
            self.assertEqual(frame_by_index[0]["origin"], [128.0, 128.0])
            self.assertEqual(frame_by_index[1]["origin"], [96.0, 96.0])
            self.assertEqual(diagnostics["mapping_mode_histogram"]["retrieved_exact_order"], 2)

    def test_attach_retrieved_layout_to_split_targets_preserves_true_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            val = Path(tmpdir) / "split" / "val"
            train.mkdir(parents=True)
            val.mkdir(parents=True)
            _write_split_row(
                train,
                stem="train_a",
                topology=_topology_target(stem="train_a", insert_label=1),
                origins=[[128.0, 128.0], [96.0, 96.0]],
            )
            _write_split_row(
                val,
                stem="val_a",
                topology=_topology_target(stem="val_a", insert_label=1),
                origins=[[10.0, 10.0], [20.0, 20.0]],
            )
            library, _summary = build_layout_retrieval_library(train)
            targets = attach_retrieved_layout_to_split_targets(
                val,
                library_entries=library,
                fallback_frames=build_layout_retrieval_fallbacks(library),
            )
            nodes_by_id = {node["id"]: node for node in targets[0]["parse_graph"]["nodes"]}

            self.assertEqual(targets[0]["metadata"]["attached_geometry_count"], 2)
            self.assertEqual(nodes_by_id["val_a_support"]["frame"]["origin"], [128.0, 128.0])
            self.assertEqual(nodes_by_id["val_a_insert"]["geometry"]["outer_local"][0], [-0.5, -0.5])
            self.assertEqual(targets[0]["parse_graph"]["relations"], _topology_target(stem="val_a")["parse_graph"]["relations"])

    def test_retrieved_frame_can_condition_local_geometry_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            train.mkdir(parents=True)
            _write_split_row(
                train,
                stem="train_a",
                topology=_topology_target(stem="train_a", insert_label=1),
                origins=[[128.0, 128.0], [96.0, 96.0]],
            )
            library, _summary = build_layout_retrieval_library(train)
            fallback = build_layout_retrieval_fallbacks(library)
            query = _topology_target(stem="query", insert_label=1)
            retrieved, _score = retrieve_layout_entry(query, library)
            frame_by_index, _diagnostics = map_retrieved_layout_frames(query, retrieved, fallback_frames=fallback)
            geometry = _geometry_target("query_support", "support_region", 0, [10.0, 10.0])
            geometry["frame"] = frame_by_index[0]
            config = ParseGraphTokenizerConfig()

            conditioned_tokens = encode_oracle_frame_conditioned_geometry_target(
                query,
                geometry,
                target_node_index=0,
                config=config,
            )
            decoded = decode_geometry_tokens_to_target(
                extract_geometry_tokens_from_oracle_frame_conditioned(conditioned_tokens),
                config=config,
                source_node_id="query_support",
            )

            self.assertAlmostEqual(decoded["frame"]["origin"][0], 128.0, delta=0.3)
            self.assertAlmostEqual(decoded["frame"]["origin"][1], 128.0, delta=0.3)

    def test_generated_topology_node_can_condition_retrieved_frame_geometry(self) -> None:
        query = _topology_target(stem="query", insert_label=1)
        node = query["parse_graph"]["nodes"][0]
        frame = {"origin": [64.0, 192.0], "scale": 32.0, "orientation": 0.0}
        geometry = geometry_condition_target_from_topology_node(node, frame=frame, source_node_id=str(node["id"]))
        config = ParseGraphTokenizerConfig()

        conditioned_tokens = encode_oracle_frame_conditioned_geometry_target(
            query,
            geometry,
            target_node_index=0,
            config=config,
        )
        decoded = decode_geometry_tokens_to_target(
            extract_geometry_tokens_from_oracle_frame_conditioned(conditioned_tokens),
            config=config,
            source_node_id=str(node["id"]),
        )

        self.assertEqual(geometry["source_node_id"], "query_support")
        self.assertAlmostEqual(decoded["frame"]["origin"][0], 64.0, delta=0.3)
        self.assertAlmostEqual(decoded["frame"]["origin"][1], 192.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
