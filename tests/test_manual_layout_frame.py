from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from partition_gen.manual_layout_frame import (
    ManualLayoutFrameDataset,
    ManualLayoutFrameMLP,
    ManualLayoutFrameMLPConfig,
    ManualLayoutFrameRegressor,
    ManualLayoutFrameRegressorConfig,
    attach_predicted_frames_to_topology_sample_rows,
    attach_predicted_frames_to_split_rows,
    bins_to_frame,
    build_layout_frame_example,
    collate_layout_frame_examples,
    evaluate_layout_frame_model,
    evaluate_role_label_frame_baseline,
    frame_to_bins,
    layout_frame_loss,
    layout_frame_regression_loss,
)
from partition_gen.manual_topology_placeholder_geometry import GeometryPlaceholderLibrary
from partition_gen.parse_graph_compact_tokenizer import encode_topology_target
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


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
                    "id": "insert_group_0",
                    "role": "insert_object_group",
                    "label": 1,
                    "renderable": False,
                    "is_reference_only": False,
                    "geometry_model": "none",
                    "children": ["insert_0"],
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
                {
                    "id": "divider_0",
                    "role": "divider_region",
                    "label": 2,
                    "renderable": True,
                    "is_reference_only": False,
                    "geometry_model": "polygon_code",
                    "geometry_ref": "divider_0",
                },
            ],
            "relations": [
                {"type": "contains", "parent": "insert_group_0", "child": "insert_0"},
                {"type": "inserted_in", "object": "insert_group_0", "container": "support_0"},
                {"type": "divides", "divider": "divider_0", "target": "insert_group_0"},
                {"type": "adjacent_to", "faces": ["support_0", "insert_0"]},
            ],
            "residuals": [],
        },
    }


def _geometry_target(source_node_id: str, role: str, label: int, *, origin: list[float]) -> dict:
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


def _write_split(tmpdir: str) -> Path:
    root = Path(tmpdir) / "split" / "val"
    topology_path = root / "topology" / "graphs" / "0.json"
    geometry_dir = root / "geometry" / "0"
    topology_path.parent.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    topology_path.write_text(json.dumps(_topology_target()), encoding="utf-8")
    geometry_paths = [
        geometry_dir / "support_0.json",
        geometry_dir / "insert_0.json",
        geometry_dir / "divider_0.json",
    ]
    targets = [
        _geometry_target("support_0", "support_region", 0, origin=[128.0, 128.0]),
        _geometry_target("insert_0", "insert_object", 1, origin=[96.0, 96.0]),
        _geometry_target("divider_0", "divider_region", 2, origin=[64.0, 64.0]),
    ]
    for path, target in zip(geometry_paths, targets):
        path.write_text(json.dumps(target), encoding="utf-8")
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


class ManualLayoutFrameTest(unittest.TestCase):
    def test_frame_bins_roundtrip(self) -> None:
        config = ParseGraphTokenizerConfig()
        frame = {"origin": [128.0, 64.0], "scale": 32.0, "orientation": 0.25}

        decoded = bins_to_frame(frame_to_bins(frame, config=config), config=config)

        self.assertAlmostEqual(decoded["origin"][0], 128.0, delta=0.3)
        self.assertAlmostEqual(decoded["origin"][1], 64.0, delta=0.3)
        self.assertAlmostEqual(decoded["scale"], 32.0, delta=0.6)

    def test_relation_features_cover_topology_context(self) -> None:
        example = build_layout_frame_example(_topology_target(), node_index=2, frame={"origin": [96.0, 96.0], "scale": 16.0})

        self.assertEqual(example["role"], "insert_object")
        self.assertEqual(example["target_bins"]["origin_x"], frame_to_bins({"origin": [96.0, 0.0]}, config=ParseGraphTokenizerConfig())["origin_x"])
        self.assertGreater(example["numeric"][12], 0.0)  # has parent group
        self.assertGreater(example["numeric"][24], 0.0)  # adjacent degree

    def test_dataset_and_mlp_forward_backward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)
            batch = collate_layout_frame_examples([dataset[0], dataset[1]])
            model = ManualLayoutFrameMLP(
                ManualLayoutFrameMLPConfig(
                    numeric_dim=dataset.numeric_dim,
                    hidden_dim=32,
                    num_layers=1,
                    position_bins=1024,
                    scale_bins=1024,
                    angle_bins=1024,
                )
            )

            logits = model(batch)
            loss = layout_frame_loss(logits, batch)
            loss.backward()

            self.assertEqual(logits["origin_x"].shape, (2, 1024))
            self.assertTrue(float(loss.item()) > 0.0)

    def test_regression_mlp_forward_backward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)
            batch = collate_layout_frame_examples([dataset[0], dataset[1]])
            model = ManualLayoutFrameRegressor(
                ManualLayoutFrameRegressorConfig(
                    numeric_dim=dataset.numeric_dim,
                    hidden_dim=32,
                    num_layers=1,
                )
            )

            predictions = model(batch)
            loss = layout_frame_regression_loss(predictions, batch)
            loss.backward()

            self.assertEqual(predictions.shape, (2, 5))
            self.assertTrue(float(loss.item()) >= 0.0)

    def test_evaluate_reports_histograms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)
            model = ManualLayoutFrameMLP(
                ManualLayoutFrameMLPConfig(
                    numeric_dim=dataset.numeric_dim,
                    hidden_dim=32,
                    num_layers=1,
                    position_bins=1024,
                    scale_bins=1024,
                    angle_bins=1024,
                )
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_layout_frame_examples,
            )

            metrics = evaluate_layout_frame_model(
                model,
                loader,
                device=torch.device("cpu"),
                config=ParseGraphTokenizerConfig(),
            )

            self.assertIn("head_histograms", metrics)
            self.assertEqual(metrics["head_histograms"]["origin_x"]["target_unique_count"], 3)
            self.assertGreaterEqual(metrics["head_histograms"]["origin_x"]["prediction_unique_count"], 1)

    def test_role_label_mean_baseline_memorizes_same_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)

            metrics = evaluate_role_label_frame_baseline(
                dataset.rows,
                dataset.rows,
                config=ParseGraphTokenizerConfig(),
            )

            self.assertEqual(metrics["example_count"], len(dataset))
            self.assertEqual(metrics["fallback_counts"], {"role_label": len(dataset)})
            self.assertAlmostEqual(metrics["origin_mae"], 0.0, delta=1e-6)
            self.assertEqual(metrics["head_accuracy"]["origin_x"], 1.0)

    def test_attach_predicted_frame_preserves_local_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)
            model = ManualLayoutFrameMLP(
                ManualLayoutFrameMLPConfig(
                    numeric_dim=dataset.numeric_dim,
                    hidden_dim=32,
                    num_layers=1,
                    position_bins=1024,
                    scale_bins=1024,
                    angle_bins=1024,
                )
            )
            targets = attach_predicted_frames_to_split_rows(
                split_root,
                model=model,
                tokenizer_config=ParseGraphTokenizerConfig(),
                device=torch.device("cpu"),
                max_samples=1,
            )
            nodes_by_id = {node["id"]: node for node in targets[0]["parse_graph"]["nodes"]}

            self.assertIn("frame", nodes_by_id["support_0"])
            self.assertIn("geometry", nodes_by_id["support_0"])
            self.assertNotIn("geometry_ref", nodes_by_id["support_0"])
            self.assertEqual(nodes_by_id["support_0"]["geometry"]["outer_local"][0], [-0.5, -0.5])

    def test_attach_predicted_frame_to_topology_samples_uses_placeholder_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_root = _write_split(tmpdir)
            dataset = ManualLayoutFrameDataset(split_root)
            model = ManualLayoutFrameMLP(
                ManualLayoutFrameMLPConfig(
                    numeric_dim=dataset.numeric_dim,
                    hidden_dim=32,
                    num_layers=1,
                    position_bins=1024,
                    scale_bins=1024,
                    angle_bins=1024,
                )
            )
            shape_library = GeometryPlaceholderLibrary(
                [
                    _geometry_target("support_src", "support_region", 0, origin=[128.0, 128.0]),
                    _geometry_target("insert_src", "insert_object", 1, origin=[96.0, 96.0]),
                    _geometry_target("divider_src", "divider_region", 2, origin=[64.0, 64.0]),
                ],
                seed=1,
            )
            targets = attach_predicted_frames_to_topology_sample_rows(
                [{"sample_index": 0, "tokens": encode_topology_target(_topology_target())}],
                model=model,
                tokenizer_config=ParseGraphTokenizerConfig(),
                device=torch.device("cpu"),
                shape_library=shape_library,
            )
            nodes_by_id = {node["id"]: node for node in targets[0]["parse_graph"]["nodes"]}

            self.assertEqual(len(targets), 1)
            self.assertIn("frame", nodes_by_id["support_0"])
            self.assertIn("geometry", nodes_by_id["insert_0"])
            self.assertNotIn("geometry_ref", nodes_by_id["divider_0"])


if __name__ == "__main__":
    unittest.main()
