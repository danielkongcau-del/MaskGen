from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from partition_gen.manual_geometry_shape_fallback import (
    build_geometry_shape_fallback_library,
    geometry_target_from_fallback_shape,
    local_bbox_quality,
    select_fallback_geometry_shape,
)
from partition_gen.manual_layout_residual import (
    ManualLayoutResidualDataset,
    ManualLayoutResidualRegressor,
    ManualLayoutResidualRegressorConfig,
    attach_retrieved_residual_layout_to_split_targets,
    build_layout_residual_example,
    clamp_frame_to_local_bbox,
    collate_layout_residual_examples,
    evaluate_layout_residual_regressor,
    frame_residual_target,
    geometry_aware_scale_max,
    layout_residual_loss,
    residual_values_to_raw_scale,
    residual_values_to_frame,
)
from partition_gen.manual_layout_retrieval import (
    build_layout_retrieval_fallbacks,
    build_layout_retrieval_library,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig


def _topology_target(*, stem: str, insert_label: int = 1) -> dict:
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
    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": [256, 256],
        "parse_graph": {
            "nodes": nodes,
            "relations": [{"type": "adjacent_to", "faces": [nodes[0]["id"], nodes[1]["id"]]}],
            "residuals": [],
        },
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


class ZeroResidualModel(torch.nn.Module):
    def forward(self, batch: dict) -> torch.Tensor:
        return torch.zeros((int(batch["numeric"].shape[0]), 4), dtype=torch.float32, device=batch["numeric"].device)


class ManualLayoutResidualTest(unittest.TestCase):
    def test_frame_residual_roundtrip(self) -> None:
        retrieved = {"origin": [64.0, 96.0], "scale": 16.0, "orientation": 0.25}
        target = {"origin": [96.0, 32.0], "scale": 32.0, "orientation": -0.5}

        residual = frame_residual_target(target, retrieved, config=ParseGraphTokenizerConfig())
        decoded = residual_values_to_frame(residual, retrieved, config=ParseGraphTokenizerConfig())

        self.assertAlmostEqual(decoded["origin"][0], 96.0, delta=1e-6)
        self.assertAlmostEqual(decoded["origin"][1], 32.0, delta=1e-6)
        self.assertAlmostEqual(decoded["scale"], 32.0, delta=1e-6)
        self.assertAlmostEqual(decoded["orientation"], -0.5, delta=1e-6)

    def test_residual_frame_clamps_scale_to_tokenizer_range(self) -> None:
        config = ParseGraphTokenizerConfig()
        retrieved = {"origin": [64.0, 96.0], "scale": 256.0, "orientation": 0.0}
        residual = [0.0, 0.0, 6.0, 0.0]

        raw_scale = residual_values_to_raw_scale(residual, retrieved)
        decoded = residual_values_to_frame(residual, retrieved, config=config)

        self.assertGreater(raw_scale, config.scale_max)
        self.assertEqual(decoded["scale"], config.scale_max)

    def test_residual_frame_clamps_scale_to_local_bbox(self) -> None:
        config = ParseGraphTokenizerConfig()
        retrieved = {"origin": [64.0, 96.0], "scale": 512.0, "orientation": 0.0}
        local_bbox = {"width": 2.0, "height": 1.0}

        scale_max = geometry_aware_scale_max(local_bbox, config=config, max_bbox_side=384.0)
        decoded = residual_values_to_frame(
            [0.0, 0.0, 0.0, 0.0],
            retrieved,
            config=config,
            local_bbox=local_bbox,
            max_bbox_side=384.0,
        )

        self.assertEqual(scale_max, 192.0)
        self.assertEqual(decoded["scale"], 192.0)

    def test_clamp_frame_to_local_bbox_reports_geometry_clamp(self) -> None:
        config = ParseGraphTokenizerConfig()
        frame = {"origin": [64.0, 96.0], "scale": 512.0, "orientation": 0.0}
        local_bbox = {"width": 4.0, "height": 1.0}

        clamped, diagnostics = clamp_frame_to_local_bbox(
            frame,
            local_bbox,
            config=config,
            max_bbox_side=384.0,
        )

        self.assertEqual(clamped["origin"], frame["origin"])
        self.assertEqual(clamped["orientation"], frame["orientation"])
        self.assertEqual(clamped["scale"], 96.0)
        self.assertTrue(diagnostics["geometry_scale_clamped"])
        self.assertTrue(diagnostics["geometry_frame_clamp_strong"])

    def test_local_bbox_quality_marks_tiny_world_bbox(self) -> None:
        quality = local_bbox_quality(
            {"width": 0.1, "height": 0.1},
            {"origin": [64.0, 96.0], "scale": 3.0, "orientation": 0.0},
            min_world_bbox_area=1.0,
        )

        self.assertFalse(quality["usable"])
        self.assertIn("tiny_world_bbox", quality["reasons"])

    def test_local_bbox_quality_marks_off_canvas_bbox(self) -> None:
        quality = local_bbox_quality(
            {"min_x": -0.5, "min_y": -0.5, "max_x": 0.5, "max_y": 0.5, "width": 1.0, "height": 1.0},
            {"origin": [512.0, 512.0], "scale": 8.0, "orientation": 0.0},
            canvas_size=[256, 256],
            min_world_bbox_area=1.0,
        )

        self.assertFalse(quality["usable"])
        self.assertIn("off_canvas_bbox", quality["reasons"])

    def test_geometry_shape_fallback_library_selects_true_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            _write_split_row(
                train,
                stem="train_a",
                topology=_topology_target(stem="train_a"),
                origins=[[128.0, 128.0], [96.0, 96.0]],
            )

            library, summary = build_geometry_shape_fallback_library(train)
            shape, mode = select_fallback_geometry_shape(
                {
                    "role": "support_region",
                    "label": 0,
                    "geometry_model": "polygon_code",
                },
                library,
            )
            geometry = geometry_target_from_fallback_shape(
                shape,
                source_node_id="query_support",
                frame={"origin": [64.0, 64.0], "scale": 8.0, "orientation": 0.0},
            )

            self.assertEqual(summary["shape_count"], 2)
            self.assertEqual(mode, "fallback_true_shape_exact")
            self.assertEqual(geometry["source_node_id"], "query_support")
            self.assertEqual(geometry["geometry"]["outer_local"][0], [-0.5, -0.5])

    def test_dataset_builds_retrieval_residual_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            val = Path(tmpdir) / "split" / "val"
            _write_split_row(
                train,
                stem="train_a",
                topology=_topology_target(stem="train_a"),
                origins=[[128.0, 128.0], [96.0, 96.0]],
            )
            _write_split_row(
                val,
                stem="val_a",
                topology=_topology_target(stem="val_a"),
                origins=[[120.0, 112.0], [88.0, 80.0]],
            )
            library, _summary = build_layout_retrieval_library(train)
            dataset = ManualLayoutResidualDataset(
                val,
                library_entries=library,
                fallback_frames=build_layout_retrieval_fallbacks(library),
            )

            row = dataset.rows[0]
            decoded = residual_values_to_frame(row["target_residual"], row["retrieved_frame"])

            self.assertEqual(len(dataset), 2)
            self.assertEqual(row["mapping_mode"], "retrieved_exact_order")
            self.assertEqual(row["local_bbox"]["width"], 1.0)
            self.assertAlmostEqual(decoded["origin"][0], row["target_frame"]["origin"][0], delta=1e-6)
            self.assertAlmostEqual(decoded["origin"][1], row["target_frame"]["origin"][1], delta=1e-6)

    def test_model_forward_evaluate_and_attach(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "split" / "train"
            val = Path(tmpdir) / "split" / "val"
            _write_split_row(
                train,
                stem="train_a",
                topology=_topology_target(stem="train_a"),
                origins=[[128.0, 128.0], [96.0, 96.0]],
            )
            _write_split_row(
                val,
                stem="val_a",
                topology=_topology_target(stem="val_a"),
                origins=[[120.0, 112.0], [88.0, 80.0]],
            )
            library, _summary = build_layout_retrieval_library(train)
            fallback = build_layout_retrieval_fallbacks(library)
            dataset = ManualLayoutResidualDataset(val, library_entries=library, fallback_frames=fallback)
            batch = collate_layout_residual_examples([dataset[0], dataset[1]])
            model = ManualLayoutResidualRegressor(
                ManualLayoutResidualRegressorConfig(
                    numeric_dim=dataset.numeric_dim,
                    retrieved_frame_dim=dataset.retrieved_frame_dim,
                    hidden_dim=32,
                    num_layers=1,
                )
            )

            predictions = model(batch)
            loss = layout_residual_loss(predictions, batch)
            loss.backward()
            metrics = evaluate_layout_residual_regressor(
                model,
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    collate_fn=collate_layout_residual_examples,
                ),
                device=torch.device("cpu"),
                config=ParseGraphTokenizerConfig(),
            )
            targets = attach_retrieved_residual_layout_to_split_targets(
                val,
                library_entries=library,
                fallback_frames=fallback,
                model=ZeroResidualModel(),
                tokenizer_config=ParseGraphTokenizerConfig(),
                device=torch.device("cpu"),
            )

            nodes_by_id = {node["id"]: node for node in targets[0]["parse_graph"]["nodes"]}
            self.assertEqual(predictions.shape, (2, 4))
            self.assertTrue(float(loss.item()) >= 0.0)
            self.assertIn("baseline_origin_mae", metrics)
            self.assertIn("scale_out_of_range_count", metrics)
            self.assertIn("bbox_huge_count", metrics)
            self.assertIn("geometry_scale_clamped_count", metrics)
            self.assertEqual(targets[0]["metadata"]["attached_geometry_count"], 2)
            self.assertEqual(nodes_by_id["val_a_support"]["frame"]["origin"], [128.0, 128.0])
            self.assertEqual(nodes_by_id["val_a_insert"]["geometry"]["outer_local"][0], [-0.5, -0.5])

    def test_prediction_example_can_be_built_without_target_frame(self) -> None:
        topology = _topology_target(stem="query")
        example = build_layout_residual_example(
            topology,
            node_index=0,
            retrieved_frame={"origin": [64.0, 64.0], "scale": 16.0, "orientation": 0.0},
            retrieval_score=3.0,
            mapping_mode="retrieved_exact_order",
            config=ParseGraphTokenizerConfig(),
        )

        self.assertNotIn("target_residual", example)
        self.assertEqual(example["mapping_mode_id"], 0)
        self.assertEqual(len(example["retrieved_frame_values"]), 6)


if __name__ == "__main__":
    unittest.main()
