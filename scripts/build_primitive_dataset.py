from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json
from partition_gen.primitive_decomposition import (
    CompositeGroupConfig,
    PrimitiveCompressionConfig,
    StripCoverConfig,
    StripRefineConfig,
    build_composite_groups,
    build_strip_cover,
    compress_primitives,
    decompose_partition_face,
    face_geometry,
    geometry_iou,
    primitives_union_geometry,
    refine_strip_cover,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate each partition face with a small set of triangles/quads.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--output-root", type=Path, default=Path("data/remote_256_primitives"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--compression-max-group-size", type=int, default=8)
    parser.add_argument("--compression-min-iou", type=float, default=0.72)
    parser.add_argument("--compression-error-weight", type=float, default=3.5)
    parser.add_argument("--cover-min-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--cover-max-angle-delta-deg", type=float, default=18.0)
    parser.add_argument("--cover-normal-offset-scale", type=float, default=1.35)
    parser.add_argument("--cover-width-scale", type=float, default=1.1)
    parser.add_argument("--cover-min-precision", type=float, default=0.72)
    parser.add_argument("--cover-min-support-ratio", type=float, default=0.72)
    parser.add_argument("--refine-max-group-size", type=int, default=4)
    parser.add_argument("--refine-min-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--refine-max-angle-delta-deg", type=float, default=18.0)
    parser.add_argument("--refine-normal-offset-scale", type=float, default=1.75)
    parser.add_argument("--refine-width-scale", type=float, default=1.1)
    parser.add_argument("--refine-merge-error-weight", type=float, default=0.55)
    parser.add_argument("--refine-min-candidate-quality", type=float, default=0.6)
    parser.add_argument("--composite-max-candidate-area-ratio", type=float, default=0.5)
    parser.add_argument("--composite-normal-offset-scale", type=float, default=2.0)
    parser.add_argument("--composite-endpoint-margin-scale", type=float, default=3.0)
    parser.add_argument("--composite-area-gain-weight", type=float, default=1.0)
    parser.add_argument("--composite-edge-cost-weight", type=float, default=0.45)
    parser.add_argument("--composite-connectivity-bonus-weight", type=float, default=0.35)
    parser.add_argument("--composite-fit-gain-weight", type=float, default=0.5)
    parser.add_argument("--composite-hole-invasion-weight", type=float, default=2.5)
    parser.add_argument("--composite-hole-loss-weight", type=float, default=0.75)
    parser.add_argument("--composite-min-score", type=float, default=0.02)
    parser.add_argument("--composite-max-fit-drop", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), indent=2)


def process_graph(
    path: Path,
    *,
    simplify_tolerance: float,
    compression_config: PrimitiveCompressionConfig,
    cover_config: StripCoverConfig,
    refine_config: StripRefineConfig,
    composite_config: CompositeGroupConfig,
) -> Dict[str, object]:
    graph_data = load_json(path)
    faces = []
    for face_data in graph_data["faces"]:
        original_geometry = face_geometry(graph_data, face_data)
        face_payload = decompose_partition_face(
            graph_data,
            face_data,
            simplify_tolerance=simplify_tolerance,
        )
        compressed = compress_primitives(face_payload["primitives"], config=compression_config)
        compressed_geometry = primitives_union_geometry(compressed["primitives"])
        compressed["approx_iou"] = float(geometry_iou(original_geometry, compressed_geometry)) if not compressed_geometry.is_empty else 0.0
        face_payload["compressed"] = compressed
        face_payload["strip_cover"] = build_strip_cover(
            compressed["primitives"],
            reference_geometry=original_geometry,
            config=cover_config,
        )
        face_payload["strip_refined"] = refine_strip_cover(
            face_payload["strip_cover"]["primitives"],
            reference_geometry=original_geometry,
            config=refine_config,
        )
        face_payload["composite_groups"] = build_composite_groups(
            face_payload["strip_refined"]["primitives"],
            compressed["primitives"],
            face_payload["primitives"],
            reference_geometry=original_geometry,
            config=composite_config,
        )
        faces.append(face_payload)

    primitive_counts = [face["primitive_count"] for face in faces]
    approx_ious = [face["approx_iou"] for face in faces]
    triangle_counts = [face["triangle_count"] for face in faces]
    quad_counts = [face["quad_count"] for face in faces]
    compressed_counts = [face["compressed"]["primitive_count"] for face in faces]
    compressed_triangle_counts = [face["compressed"]["triangle_count"] for face in faces]
    compressed_quad_counts = [face["compressed"]["quad_count"] for face in faces]
    strip_cover_counts = [face["strip_cover"]["primitive_count"] for face in faces]
    strip_cover_triangle_counts = [face["strip_cover"]["triangle_count"] for face in faces]
    strip_cover_quad_counts = [face["strip_cover"]["quad_count"] for face in faces]
    strip_cover_convex_counts = [face["strip_cover"].get("convex_count", 0) for face in faces]
    strip_refined_counts = [face["strip_refined"]["primitive_count"] for face in faces]
    strip_refined_triangle_counts = [face["strip_refined"]["triangle_count"] for face in faces]
    strip_refined_quad_counts = [face["strip_refined"]["quad_count"] for face in faces]
    strip_refined_convex_counts = [face["strip_refined"].get("convex_count", 0) for face in faces]
    composite_group_counts = [face["composite_groups"]["group_count"] for face in faces]
    composite_atom_counts = [face["composite_groups"]["mean_atom_count"] for face in faces]
    composite_hole_counts = [face["composite_groups"]["mean_hole_count"] for face in faces]

    return {
        "source_partition_graph": str(path.as_posix()),
        "source_mask": graph_data.get("source_mask"),
        "size": graph_data["size"],
        "simplify_tolerance": float(simplify_tolerance),
        "faces": faces,
        "stats": {
            "num_faces": int(len(faces)),
            "mean_primitives_per_face": float(np.mean(primitive_counts)) if primitive_counts else 0.0,
            "mean_approx_iou": float(np.mean(approx_ious)) if approx_ious else 0.0,
            "mean_triangles_per_face": float(np.mean(triangle_counts)) if triangle_counts else 0.0,
            "mean_quads_per_face": float(np.mean(quad_counts)) if quad_counts else 0.0,
            "max_primitives_per_face": int(max(primitive_counts)) if primitive_counts else 0,
            "mean_compressed_primitives_per_face": float(np.mean(compressed_counts)) if compressed_counts else 0.0,
            "mean_compressed_triangles_per_face": float(np.mean(compressed_triangle_counts))
            if compressed_triangle_counts
            else 0.0,
            "mean_compressed_quads_per_face": float(np.mean(compressed_quad_counts))
            if compressed_quad_counts
            else 0.0,
            "max_compressed_primitives_per_face": int(max(compressed_counts)) if compressed_counts else 0,
            "mean_strip_cover_primitives_per_face": float(np.mean(strip_cover_counts)) if strip_cover_counts else 0.0,
            "mean_strip_cover_triangles_per_face": float(np.mean(strip_cover_triangle_counts))
            if strip_cover_triangle_counts
            else 0.0,
            "mean_strip_cover_quads_per_face": float(np.mean(strip_cover_quad_counts))
            if strip_cover_quad_counts
            else 0.0,
            "mean_strip_cover_convex_per_face": float(np.mean(strip_cover_convex_counts))
            if strip_cover_convex_counts
            else 0.0,
            "max_strip_cover_primitives_per_face": int(max(strip_cover_counts)) if strip_cover_counts else 0,
            "mean_strip_refined_primitives_per_face": float(np.mean(strip_refined_counts))
            if strip_refined_counts
            else 0.0,
            "mean_strip_refined_triangles_per_face": float(np.mean(strip_refined_triangle_counts))
            if strip_refined_triangle_counts
            else 0.0,
            "mean_strip_refined_quads_per_face": float(np.mean(strip_refined_quad_counts))
            if strip_refined_quad_counts
            else 0.0,
            "mean_strip_refined_convex_per_face": float(np.mean(strip_refined_convex_counts))
            if strip_refined_convex_counts
            else 0.0,
            "max_strip_refined_primitives_per_face": int(max(strip_refined_counts)) if strip_refined_counts else 0,
            "mean_composite_groups_per_face": float(np.mean(composite_group_counts)) if composite_group_counts else 0.0,
            "mean_composite_atoms_per_group": float(np.mean(composite_atom_counts)) if composite_atom_counts else 0.0,
            "mean_composite_holes_per_group": float(np.mean(composite_hole_counts)) if composite_hole_counts else 0.0,
        },
    }


def process_split(
    split: str,
    *,
    partition_root: Path,
    output_root: Path,
    simplify_tolerance: float,
    max_samples: int | None,
    compression_config: PrimitiveCompressionConfig,
    cover_config: StripCoverConfig,
    refine_config: StripRefineConfig,
    composite_config: CompositeGroupConfig,
) -> Dict[str, object]:
    graph_paths = sorted((partition_root / split / "graphs").glob("*.json"))
    if max_samples is not None:
        graph_paths = graph_paths[:max_samples]

    summary_rows: List[Dict[str, float]] = []
    for index, graph_path in enumerate(graph_paths, start=1):
        payload = process_graph(
            graph_path,
            simplify_tolerance=simplify_tolerance,
            compression_config=compression_config,
            cover_config=cover_config,
            refine_config=refine_config,
            composite_config=composite_config,
        )
        dump_json(output_root / split / "graphs" / graph_path.name, payload)
        summary_rows.append(payload["stats"])
        if index % 50 == 0 or index == len(graph_paths):
            print(
                f"[{split}] processed {index}/{len(graph_paths)} "
                f"(mean primitives/face={np.mean([row['mean_primitives_per_face'] for row in summary_rows]):.2f}, "
                f"compressed={np.mean([row['mean_compressed_primitives_per_face'] for row in summary_rows]):.2f}, "
                f"strip-cover={np.mean([row['mean_strip_cover_primitives_per_face'] for row in summary_rows]):.2f}, "
                f"strip-refined={np.mean([row['mean_strip_refined_primitives_per_face'] for row in summary_rows]):.2f}, "
                f"composite-groups={np.mean([row['mean_composite_groups_per_face'] for row in summary_rows]):.2f}, "
                f"mean IoU={np.mean([row['mean_approx_iou'] for row in summary_rows]):.3f})"
            )

    return {
        "split": split,
        "num_graphs": int(len(graph_paths)),
        "mean_primitives_per_face": float(np.mean([row["mean_primitives_per_face"] for row in summary_rows]))
        if summary_rows
        else 0.0,
        "mean_approx_iou": float(np.mean([row["mean_approx_iou"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_triangles_per_face": float(np.mean([row["mean_triangles_per_face"] for row in summary_rows]))
        if summary_rows
        else 0.0,
        "mean_quads_per_face": float(np.mean([row["mean_quads_per_face"] for row in summary_rows]))
        if summary_rows
        else 0.0,
        "max_primitives_per_face": int(max([row["max_primitives_per_face"] for row in summary_rows])) if summary_rows else 0,
        "mean_compressed_primitives_per_face": float(
            np.mean([row["mean_compressed_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_compressed_triangles_per_face": float(
            np.mean([row["mean_compressed_triangles_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_compressed_quads_per_face": float(
            np.mean([row["mean_compressed_quads_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "max_compressed_primitives_per_face": int(
            max([row["max_compressed_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0,
        "mean_strip_cover_primitives_per_face": float(
            np.mean([row["mean_strip_cover_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_strip_cover_triangles_per_face": float(
            np.mean([row["mean_strip_cover_triangles_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_strip_cover_quads_per_face": float(
            np.mean([row["mean_strip_cover_quads_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "max_strip_cover_primitives_per_face": int(
            max([row["max_strip_cover_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0,
        "mean_strip_refined_primitives_per_face": float(
            np.mean([row["mean_strip_refined_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_strip_refined_triangles_per_face": float(
            np.mean([row["mean_strip_refined_triangles_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_strip_refined_quads_per_face": float(
            np.mean([row["mean_strip_refined_quads_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "max_strip_refined_primitives_per_face": int(
            max([row["max_strip_refined_primitives_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0,
        "mean_composite_groups_per_face": float(
            np.mean([row["mean_composite_groups_per_face"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_composite_atoms_per_group": float(
            np.mean([row["mean_composite_atoms_per_group"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
        "mean_composite_holes_per_group": float(
            np.mean([row["mean_composite_holes_per_group"] for row in summary_rows])
        )
        if summary_rows
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    compression_config = PrimitiveCompressionConfig(
        max_group_size=args.compression_max_group_size,
        min_iou=args.compression_min_iou,
        error_weight=args.compression_error_weight,
    )
    cover_config = StripCoverConfig(
        min_aspect_ratio=args.cover_min_aspect_ratio,
        max_angle_delta_deg=args.cover_max_angle_delta_deg,
        normal_offset_scale=args.cover_normal_offset_scale,
        width_scale=args.cover_width_scale,
        min_precision=args.cover_min_precision,
        min_support_ratio=args.cover_min_support_ratio,
    )
    refine_config = StripRefineConfig(
        max_group_size=args.refine_max_group_size,
        min_aspect_ratio=args.refine_min_aspect_ratio,
        max_angle_delta_deg=args.refine_max_angle_delta_deg,
        normal_offset_scale=args.refine_normal_offset_scale,
        width_scale=args.refine_width_scale,
        merge_error_weight=args.refine_merge_error_weight,
        min_candidate_quality=args.refine_min_candidate_quality,
    )
    composite_config = CompositeGroupConfig(
        max_candidate_area_ratio=args.composite_max_candidate_area_ratio,
        normal_offset_scale=args.composite_normal_offset_scale,
        endpoint_margin_scale=args.composite_endpoint_margin_scale,
        area_gain_weight=args.composite_area_gain_weight,
        edge_cost_weight=args.composite_edge_cost_weight,
        connectivity_bonus_weight=args.composite_connectivity_bonus_weight,
        fit_gain_weight=args.composite_fit_gain_weight,
        hole_invasion_weight=args.composite_hole_invasion_weight,
        hole_loss_weight=args.composite_hole_loss_weight,
        min_score=args.composite_min_score,
        max_fit_drop=args.composite_max_fit_drop,
    )

    split_summaries = []
    for split in args.splits:
        split_summaries.append(
            process_split(
                split,
                partition_root=args.partition_root,
                output_root=args.output_root,
                simplify_tolerance=args.simplify_tolerance,
                max_samples=args.max_samples,
                compression_config=compression_config,
                cover_config=cover_config,
                refine_config=refine_config,
                composite_config=composite_config,
            )
        )

    dump_json(
        args.output_root / "meta" / "summary.json",
        {
            "partition_root": str(args.partition_root.as_posix()),
            "output_root": str(args.output_root.as_posix()),
            "simplify_tolerance": float(args.simplify_tolerance),
            "compression": {
                "max_group_size": int(compression_config.max_group_size),
                "min_iou": float(compression_config.min_iou),
                "error_weight": float(compression_config.error_weight),
            },
            "strip_cover": {
                "min_aspect_ratio": float(cover_config.min_aspect_ratio),
                "max_angle_delta_deg": float(cover_config.max_angle_delta_deg),
                "normal_offset_scale": float(cover_config.normal_offset_scale),
                "width_scale": float(cover_config.width_scale),
                "min_precision": float(cover_config.min_precision),
                "min_support_ratio": float(cover_config.min_support_ratio),
            },
            "strip_refine": {
                "max_group_size": int(refine_config.max_group_size),
                "min_aspect_ratio": float(refine_config.min_aspect_ratio),
                "max_angle_delta_deg": float(refine_config.max_angle_delta_deg),
                "normal_offset_scale": float(refine_config.normal_offset_scale),
                "width_scale": float(refine_config.width_scale),
                "merge_error_weight": float(refine_config.merge_error_weight),
                "min_candidate_quality": float(refine_config.min_candidate_quality),
            },
            "composite_groups": {
                "max_candidate_area_ratio": float(composite_config.max_candidate_area_ratio),
                "normal_offset_scale": float(composite_config.normal_offset_scale),
                "endpoint_margin_scale": float(composite_config.endpoint_margin_scale),
                "area_gain_weight": float(composite_config.area_gain_weight),
                "edge_cost_weight": float(composite_config.edge_cost_weight),
                "connectivity_bonus_weight": float(composite_config.connectivity_bonus_weight),
                "fit_gain_weight": float(composite_config.fit_gain_weight),
                "min_score": float(composite_config.min_score),
                "max_fit_drop": float(composite_config.max_fit_drop),
            },
            "splits": split_summaries,
        },
    )
    print(json.dumps({"splits": split_summaries}, indent=2))


if __name__ == "__main__":
    main()
