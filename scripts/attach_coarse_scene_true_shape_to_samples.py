from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_coarse_scene_ar import decode_coarse_scene_tokens_to_target  # noqa: E402
from partition_gen.manual_geometry_shape_fallback import (  # noqa: E402
    build_geometry_shape_fallback_library,
    geometry_target_from_fallback_shape,
    geometry_target_quality,
    select_fallback_geometry_shape,
)
from partition_gen.manual_layout_residual import geometry_local_bbox  # noqa: E402
from partition_gen.manual_layout_retrieval import write_jsonl  # noqa: E402
from partition_gen.manual_topology_placeholder_geometry import iter_jsonl  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach split true-shape fallbacks to coarse-scene sampled frames.")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL rows containing coarse scene tokens.")
    parser.add_argument("--library-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--min-true-shape-world-bbox-area", type=float, default=1.0)
    parser.add_argument("--min-true-shape-local-bbox-side", type=float, default=1e-6)
    parser.add_argument("--scale-fit-mode", type=str, default="cover", choices=["cover", "contain", "frame"])
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "max": float(max(floats)),
    }


def _bbox_metrics(bbox: list[float]) -> dict:
    width = max(1e-6, float(bbox[2]) - float(bbox[0]))
    height = max(1e-6, float(bbox[3]) - float(bbox[1]))
    return {
        "width": float(width),
        "height": float(height),
        "center_x": float((float(bbox[0]) + float(bbox[2])) / 2.0),
        "center_y": float((float(bbox[1]) + float(bbox[3])) / 2.0),
    }


def _fit_frame_to_shape_bbox(frame: dict, coarse_bbox: list[float] | None, local_bbox: dict, *, mode: str) -> dict:
    if coarse_bbox is None or str(mode) == "frame":
        return copy.deepcopy(frame)
    metrics = _bbox_metrics(coarse_bbox)
    local_width = abs(float(local_bbox.get("width", 1.0)))
    local_height = abs(float(local_bbox.get("height", 1.0)))
    if local_width <= 1e-6 or local_height <= 1e-6:
        return copy.deepcopy(frame)
    scale_x = metrics["width"] / local_width
    scale_y = metrics["height"] / local_height
    scale = max(scale_x, scale_y) if str(mode) == "cover" else min(scale_x, scale_y)
    local_center_x = float(local_bbox.get("min_x", -local_width / 2.0)) + local_width / 2.0
    local_center_y = float(local_bbox.get("min_y", -local_height / 2.0)) + local_height / 2.0
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    offset_x = (local_center_x * cos_theta - local_center_y * sin_theta) * scale
    offset_y = (local_center_x * sin_theta + local_center_y * cos_theta) * scale
    output = copy.deepcopy(frame)
    output["origin"] = [float(metrics["center_x"] - offset_x), float(metrics["center_y"] - offset_y)]
    output["scale"] = float(max(1.0, scale))
    output["orientation"] = float(theta)
    return output


def main() -> None:
    args = parse_args()
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]
    shape_library, shape_summary = build_geometry_shape_fallback_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
        min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
    )
    manifest_rows: list[dict] = []
    shape_modes: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    attached_total = 0
    missing_total = 0
    quality_failure_total = 0
    scale_values: list[float] = []
    tokenizer_config = ParseGraphTokenizerConfig()

    for fallback_index, row in enumerate(sample_rows):
        sample_index = int(row.get("sample_index", fallback_index))
        try:
            target = decode_coarse_scene_tokens_to_target([str(token) for token in row.get("tokens", []) or []], config=tokenizer_config)
        except Exception as exc:
            error_histogram[type(exc).__name__] += 1
            continue
        graph = target.get("parse_graph", {}) or {}
        output_nodes: list[dict] = []
        geometry_rows: list[dict] = []
        sample_attached = 0
        sample_missing = 0
        sample_quality_failure = 0
        sample_shape_modes: Counter[str] = Counter()
        sample_quality_reasons: Counter[str] = Counter()

        for node in graph.get("nodes", []) or []:
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref:
                geometry_row = {
                    "node_id": str(output_node.get("id", "")),
                    "role": str(output_node.get("role", "")),
                    "label": int(output_node.get("label", 0)),
                    "geometry_model": str(output_node.get("geometry_model", "none")),
                }
                try:
                    shape, shape_mode = select_fallback_geometry_shape(output_node, shape_library)
                    if shape is None:
                        raise RuntimeError("true shape library produced no candidate")
                    local_bbox = copy.deepcopy(shape.get("local_bbox") or {})
                    if not local_bbox:
                        probe = geometry_target_from_fallback_shape(
                            shape,
                            source_node_id=str(geometry_ref),
                            frame=output_node.get("frame", {}),
                        )
                        local_bbox = geometry_local_bbox(probe)
                    final_frame = _fit_frame_to_shape_bbox(
                        output_node.get("frame", {}),
                        output_node.get("coarse_bbox"),
                        local_bbox,
                        mode=str(args.scale_fit_mode),
                    )
                    generated = geometry_target_from_fallback_shape(shape, source_node_id=str(geometry_ref), frame=final_frame)
                    quality = geometry_target_quality(
                        generated,
                        final_frame,
                        canvas_size=target.get("size", [256, 256]),
                        min_world_bbox_area=float(args.min_true_shape_world_bbox_area),
                        min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
                    )
                    if not bool(quality.get("usable", False)):
                        quality_failure_total += 1
                        sample_quality_failure += 1
                        quality_reasons.update(quality.get("reasons", []) or [])
                        sample_quality_reasons.update(quality.get("reasons", []) or [])
                    shape_modes[shape_mode] += 1
                    sample_shape_modes[shape_mode] += 1
                    scale_values.append(float(final_frame.get("scale", 1.0)))
                    output_node["geometry_model"] = copy.deepcopy(generated.get("geometry_model", output_node.get("geometry_model")))
                    output_node["frame"] = copy.deepcopy(final_frame)
                    output_node["true_shape_local_bbox"] = local_bbox
                    output_node["local_bbox_quality"] = quality
                    output_node["geometry_fallback_mode"] = shape_mode
                    if "geometry" in generated:
                        output_node["geometry"] = copy.deepcopy(generated["geometry"])
                    if "atoms" in generated:
                        output_node["atoms"] = copy.deepcopy(generated["atoms"])
                    output_node["layout_frame_source"] = "coarse_scene"
                    output_node["layout_shape_attach_mode"] = "true_shape_fallback"
                    geometry_row.update(
                        {
                            "valid": True,
                            "final_frame": copy.deepcopy(final_frame),
                            "true_shape_local_bbox": local_bbox,
                            "local_bbox_quality": quality,
                            "shape_mode": shape_mode,
                        }
                    )
                    sample_attached += 1
                    attached_total += 1
                except Exception as exc:
                    sample_missing += 1
                    missing_total += 1
                    error_histogram[type(exc).__name__] += 1
                    geometry_row.update({"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]})
                    output_node["coarse_scene_true_shape_error"] = f"{type(exc).__name__}: {exc}"
                geometry_rows.append(geometry_row)
            output_nodes.append(output_node)

        attached_target = {
            "format": "maskgen_generator_target_v1",
            "target_type": "parse_graph",
            "size": copy.deepcopy(target.get("size", [256, 256])),
            "parse_graph": {
                "nodes": output_nodes,
                "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
                "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
            },
            "metadata": {
                "coarse_scene_true_shape": True,
                "sample_index": int(sample_index),
                "checkpoint": row.get("checkpoint"),
                "geometry_valid_count": int(sample_attached),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "true_shape_modes": dict(sample_shape_modes),
                "true_shape_quality_failure_count": int(sample_quality_failure),
                "true_shape_quality_reasons": dict(sample_quality_reasons),
                "geometry_rows": geometry_rows,
                "scale_fit_mode": str(args.scale_fit_mode),
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, attached_target)
        manifest_rows.append(
            {
                "sample_index": int(sample_index),
                "output_path": str(output_path.as_posix()),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "true_shape_quality_failure_count": int(sample_quality_failure),
            }
        )
        if int(args.progress_every) > 0 and len(manifest_rows) % int(args.progress_every) == 0:
            print(f"coarse_scene_true_shape_attach {len(manifest_rows)}/{len(sample_rows)}", flush=True)

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_coarse_scene_true_shape_attach_summary_v1",
        "samples": str(args.samples.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(sample_rows)),
        "output_count": int(len(manifest_rows)),
        "shape_fallback_summary": shape_summary,
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "true_shape_modes": dict(shape_modes),
        "true_shape_quality_failure_count": int(quality_failure_total),
        "true_shape_quality_reasons": dict(quality_reasons),
        "final_scale_stats": _numeric_stats(scale_values),
        "error_histogram": dict(error_histogram),
        "scale_fit_mode": str(args.scale_fit_mode),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached coarse-scene true-shape samples={summary['output_count']} "
        f"attached={attached_total} missing={missing_total} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
