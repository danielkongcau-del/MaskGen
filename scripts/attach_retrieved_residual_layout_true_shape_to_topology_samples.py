from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import sys
from pathlib import Path
from statistics import mean, median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_geometry_conditioning import renderable_geometry_node_indices  # noqa: E402
from partition_gen.manual_geometry_shape_fallback import (  # noqa: E402
    build_geometry_shape_fallback_library,
    geometry_target_from_fallback_shape,
    geometry_target_quality,
    select_fallback_geometry_shape,
)
from partition_gen.manual_layout_residual import (  # noqa: E402
    build_layout_residual_example,
    clamp_frame_to_local_bbox,
    geometry_local_bbox,
    load_layout_residual_checkpoint,
    predict_residual_frame,
)
from partition_gen.manual_layout_retrieval import (  # noqa: E402
    build_layout_retrieval_fallbacks,
    build_layout_retrieval_library,
    retrieve_layout_entry,
    write_jsonl,
)
from partition_gen.manual_topology_placeholder_geometry import decode_topology_tokens_to_target, iter_jsonl  # noqa: E402
from partition_gen.manual_topology_sample_validation import validate_topology_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach residual-corrected retrieved-layout frames and retrieved true local shapes "
            "to generated topology samples."
        )
    )
    parser.add_argument("--samples", type=Path, required=True, help="JSONL topology samples containing token rows.")
    parser.add_argument("--library-split-root", type=Path, required=True, help="Retrieval/shape library split root, usually train.")
    parser.add_argument("--layout-residual-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--exclude-same-stem", action="store_true")
    parser.add_argument("--min-true-shape-world-bbox-area", type=float, default=1.0)
    parser.add_argument("--min-true-shape-local-bbox-side", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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


def _node_key(node: dict, *, level: str) -> tuple:
    role = str(node.get("role", ""))
    label = int(node.get("label", 0))
    geometry_model = str(node.get("geometry_model", "polygon_code"))
    if level == "exact":
        return role, label, geometry_model
    if level == "role_label":
        return role, label
    if level == "role":
        return (role,)
    return ()


def _frame_clamp_mode(diagnostics: dict) -> str:
    if diagnostics.get("geometry_frame_clamp_strong"):
        return "strong_geometry_clamp"
    if diagnostics.get("geometry_scale_clamped"):
        return "geometry_clamp"
    if diagnostics.get("tokenizer_scale_clamped"):
        return "tokenizer_clamp"
    if diagnostics.get("scale_clamped"):
        return "scale_clamp"
    return "none"


def _choose_fallback_frame(node: dict, fallback_frames: dict) -> tuple[str, dict]:
    for level in ("exact", "role_label", "role"):
        frame = (fallback_frames.get(level, {}) or {}).get(_node_key(node, level=level))
        if frame is not None:
            return f"fallback_{level}_median", frame
    return "fallback_global_median", fallback_frames["global"]


def _map_retrieved_layout_frames_and_rows(
    topology_target: dict,
    retrieved_entry: dict,
    *,
    fallback_frames: dict,
) -> tuple[dict[int, dict], dict[int, dict], dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    retrieved_rows = list(retrieved_entry.get("layout_rows", []) or [])
    used: set[int] = set()
    frame_by_node_index: dict[int, dict] = {}
    retrieved_row_by_node_index: dict[int, dict] = {}
    mapping_modes: Counter[str] = Counter()
    node_mapping_modes: dict[int, str] = {}

    def choose_row(node: dict, *, level: str) -> tuple[int, dict] | None:
        key = _node_key(node, level=level)
        for row_index, row in enumerate(retrieved_rows):
            if row_index in used:
                continue
            if _node_key(row, level=level) == key:
                return row_index, row
        return None

    for node_index in renderable_geometry_node_indices(topology_target):
        node = nodes[int(node_index)]
        selected = None
        selected_mode = ""
        for level in ("exact", "role_label", "role"):
            selected = choose_row(node, level=level)
            if selected is not None:
                selected_mode = f"retrieved_{level}_order"
                break
        if selected is not None:
            row_index, row = selected
            used.add(row_index)
            frame = row["frame"]
            retrieved_row_by_node_index[int(node_index)] = copy.deepcopy(row)
        else:
            selected_mode, frame = _choose_fallback_frame(node, fallback_frames)
        frame_by_node_index[int(node_index)] = copy.deepcopy(frame)
        mapping_modes[selected_mode] += 1
        node_mapping_modes[int(node_index)] = str(selected_mode)

    diagnostics = {
        "mapping_mode_histogram": dict(mapping_modes),
        "node_mapping_modes": {str(key): value for key, value in sorted(node_mapping_modes.items())},
        "mapped_frame_count": int(len(frame_by_node_index)),
        "retrieved_frame_count": int(len(retrieved_rows)),
        "unused_retrieved_frame_count": int(len(retrieved_rows) - len(used)),
    }
    return frame_by_node_index, retrieved_row_by_node_index, diagnostics


def _shape_from_retrieved_row(
    *,
    retrieved_entry: dict,
    retrieved_row: dict | None,
    query_node: dict,
    shape_library: dict,
) -> tuple[dict | None, str]:
    if retrieved_row is not None:
        source_key = (
            str(retrieved_entry.get("stem")),
            str(retrieved_row.get("geometry_ref", retrieved_row.get("source_node_id", ""))),
        )
        shape = (shape_library.get("source", {}) or {}).get(source_key)
        if shape is not None:
            return copy.deepcopy(shape), "retrieved_true_shape_source_row"
    return select_fallback_geometry_shape(query_node, shape_library)


def main() -> None:
    args = parse_args()
    _layout_checkpoint, layout_model, tokenizer_config = load_layout_residual_checkpoint(
        args.layout_residual_checkpoint,
        map_location="cpu",
    )
    device = torch.device(args.device)
    layout_model = layout_model.to(device)
    layout_model.eval()

    library_entries, library_summary = build_layout_retrieval_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
    )
    fallback_frames = build_layout_retrieval_fallbacks(library_entries)
    shape_library, shape_summary = build_geometry_shape_fallback_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
        min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
    )

    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]

    manifest_rows: list[dict] = []
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    frame_clamp_modes: Counter[str] = Counter()
    shape_modes: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    final_scale_ratios: list[float] = []
    skipped_invalid = 0
    request_total = 0
    valid_total = 0
    attached_total = 0
    missing_total = 0
    quality_failure_total = 0
    last_progress = 0

    for fallback_index, row in enumerate(sample_rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not bool(args.include_invalid):
            skipped_invalid += 1
            continue

        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target.setdefault("metadata", {}).update(
            {
                "sample_index": int(sample_index),
                "semantic_valid": bool(validation["semantic_valid"]),
                "checkpoint": row.get("checkpoint"),
            }
        )
        retrieved_entry, retrieval_score = retrieve_layout_entry(
            topology_target,
            library_entries,
            exclude_stem=row.get("stem") if bool(args.exclude_same_stem) else None,
        )
        frame_by_index, retrieved_row_by_index, mapping_diagnostics = _map_retrieved_layout_frames_and_rows(
            topology_target,
            retrieved_entry,
            fallback_frames=fallback_frames,
        )
        node_mapping_modes = {
            int(key): str(value)
            for key, value in (mapping_diagnostics.get("node_mapping_modes", {}) or {}).items()
        }
        retrieval_scores.append(float(retrieval_score))
        mapping_modes.update(mapping_diagnostics.get("mapping_mode_histogram", {}) or {})

        graph = topology_target.get("parse_graph", {}) or {}
        output_nodes: list[dict] = []
        geometry_rows: list[dict] = []
        sample_request = 0
        sample_valid = 0
        sample_attached = 0
        sample_missing = 0
        sample_quality_failure = 0
        sample_attach_modes: Counter[str] = Counter()
        sample_frame_clamp_modes: Counter[str] = Counter()
        sample_shape_modes: Counter[str] = Counter()
        sample_quality_reasons: Counter[str] = Counter()

        for node_index, node in enumerate(graph.get("nodes", []) or []):
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref:
                retrieved_frame = frame_by_index.get(int(node_index))
                geometry_row = {
                    "node_index": int(node_index),
                    "node_id": str(output_node.get("id", "")),
                    "role": str(output_node.get("role", "")),
                    "label": int(output_node.get("label", 0)),
                    "geometry_model": str(output_node.get("geometry_model", "none")),
                }
                if retrieved_frame is None:
                    sample_missing += 1
                    missing_total += 1
                    attach_modes["missing"] += 1
                    sample_attach_modes["missing"] += 1
                    geometry_row.update({"valid": False, "errors": ["missing retrieved frame"]})
                    output_node["retrieved_residual_true_shape_error"] = "missing retrieved frame"
                else:
                    request_total += 1
                    sample_request += 1
                    try:
                        shape, shape_mode = _shape_from_retrieved_row(
                            retrieved_entry=retrieved_entry,
                            retrieved_row=retrieved_row_by_index.get(int(node_index)),
                            query_node=output_node,
                            shape_library=shape_library,
                        )
                        if shape is None:
                            raise RuntimeError("true shape library produced no candidate")

                        true_shape_local_bbox = copy.deepcopy(shape.get("local_bbox") or {})
                        if not true_shape_local_bbox:
                            probe = geometry_target_from_fallback_shape(
                                shape,
                                source_node_id=str(geometry_ref),
                                frame=retrieved_frame,
                            )
                            true_shape_local_bbox = geometry_local_bbox(probe)

                        residual_example = build_layout_residual_example(
                            topology_target,
                            node_index=int(node_index),
                            retrieved_frame=retrieved_frame,
                            local_bbox=true_shape_local_bbox,
                            retrieval_score=float(retrieval_score),
                            mapping_mode=node_mapping_modes.get(int(node_index), "unknown"),
                            config=tokenizer_config,
                        )
                        refined_frame, residual = predict_residual_frame(
                            layout_model,
                            residual_example,
                            device=device,
                            config=tokenizer_config,
                        )
                        final_frame, clamp_diagnostics = clamp_frame_to_local_bbox(
                            refined_frame,
                            true_shape_local_bbox,
                            config=tokenizer_config,
                        )
                        generated = geometry_target_from_fallback_shape(
                            shape,
                            source_node_id=str(geometry_ref),
                            frame=final_frame,
                        )
                        quality = geometry_target_quality(
                            generated,
                            final_frame,
                            canvas_size=topology_target.get("size", [256, 256]),
                            min_world_bbox_area=float(args.min_true_shape_world_bbox_area),
                            min_local_bbox_side=float(args.min_true_shape_local_bbox_side),
                        )
                        if not bool(quality.get("usable", False)):
                            quality_failure_total += 1
                            sample_quality_failure += 1
                            quality_reasons.update(quality.get("reasons", []) or [])
                            sample_quality_reasons.update(quality.get("reasons", []) or [])

                        frame_clamp_mode = _frame_clamp_mode(clamp_diagnostics)
                        frame_clamp_modes[frame_clamp_mode] += 1
                        sample_frame_clamp_modes[frame_clamp_mode] += 1
                        shape_modes[shape_mode] += 1
                        sample_shape_modes[shape_mode] += 1
                        final_scale_ratios.append(float(clamp_diagnostics["scale_ratio"]))

                        attach_mode = (
                            "retrieved_residual_frame_retrieved_true_shape"
                            if shape_mode == "retrieved_true_shape_source_row"
                            else "retrieved_residual_frame_fallback_true_shape"
                        )
                        geometry_row.update(
                            {
                                "valid": True,
                                "retrieved_frame": copy.deepcopy(retrieved_frame),
                                "refined_frame": copy.deepcopy(refined_frame),
                                "final_frame": copy.deepcopy(final_frame),
                                "layout_residual": residual,
                                "true_shape_local_bbox": true_shape_local_bbox,
                                "generated_local_bbox": true_shape_local_bbox,
                                "frame_clamp": clamp_diagnostics,
                                "frame_clamp_mode": frame_clamp_mode,
                                "local_bbox_quality": quality,
                                "shape_mode": shape_mode,
                                "shape_source_stem": shape.get("source_stem"),
                                "shape_source_node_id": shape.get("source_node_id"),
                            }
                        )
                        output_node["geometry_model"] = copy.deepcopy(
                            generated.get("geometry_model", output_node.get("geometry_model"))
                        )
                        output_node["frame"] = copy.deepcopy(final_frame)
                        output_node["retrieved_frame"] = copy.deepcopy(retrieved_frame)
                        output_node["refined_frame"] = copy.deepcopy(refined_frame)
                        output_node["layout_residual"] = residual
                        output_node["generated_local_bbox"] = true_shape_local_bbox
                        output_node["true_shape_local_bbox"] = true_shape_local_bbox
                        output_node["frame_clamp"] = clamp_diagnostics
                        output_node["local_bbox_quality"] = quality
                        output_node["geometry_fallback_mode"] = shape_mode
                        if "geometry" in generated:
                            output_node["geometry"] = copy.deepcopy(generated["geometry"])
                        if "atoms" in generated:
                            output_node["atoms"] = copy.deepcopy(generated["atoms"])
                        output_node["layout_frame_source"] = "retrieved_residual_layout"
                        output_node["layout_shape_attach_mode"] = "true_shape_ablation"
                        output_node["retrieved_residual_frame_true_shape"] = True

                        sample_valid += 1
                        valid_total += 1
                        sample_attached += 1
                        attached_total += 1
                        attach_modes[attach_mode] += 1
                        sample_attach_modes[attach_mode] += 1
                    except Exception as exc:
                        sample_missing += 1
                        missing_total += 1
                        attach_modes["missing"] += 1
                        sample_attach_modes["missing"] += 1
                        error_key = type(exc).__name__
                        error_histogram[error_key] += 1
                        geometry_row.update({"valid": False, "errors": [f"{error_key}: {exc}"]})
                        output_node["retrieved_residual_true_shape_error"] = f"{error_key}: {exc}"
                geometry_rows.append(geometry_row)
                if (
                    int(args.progress_every) > 0
                    and request_total > 0
                    and request_total % int(args.progress_every) == 0
                    and request_total != last_progress
                ):
                    print(f"retrieved_residual_true_shape_request {request_total}")
                    last_progress = int(request_total)
            output_nodes.append(output_node)

        target = {
            "format": "maskgen_generator_target_v1",
            "target_type": "parse_graph",
            "size": copy.deepcopy(topology_target.get("size", [256, 256])),
            "parse_graph": {
                "nodes": output_nodes,
                "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
                "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
            },
            "metadata": {
                "retrieved_residual_frame_true_shape": True,
                "source_topology_metadata": copy.deepcopy(topology_target.get("metadata", {})),
                "sample_index": int(sample_index),
                "semantic_valid": bool(validation["semantic_valid"]),
                "retrieved_stem": retrieved_entry.get("stem"),
                "retrieved_library_index": int(retrieved_entry.get("library_index", -1)),
                "retrieved_topology_path": retrieved_entry.get("topology_path"),
                "retrieval_score": float(retrieval_score),
                "mapping_diagnostics": mapping_diagnostics,
                "geometry_request_count": int(sample_request),
                "geometry_valid_count": int(sample_valid),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "attach_modes": dict(sample_attach_modes),
                "frame_clamp_modes": dict(sample_frame_clamp_modes),
                "true_shape_modes": dict(sample_shape_modes),
                "true_shape_quality_failure_count": int(sample_quality_failure),
                "true_shape_quality_reasons": dict(sample_quality_reasons),
                "geometry_rows": geometry_rows,
                "layout_residual_checkpoint": str(args.layout_residual_checkpoint.as_posix()),
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append(
            {
                "sample_index": int(sample_index),
                "output_path": str(output_path.as_posix()),
                "retrieval_score": float(retrieval_score),
                "geometry_request_count": int(sample_request),
                "geometry_valid_count": int(sample_valid),
                "attached_geometry_count": int(sample_attached),
                "missing_geometry_count": int(sample_missing),
                "attach_modes": dict(sample_attach_modes),
                "frame_clamp_modes": dict(sample_frame_clamp_modes),
                "true_shape_modes": dict(sample_shape_modes),
                "true_shape_quality_failure_count": int(sample_quality_failure),
            }
        )

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_retrieved_residual_layout_true_shape_topology_attach_summary_v1",
        "samples": str(args.samples.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "layout_residual_checkpoint": str(args.layout_residual_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(sample_rows)),
        "output_count": int(len(manifest_rows)),
        "skipped_invalid_count": int(skipped_invalid),
        "max_library_samples": args.max_library_samples,
        "exclude_same_stem": bool(args.exclude_same_stem),
        "library_summary": library_summary,
        "shape_fallback_summary": shape_summary,
        "geometry_request_count": int(request_total),
        "geometry_valid_count": int(valid_total),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "frame_clamp_modes": dict(frame_clamp_modes),
        "true_shape_modes": dict(shape_modes),
        "true_shape_quality_failure_count": int(quality_failure_total),
        "true_shape_quality_reasons": dict(quality_reasons),
        "min_true_shape_world_bbox_area": float(args.min_true_shape_world_bbox_area),
        "min_true_shape_local_bbox_side": float(args.min_true_shape_local_bbox_side),
        "final_scale_ratio_stats": _numeric_stats(final_scale_ratios),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
        "error_histogram": dict(error_histogram),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached retrieved-residual-layout true-shape samples={summary['output_count']} "
        f"requests={request_total} valid={valid_total} attached={attached_total} missing={missing_total} "
        f"library={library_summary['entry_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
