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

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_geometry_conditioning import iter_jsonl  # noqa: E402
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig, sample_geometry_constrained  # noqa: E402
from partition_gen.manual_geometry_oracle_frame_conditioning import (  # noqa: E402
    extract_geometry_tokens_from_oracle_frame_conditioned,
    oracle_frame_conditioned_geometry_prefix_tokens,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target, validate_geometry_tokens  # noqa: E402
from partition_gen.manual_geometry_shape_fallback import (  # noqa: E402
    build_geometry_shape_fallback_library,
    geometry_target_from_fallback_shape,
    local_bbox_quality,
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
    load_split_row,
    map_retrieved_layout_frames,
    retrieve_layout_entry,
    write_jsonl,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach generated local geometry to split topology using residual-corrected retrieved frames."
    )
    parser.add_argument("--split-root", type=Path, required=True, help="Query split root, usually val.")
    parser.add_argument("--library-split-root", type=Path, required=True, help="Retrieval library split root, usually train.")
    parser.add_argument("--layout-residual-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--exclude-same-stem", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-polygons", type=int, default=8)
    parser.add_argument("--max-points-per-ring", type=int, default=128)
    parser.add_argument("--max-holes-per-polygon", type=int, default=8)
    parser.add_argument("--geometry-retry-count", type=int, default=2)
    parser.add_argument("--min-generated-world-bbox-area", type=float, default=1.0)
    parser.add_argument("--min-generated-local-bbox-side", type=float, default=1e-6)
    parser.add_argument("--disable-true-shape-fallback", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {}) or {}
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


def _geometry_targets_by_source_node_id(geometry_targets: list[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


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


def _generated_node_from_sample(
    *,
    model,
    vocab: dict[str, int],
    topology_target: dict,
    source_geometry: dict,
    conditioning_frame: dict,
    node_index: int,
    geometry_ref: str,
    tokenizer_config: ParseGraphTokenizerConfig,
    constraint_config: GeometryConstrainedSamplerConfig,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> tuple[dict, dict]:
    frame_conditioned_geometry = copy.deepcopy(source_geometry)
    frame_conditioned_geometry["frame"] = copy.deepcopy(conditioning_frame)
    prefix = oracle_frame_conditioned_geometry_prefix_tokens(
        topology_target,
        frame_conditioned_geometry,
        target_node_index=int(node_index),
        config=tokenizer_config,
    )
    sample = sample_geometry_constrained(
        model,
        vocab,
        prefix_tokens=prefix,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=top_k,
        constraint_config=constraint_config,
        device=device,
    )
    conditioned_tokens = [str(token) for token in sample.get("tokens", []) or []]
    geometry_tokens = extract_geometry_tokens_from_oracle_frame_conditioned(conditioned_tokens)
    validation = validate_geometry_tokens(geometry_tokens, config=tokenizer_config)
    diagnostics = {
        "valid": bool(validation["valid"]),
        "errors": list(validation["errors"]),
        "length": int(len(geometry_tokens)),
        "conditioned_length": int(sample.get("length", len(conditioned_tokens))),
        "hit_eos": bool(sample.get("hit_eos", False)),
        "stopped_reason": sample.get("stopped_reason"),
        "prefix_length": int(len(prefix)),
    }
    if not bool(validation["valid"]):
        raise ValueError(";".join(validation["errors"]))
    generated = decode_geometry_tokens_to_target(
        geometry_tokens,
        config=tokenizer_config,
        source_node_id=str(geometry_ref),
    )
    generated["frame"] = copy.deepcopy(conditioning_frame)
    return generated, diagnostics


def main() -> None:
    args = parse_args()
    geometry_checkpoint, geometry_model, _optimizer = load_checkpoint(
        args.geometry_checkpoint,
        map_location="cpu",
        load_optimizer=False,
    )
    _layout_checkpoint, layout_model, tokenizer_config = load_layout_residual_checkpoint(
        args.layout_residual_checkpoint,
        map_location="cpu",
    )
    vocab = load_vocabulary(_resolve_vocab_path(geometry_checkpoint))
    device = torch.device(args.device)
    geometry_model = geometry_model.to(device)
    geometry_model.eval()
    layout_model = layout_model.to(device)
    layout_model.eval()

    constraint_config = GeometryConstrainedSamplerConfig(
        max_polygons=int(args.max_polygons),
        max_points_per_ring=int(args.max_points_per_ring),
        max_holes_per_polygon=int(args.max_holes_per_polygon),
    )
    library_entries, library_summary = build_layout_retrieval_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
    )
    fallback_frames = build_layout_retrieval_fallbacks(library_entries)
    shape_fallback_library, shape_fallback_summary = build_geometry_shape_fallback_library(
        args.library_split_root,
        max_samples=args.max_library_samples,
        min_local_bbox_side=float(args.min_generated_local_bbox_side),
    )

    manifest_path = args.split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    output_manifest: list[dict] = []
    request_total = 0
    valid_total = 0
    attached_total = 0
    missing_total = 0
    hit_eos_total = 0
    sample_request_total = 0
    sample_valid_total = 0
    geometry_retry_total = 0
    geometry_retry_success_total = 0
    geometry_quality_reject_total = 0
    geometry_fallback_total = 0
    final_quality_failure_total = 0
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    frame_clamp_modes: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    final_scale_ratios: list[float] = []
    last_progress = 0
    quality_reasons: Counter[str] = Counter()

    for sample_index, row in enumerate(rows):
        topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=args.split_root,
            manifest_parent=manifest_path.parent,
        )
        retrieved_entry, retrieval_score = retrieve_layout_entry(
            topology_target,
            library_entries,
            exclude_stem=row.get("stem") if bool(args.exclude_same_stem) else None,
        )
        frame_by_index, mapping_diagnostics = map_retrieved_layout_frames(
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
        nodes = list(graph.get("nodes", []) or [])
        geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
        output_nodes: list[dict] = []
        geometry_rows: list[dict] = []
        sample_request = 0
        sample_attached = 0
        sample_missing = 0
        sample_valid = 0
        sample_attach_modes: Counter[str] = Counter()
        sample_frame_clamp_modes: Counter[str] = Counter()
        sample_quality_reasons: Counter[str] = Counter()
        sample_retry_count = 0
        sample_retry_success_count = 0
        sample_quality_reject_count = 0
        sample_fallback_count = 0
        sample_final_quality_failure_count = 0
        sample_sample_request_count = 0
        sample_sample_valid_count = 0

        for node_index, node in enumerate(nodes):
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref:
                source_geometry = geometry_by_id.get(str(geometry_ref))
                retrieved_frame = frame_by_index.get(int(node_index))
                geometry_row = {
                    "node_index": int(node_index),
                    "node_id": str(output_node.get("id", "")),
                    "role": str(output_node.get("role", "")),
                    "label": int(output_node.get("label", 0)),
                    "geometry_model": str(output_node.get("geometry_model", "none")),
                }
                if source_geometry is None or retrieved_frame is None:
                    sample_missing += 1
                    missing_total += 1
                    attach_modes["missing"] += 1
                    sample_attach_modes["missing"] += 1
                    geometry_row.update({"valid": False, "errors": ["missing source geometry or retrieved frame"]})
                    output_node["retrieved_residual_frame_geometry_error"] = "missing source geometry or retrieved frame"
                else:
                    request_total += 1
                    sample_request += 1
                    try:
                        residual_example = build_layout_residual_example(
                            topology_target,
                            node_index=int(node_index),
                            retrieved_frame=retrieved_frame,
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
                        retry_count = max(0, int(args.geometry_retry_count))
                        generated = None
                        diagnostics: dict = {}
                        generated_local_bbox: dict = {}
                        final_frame: dict = {}
                        clamp_diagnostics: dict = {}
                        quality: dict = {}
                        attempt_rows: list[dict] = []
                        used_fallback = False
                        fallback_mode = None
                        fallback_shape_source = None
                        for attempt_index in range(retry_count + 1):
                            sample_request_total += 1
                            sample_sample_request_count += 1
                            generated, diagnostics = _generated_node_from_sample(
                                model=geometry_model,
                                vocab=vocab,
                                topology_target=topology_target,
                                source_geometry=source_geometry,
                                conditioning_frame=refined_frame,
                                node_index=int(node_index),
                                geometry_ref=str(geometry_ref),
                                tokenizer_config=tokenizer_config,
                                constraint_config=constraint_config,
                                max_new_tokens=int(args.max_new_tokens),
                                temperature=float(args.temperature),
                                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                                device=device,
                            )
                            sample_valid_total += 1
                            sample_sample_valid_count += 1
                            generated_local_bbox = geometry_local_bbox(generated)
                            final_frame, clamp_diagnostics = clamp_frame_to_local_bbox(
                                refined_frame,
                                generated_local_bbox,
                                config=tokenizer_config,
                            )
                            quality = local_bbox_quality(
                                generated_local_bbox,
                                final_frame,
                                canvas_size=topology_target.get("size", [256, 256]),
                                min_world_bbox_area=float(args.min_generated_world_bbox_area),
                                min_local_bbox_side=float(args.min_generated_local_bbox_side),
                            )
                            attempt_rows.append(
                                {
                                    "attempt_index": int(attempt_index),
                                    "quality": quality,
                                    "diagnostics": diagnostics,
                                    "frame_clamp": clamp_diagnostics,
                                }
                            )
                            if bool(quality["usable"]):
                                if attempt_index > 0:
                                    geometry_retry_success_total += 1
                                    sample_retry_success_count += 1
                                break
                            geometry_quality_reject_total += 1
                            sample_quality_reject_count += 1
                            quality_reasons.update(quality.get("reasons", []) or [])
                            sample_quality_reasons.update(quality.get("reasons", []) or [])
                            if attempt_index < retry_count:
                                geometry_retry_total += 1
                                sample_retry_count += 1
                        if not bool(quality.get("usable", False)) and not bool(args.disable_true_shape_fallback):
                            fallback_shape, fallback_mode = select_fallback_geometry_shape(output_node, shape_fallback_library)
                            if fallback_shape is not None:
                                generated = geometry_target_from_fallback_shape(
                                    fallback_shape,
                                    source_node_id=str(geometry_ref),
                                    frame=refined_frame,
                                )
                                generated_local_bbox = geometry_local_bbox(generated)
                                final_frame, clamp_diagnostics = clamp_frame_to_local_bbox(
                                    refined_frame,
                                    generated_local_bbox,
                                    config=tokenizer_config,
                                )
                                quality = local_bbox_quality(
                                    generated_local_bbox,
                                    final_frame,
                                    canvas_size=topology_target.get("size", [256, 256]),
                                    min_world_bbox_area=float(args.min_generated_world_bbox_area),
                                    min_local_bbox_side=float(args.min_generated_local_bbox_side),
                                )
                                used_fallback = True
                                fallback_shape_source = fallback_shape.get("source_stem")
                                geometry_fallback_total += 1
                                sample_fallback_count += 1
                        if generated is None:
                            raise RuntimeError("geometry sampling produced no candidate")
                        if not bool(quality.get("usable", False)):
                            final_quality_failure_total += 1
                            sample_final_quality_failure_count += 1
                            quality_reasons.update(quality.get("reasons", []) or [])
                            sample_quality_reasons.update(quality.get("reasons", []) or [])
                        frame_clamp_mode = _frame_clamp_mode(clamp_diagnostics)
                        frame_clamp_modes[frame_clamp_mode] += 1
                        sample_frame_clamp_modes[frame_clamp_mode] += 1
                        final_scale_ratios.append(float(clamp_diagnostics["scale_ratio"]))
                        generated["frame"] = copy.deepcopy(final_frame)
                        attach_mode = (
                            "retrieved_residual_frame_fallback_true_shape"
                            if used_fallback
                            else "retrieved_residual_frame_generated"
                        )
                        shape_attach_mode = "fallback_true_shape" if used_fallback else "generated_shape"
                        geometry_row.update(
                            {
                                **diagnostics,
                                "retrieved_frame": copy.deepcopy(retrieved_frame),
                                "refined_frame": copy.deepcopy(refined_frame),
                                "final_frame": copy.deepcopy(final_frame),
                                "layout_residual": residual,
                                "generated_local_bbox": generated_local_bbox,
                                "frame_clamp": clamp_diagnostics,
                                "frame_clamp_mode": frame_clamp_mode,
                                "local_bbox_quality": quality,
                                "geometry_attempts": attempt_rows,
                                "geometry_attempt_count": int(len(attempt_rows)),
                                "used_fallback_true_shape": bool(used_fallback),
                                "fallback_mode": fallback_mode,
                                "fallback_shape_source": fallback_shape_source,
                            }
                        )
                        hit_eos_total += int(bool(diagnostics.get("hit_eos", False)))
                        output_node["geometry_model"] = copy.deepcopy(
                            generated.get("geometry_model", output_node.get("geometry_model"))
                        )
                        output_node["frame"] = copy.deepcopy(final_frame)
                        output_node["retrieved_frame"] = copy.deepcopy(retrieved_frame)
                        output_node["refined_frame"] = copy.deepcopy(refined_frame)
                        output_node["layout_residual"] = residual
                        output_node["generated_local_bbox"] = generated_local_bbox
                        output_node["frame_clamp"] = clamp_diagnostics
                        output_node["local_bbox_quality"] = quality
                        output_node["geometry_attempt_count"] = int(len(attempt_rows))
                        output_node["geometry_fallback_mode"] = fallback_mode
                        if "geometry" in generated:
                            output_node["geometry"] = copy.deepcopy(generated["geometry"])
                        if "atoms" in generated:
                            output_node["atoms"] = copy.deepcopy(generated["atoms"])
                        output_node["layout_frame_source"] = "retrieved_residual_layout"
                        output_node["layout_shape_attach_mode"] = shape_attach_mode
                        output_node["retrieved_residual_frame_geometry"] = True
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
                        output_node["retrieved_residual_frame_geometry_error"] = f"{error_key}: {exc}"
                geometry_rows.append(geometry_row)
                if (
                    int(args.progress_every) > 0
                    and request_total > 0
                    and request_total % int(args.progress_every) == 0
                    and request_total != last_progress
                ):
                    print(f"retrieved_residual_frame_geometry_request {request_total}")
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
                "retrieved_residual_frame_generated_geometry": True,
                "sample_index": int(sample_index),
                "source_topology": str(topology_path.as_posix()),
                "query_stem": row.get("stem"),
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
                "geometry_sample_request_count": int(sample_sample_request_count),
                "geometry_sample_valid_count": int(sample_sample_valid_count),
                "geometry_retry_count": int(sample_retry_count),
                "geometry_retry_success_count": int(sample_retry_success_count),
                "geometry_quality_reject_count": int(sample_quality_reject_count),
                "geometry_fallback_true_shape_count": int(sample_fallback_count),
                "geometry_final_quality_failure_count": int(sample_final_quality_failure_count),
                "geometry_quality_reasons": dict(sample_quality_reasons),
                "geometry_rows": geometry_rows,
                "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
                "layout_residual_checkpoint": str(args.layout_residual_checkpoint.as_posix()),
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        output_manifest.append(
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
                "geometry_sample_request_count": int(sample_sample_request_count),
                "geometry_retry_count": int(sample_retry_count),
                "geometry_fallback_true_shape_count": int(sample_fallback_count),
                "geometry_final_quality_failure_count": int(sample_final_quality_failure_count),
            }
        )

    write_jsonl(args.output_root / "manifest.jsonl", output_manifest)
    summary = {
        "format": "maskgen_retrieved_residual_layout_oracle_frame_geometry_attach_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
        "layout_residual_checkpoint": str(args.layout_residual_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(rows)),
        "output_count": int(len(output_manifest)),
        "max_library_samples": args.max_library_samples,
        "exclude_same_stem": bool(args.exclude_same_stem),
        "library_summary": library_summary,
        "shape_fallback_summary": shape_fallback_summary,
        "geometry_request_count": int(request_total),
        "geometry_valid_count": int(valid_total),
        "geometry_hit_eos_count": int(hit_eos_total),
        "geometry_sample_request_count": int(sample_request_total),
        "geometry_sample_valid_count": int(sample_valid_total),
        "geometry_retry_count": int(geometry_retry_total),
        "geometry_retry_success_count": int(geometry_retry_success_total),
        "geometry_quality_reject_count": int(geometry_quality_reject_total),
        "geometry_fallback_true_shape_count": int(geometry_fallback_total),
        "geometry_final_quality_failure_count": int(final_quality_failure_total),
        "geometry_quality_reasons": dict(quality_reasons),
        "min_generated_world_bbox_area": float(args.min_generated_world_bbox_area),
        "min_generated_local_bbox_side": float(args.min_generated_local_bbox_side),
        "disable_true_shape_fallback": bool(args.disable_true_shape_fallback),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "frame_clamp_modes": dict(frame_clamp_modes),
        "final_scale_ratio_stats": _numeric_stats(final_scale_ratios),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
        "error_histogram": dict(error_histogram),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached retrieved-residual-layout generated-geometry samples={summary['output_count']} "
        f"requests={request_total} valid={valid_total} attached={attached_total} missing={missing_total} "
        f"library={library_summary['entry_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
