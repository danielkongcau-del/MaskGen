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
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig, sample_geometry_constrained  # noqa: E402
from partition_gen.manual_geometry_oracle_frame_conditioning import (  # noqa: E402
    extract_geometry_tokens_from_oracle_frame_conditioned,
    oracle_frame_conditioned_geometry_prefix_tokens,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target, validate_geometry_tokens  # noqa: E402
from partition_gen.manual_layout_retrieval import (  # noqa: E402
    build_layout_retrieval_fallbacks,
    build_layout_retrieval_library,
    geometry_condition_target_from_topology_node,
    map_retrieved_layout_frames,
    retrieve_layout_entry,
    write_jsonl,
)
from partition_gen.manual_topology_placeholder_geometry import decode_topology_tokens_to_target, iter_jsonl  # noqa: E402
from partition_gen.manual_topology_sample_validation import validate_topology_tokens  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach retrieved-layout frames and generated local geometry to generated topology samples."
    )
    parser.add_argument("--samples", type=Path, required=True, help="JSONL topology samples containing token rows.")
    parser.add_argument("--library-split-root", type=Path, required=True, help="Retrieval library split root, usually train.")
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--exclude-same-stem", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-polygons", type=int, default=8)
    parser.add_argument("--max-points-per-ring", type=int, default=128)
    parser.add_argument("--max-holes-per-polygon", type=int, default=8)
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


def _sample_local_geometry(
    *,
    model,
    vocab: dict[str, int],
    topology_target: dict,
    node: dict,
    node_index: int,
    geometry_ref: str,
    retrieved_frame: dict,
    tokenizer_config: ParseGraphTokenizerConfig,
    constraint_config: GeometryConstrainedSamplerConfig,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> tuple[dict, dict]:
    condition_geometry = geometry_condition_target_from_topology_node(
        node,
        frame=retrieved_frame,
        source_node_id=str(geometry_ref),
    )
    prefix = oracle_frame_conditioned_geometry_prefix_tokens(
        topology_target,
        condition_geometry,
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
    generated["frame"] = copy.deepcopy(retrieved_frame)
    return generated, diagnostics


def main() -> None:
    args = parse_args()
    checkpoint, model, _optimizer = load_checkpoint(args.geometry_checkpoint, map_location="cpu", load_optimizer=False)
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    tokenizer_config = ParseGraphTokenizerConfig()
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

    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]

    manifest_rows: list[dict] = []
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    skipped_invalid = 0
    request_total = 0
    valid_total = 0
    attached_total = 0
    missing_total = 0
    hit_eos_total = 0
    last_progress = 0

    for fallback_index, row in enumerate(sample_rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not bool(args.include_invalid):
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target["metadata"].update(
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
        frame_by_index, mapping_diagnostics = map_retrieved_layout_frames(
            topology_target,
            retrieved_entry,
            fallback_frames=fallback_frames,
        )
        retrieval_scores.append(float(retrieval_score))
        mapping_modes.update(mapping_diagnostics.get("mapping_mode_histogram", {}) or {})

        graph = topology_target.get("parse_graph", {}) or {}
        output_nodes: list[dict] = []
        sample_request = 0
        sample_valid = 0
        sample_attached = 0
        sample_missing = 0
        sample_attach_modes: Counter[str] = Counter()
        geometry_rows: list[dict] = []

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
                    output_node["retrieved_frame_geometry_error"] = "missing retrieved frame"
                else:
                    request_total += 1
                    sample_request += 1
                    try:
                        generated, diagnostics = _sample_local_geometry(
                            model=model,
                            vocab=vocab,
                            topology_target=topology_target,
                            node=output_node,
                            node_index=int(node_index),
                            geometry_ref=str(geometry_ref),
                            retrieved_frame=retrieved_frame,
                            tokenizer_config=tokenizer_config,
                            constraint_config=constraint_config,
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                            device=device,
                        )
                        geometry_row.update(diagnostics)
                        hit_eos_total += int(bool(diagnostics.get("hit_eos", False)))
                        output_node["geometry_model"] = copy.deepcopy(
                            generated.get("geometry_model", output_node.get("geometry_model"))
                        )
                        output_node["frame"] = copy.deepcopy(generated["frame"])
                        if "geometry" in generated:
                            output_node["geometry"] = copy.deepcopy(generated["geometry"])
                        if "atoms" in generated:
                            output_node["atoms"] = copy.deepcopy(generated["atoms"])
                        output_node["layout_frame_source"] = "retrieved_layout"
                        output_node["layout_shape_attach_mode"] = "generated_shape"
                        output_node["retrieved_frame_geometry"] = True
                        sample_valid += 1
                        valid_total += 1
                        sample_attached += 1
                        attached_total += 1
                        attach_modes["retrieved_frame_generated"] += 1
                        sample_attach_modes["retrieved_frame_generated"] += 1
                    except Exception as exc:
                        sample_missing += 1
                        missing_total += 1
                        attach_modes["missing"] += 1
                        sample_attach_modes["missing"] += 1
                        error_key = type(exc).__name__
                        error_histogram[error_key] += 1
                        geometry_row.update({"valid": False, "errors": [f"{error_key}: {exc}"]})
                        output_node["retrieved_frame_geometry_error"] = f"{error_key}: {exc}"
                geometry_rows.append(geometry_row)
                if (
                    int(args.progress_every) > 0
                    and request_total > 0
                    and request_total % int(args.progress_every) == 0
                    and request_total != last_progress
                ):
                    print(f"retrieved_frame_topology_geometry_request {request_total}")
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
                "retrieved_frame_generated_geometry": True,
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
                "geometry_rows": geometry_rows,
                "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
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
            }
        )

    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_retrieved_layout_oracle_frame_geometry_topology_attach_summary_v1",
        "samples": str(args.samples.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(sample_rows)),
        "output_count": int(len(manifest_rows)),
        "skipped_invalid_count": int(skipped_invalid),
        "max_library_samples": args.max_library_samples,
        "exclude_same_stem": bool(args.exclude_same_stem),
        "library_summary": library_summary,
        "geometry_request_count": int(request_total),
        "geometry_valid_count": int(valid_total),
        "geometry_hit_eos_count": int(hit_eos_total),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
        "error_histogram": dict(error_histogram),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached retrieved-layout generated topology geometry samples={summary['output_count']} "
        f"requests={request_total} valid={valid_total} attached={attached_total} missing={missing_total} "
        f"library={library_summary['entry_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
