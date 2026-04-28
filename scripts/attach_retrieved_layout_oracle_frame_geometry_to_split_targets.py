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
        description="Attach generated local geometry to split topology using nearest-neighbor retrieved frames."
    )
    parser.add_argument("--split-root", type=Path, required=True, help="Query split root, usually val.")
    parser.add_argument("--library-split-root", type=Path, required=True, help="Retrieval library split root, usually train.")
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


def _generated_node_from_sample(
    *,
    model,
    vocab: dict[str, int],
    topology_target: dict,
    source_geometry: dict,
    retrieved_frame: dict,
    node_index: int,
    geometry_ref: str,
    tokenizer_config: ParseGraphTokenizerConfig,
    constraint_config: GeometryConstrainedSamplerConfig,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> dict:
    frame_conditioned_geometry = copy.deepcopy(source_geometry)
    frame_conditioned_geometry["frame"] = copy.deepcopy(retrieved_frame)
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
    if not bool(validation["valid"]):
        raise ValueError(";".join(validation["errors"]))
    generated = decode_geometry_tokens_to_target(
        geometry_tokens,
        config=tokenizer_config,
        source_node_id=str(geometry_ref),
    )
    generated["frame"] = copy.deepcopy(retrieved_frame)
    return generated


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

    manifest_path = args.split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    output_manifest: list[dict] = []
    request_total = 0
    valid_total = 0
    attached_total = 0
    missing_total = 0
    attach_modes: Counter[str] = Counter()
    mapping_modes: Counter[str] = Counter()
    error_histogram: Counter[str] = Counter()
    retrieval_scores: list[float] = []
    last_progress = 0

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
        retrieval_scores.append(float(retrieval_score))
        mapping_modes.update(mapping_diagnostics.get("mapping_mode_histogram", {}) or {})

        graph = topology_target.get("parse_graph", {}) or {}
        nodes = list(graph.get("nodes", []) or [])
        geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
        output_nodes: list[dict] = []
        sample_request = 0
        sample_attached = 0
        sample_missing = 0
        sample_valid = 0
        sample_attach_modes: Counter[str] = Counter()

        for node_index, node in enumerate(nodes):
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref:
                source_geometry = geometry_by_id.get(str(geometry_ref))
                retrieved_frame = frame_by_index.get(int(node_index))
                if source_geometry is None or retrieved_frame is None:
                    sample_missing += 1
                    missing_total += 1
                    attach_modes["missing"] += 1
                    sample_attach_modes["missing"] += 1
                    output_node["retrieved_frame_geometry_error"] = "missing source geometry or retrieved frame"
                else:
                    request_total += 1
                    sample_request += 1
                    try:
                        generated = _generated_node_from_sample(
                            model=model,
                            vocab=vocab,
                            topology_target=topology_target,
                            source_geometry=source_geometry,
                            retrieved_frame=retrieved_frame,
                            node_index=int(node_index),
                            geometry_ref=str(geometry_ref),
                            tokenizer_config=tokenizer_config,
                            constraint_config=constraint_config,
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                            device=device,
                        )
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
                        output_node["retrieved_frame_geometry_error"] = f"{error_key}: {exc}"
                if (
                    int(args.progress_every) > 0
                    and request_total > 0
                    and request_total % int(args.progress_every) == 0
                    and request_total != last_progress
                ):
                    print(f"retrieved_frame_geometry_request {request_total}")
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
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        output_manifest.append({"sample_index": int(sample_index), "output_path": str(output_path.as_posix())})

    write_jsonl(args.output_root / "manifest.jsonl", output_manifest)
    summary = {
        "format": "maskgen_retrieved_layout_oracle_frame_geometry_attach_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(rows)),
        "output_count": int(len(output_manifest)),
        "max_library_samples": args.max_library_samples,
        "exclude_same_stem": bool(args.exclude_same_stem),
        "library_summary": library_summary,
        "geometry_request_count": int(request_total),
        "geometry_valid_count": int(valid_total),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "mapping_mode_histogram": dict(mapping_modes),
        "retrieval_score_stats": _numeric_stats(retrieval_scores),
        "error_histogram": dict(error_histogram),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached retrieved-layout generated-geometry samples={summary['output_count']} "
        f"requests={request_total} valid={valid_total} attached={attached_total} missing={missing_total} "
        f"library={library_summary['entry_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
