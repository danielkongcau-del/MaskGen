from __future__ import annotations

from collections import Counter
import copy
import json
import sys
from pathlib import Path
import argparse

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_geometry_conditioning import iter_jsonl, load_json, topology_node_index_by_id, _resolve_path  # noqa: E402
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig, sample_geometry_constrained  # noqa: E402
from partition_gen.manual_geometry_oracle_frame_conditioning import (  # noqa: E402
    extract_geometry_tokens_from_oracle_frame_conditioned,
    oracle_frame_conditioned_geometry_prefix_tokens,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target, validate_geometry_tokens  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach generated local geometry using oracle true frames from split targets.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {})
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


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
    manifest_path = args.split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    output_manifest: list[dict] = []
    attached_total = 0
    valid_total = 0
    request_total = 0
    missing_total = 0
    attach_modes: Counter[str] = Counter()
    for sample_index, row in enumerate(rows):
        topology_path = _resolve_path(row["topology_path"], split_root=args.split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
        node_index_by_id = topology_node_index_by_id(topology_target)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=args.split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        true_geometry_by_id = {str(target.get("source_node_id")): target for target in geometry_targets}
        output_nodes = []
        for node_index, node in enumerate(nodes):
            output_node = copy.deepcopy(node)
            geometry_ref = output_node.pop("geometry_ref", None)
            if geometry_ref and str(geometry_ref) in true_geometry_by_id:
                request_total += 1
                true_geometry = true_geometry_by_id[str(geometry_ref)]
                prefix = oracle_frame_conditioned_geometry_prefix_tokens(
                    topology_target,
                    true_geometry,
                    target_node_index=int(node_index_by_id[str(geometry_ref)]),
                    config=tokenizer_config,
                )
                sample = sample_geometry_constrained(
                    model,
                    vocab,
                    prefix_tokens=prefix,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                    constraint_config=constraint_config,
                    device=device,
                )
                conditioned_tokens = [str(token) for token in sample.get("tokens", []) or []]
                try:
                    geometry_tokens = extract_geometry_tokens_from_oracle_frame_conditioned(conditioned_tokens)
                    validation = validate_geometry_tokens(geometry_tokens, config=tokenizer_config)
                    if not bool(validation["valid"]):
                        raise ValueError(";".join(validation["errors"]))
                    generated = decode_geometry_tokens_to_target(
                        geometry_tokens,
                        config=tokenizer_config,
                        source_node_id=str(geometry_ref),
                    )
                    generated["frame"] = copy.deepcopy(true_geometry["frame"])
                    output_node["geometry_model"] = copy.deepcopy(generated.get("geometry_model", output_node.get("geometry_model")))
                    output_node["frame"] = copy.deepcopy(generated["frame"])
                    if "geometry" in generated:
                        output_node["geometry"] = copy.deepcopy(generated["geometry"])
                    if "atoms" in generated:
                        output_node["atoms"] = copy.deepcopy(generated["atoms"])
                    output_node["oracle_frame_geometry"] = True
                    valid_total += 1
                    attached_total += 1
                    attach_modes["oracle_frame_generated"] += 1
                except Exception as exc:
                    output_node["oracle_frame_geometry_error"] = f"{type(exc).__name__}: {exc}"
                    missing_total += 1
                    attach_modes["missing"] += 1
            output_nodes.append(output_node)
            if int(args.progress_every) > 0 and request_total > 0 and request_total % int(args.progress_every) == 0:
                print(f"oracle_frame_geometry_request {request_total}")
        target = {
            "format": "maskgen_generator_target_v1",
            "target_type": "parse_graph",
            "size": copy.deepcopy(topology_target.get("size", [256, 256])),
            "parse_graph": {
                "nodes": output_nodes,
                "relations": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("relations", []) or []),
                "residuals": copy.deepcopy((topology_target.get("parse_graph", {}) or {}).get("residuals", []) or []),
            },
            "metadata": {
                "oracle_frame_geometry": True,
                "sample_index": int(sample_index),
                "source_topology": str(topology_path.as_posix()),
            },
        }
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        output_manifest.append({"sample_index": sample_index, "output_path": str(output_path.as_posix())})
    write_jsonl(args.output_root / "manifest.jsonl", output_manifest)
    summary = {
        "format": "maskgen_oracle_frame_geometry_attach_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(rows)),
        "output_count": int(len(output_manifest)),
        "geometry_request_count": int(request_total),
        "geometry_valid_count": int(valid_total),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached oracle-frame geometry samples={summary['output_count']} requests={request_total} "
        f"valid={valid_total} attached={attached_total} missing={missing_total} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
