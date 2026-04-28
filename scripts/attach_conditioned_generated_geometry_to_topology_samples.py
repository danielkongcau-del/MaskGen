from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_geometry_conditioning import (  # noqa: E402
    conditioned_geometry_prefix_tokens,
    extract_geometry_tokens_from_conditioned,
)
from partition_gen.manual_geometry_constrained_sampling import (  # noqa: E402
    GeometryConstrainedSamplerConfig,
    sample_geometry_constrained,
)
from partition_gen.manual_geometry_sample_validation import decode_geometry_tokens_to_target, validate_geometry_tokens  # noqa: E402
from partition_gen.manual_topology_generated_geometry import (  # noqa: E402
    build_conditioned_generated_geometry_targets_from_sample_rows,
)
from partition_gen.manual_topology_placeholder_geometry import iter_jsonl  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach topology-conditioned generated geometry to topology samples.")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL topology samples containing token rows.")
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-polygons", type=int, default=8)
    parser.add_argument("--max-points-per-ring", type=int, default=128)
    parser.add_argument("--max-holes-per-polygon", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


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


def _make_conditioned_geometry_sampler(args: argparse.Namespace):
    checkpoint, model, _optimizer = load_checkpoint(args.geometry_checkpoint, map_location="cpu", load_optimizer=False)
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    constraint_config = GeometryConstrainedSamplerConfig(
        max_polygons=int(args.max_polygons),
        max_points_per_ring=int(args.max_points_per_ring),
        max_holes_per_polygon=int(args.max_holes_per_polygon),
    )
    tokenizer_config = ParseGraphTokenizerConfig()
    request_count = 0

    def geometry_sampler(topology_target: dict, node: dict, node_index: int) -> tuple[dict | None, dict]:
        nonlocal request_count
        request_count += 1
        prefix = conditioned_geometry_prefix_tokens(
            topology_target,
            target_node_index=int(node_index),
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
            tokens = extract_geometry_tokens_from_conditioned(conditioned_tokens)
        except Exception as exc:
            return None, {
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "length": int(sample.get("length", len(conditioned_tokens))),
                "conditioned_length": int(len(conditioned_tokens)),
                "hit_eos": bool(sample.get("hit_eos", False)),
                "stopped_reason": sample.get("stopped_reason"),
                "prefix_length": int(len(prefix)),
            }
        validation = validate_geometry_tokens(tokens)
        diagnostics = {
            "valid": bool(validation["valid"]),
            "errors": list(validation["errors"]),
            "length": int(len(tokens)),
            "conditioned_length": int(sample.get("length", len(conditioned_tokens))),
            "hit_eos": bool(sample.get("hit_eos", False)),
            "stopped_reason": sample.get("stopped_reason"),
            "prefix_length": int(len(prefix)),
        }
        if int(args.progress_every) > 0 and request_count % int(args.progress_every) == 0:
            print(f"conditioned_generated_geometry_request {request_count}")
        if not bool(validation["valid"]):
            return None, diagnostics
        try:
            geometry_target = decode_geometry_tokens_to_target(
                tokens,
                source_node_id=str(node.get("id", f"node_{node_index}")),
            )
        except Exception as exc:
            diagnostics["valid"] = False
            diagnostics["errors"].append(f"{type(exc).__name__}: {exc}")
            return None, diagnostics
        geometry_target["metadata"].update(
            {
                "generated_by_checkpoint": str(args.geometry_checkpoint.as_posix()),
                "node_index": int(node_index),
                "conditioned_prefix_length": int(len(prefix)),
            }
        )
        return geometry_target, diagnostics

    return checkpoint, vocab, constraint_config, geometry_sampler


def main() -> None:
    args = parse_args()
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]

    checkpoint, vocab, constraint_config, geometry_sampler = _make_conditioned_geometry_sampler(args)
    targets, summary = build_conditioned_generated_geometry_targets_from_sample_rows(
        sample_rows,
        geometry_sampler,
        include_invalid=bool(args.include_invalid),
    )

    manifest_rows = []
    for index, target in enumerate(targets):
        sample_index = int(target.get("metadata", {}).get("sample_index", index))
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append(
            {
                "format": "maskgen_conditioned_generated_geometry_manifest_row_v1",
                "sample_index": sample_index,
                "output_path": str(output_path.as_posix()),
                "node_count": int(len(target.get("parse_graph", {}).get("nodes", []) or [])),
                "relation_count": int(len(target.get("parse_graph", {}).get("relations", []) or [])),
                "geometry_request_count": int(target.get("metadata", {}).get("geometry_request_count", 0)),
                "geometry_valid_count": int(target.get("metadata", {}).get("geometry_valid_count", 0)),
                "attached_geometry_count": int(target.get("metadata", {}).get("attached_geometry_count", 0)),
                "missing_geometry_count": int(target.get("metadata", {}).get("missing_geometry_count", 0)),
                "attach_modes": target.get("metadata", {}).get("attach_modes", {}),
            }
        )
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary.update(
        {
            "samples": str(args.samples.as_posix()),
            "geometry_checkpoint": str(args.geometry_checkpoint.as_posix()),
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "output_root": str(args.output_root.as_posix()),
            "vocab_size": int(len(vocab)),
            "sampling_config": {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "constraint_config": constraint_config.__dict__,
            },
        }
    )
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached conditioned geometry samples={summary['output_count']} "
        f"requests={summary['geometry_request_count']} "
        f"valid={summary['geometry_valid_count']} "
        f"attached={summary['attached_geometry_count']} "
        f"missing={summary['missing_geometry_count']} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
