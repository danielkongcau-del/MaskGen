from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig  # noqa: E402
from partition_gen.manual_geometry_evaluation import (  # noqa: E402
    evaluate_geometry_sample_rows,
    sample_model_geometry_rows,
    write_geometry_sample_evaluation_markdown,
    write_geometry_sample_rows,
)
from partition_gen.manual_split_token_dataset import ManualSplitTokenSequenceDataset  # noqa: E402
from partition_gen.parse_graph_tokenizer import load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and evaluate a manual geometry AR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--token-root", type=Path, default=None, help="Optional token root used to source forced geometry prefixes.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--constrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-role", type=str, default="support_region")
    parser.add_argument("--prefix-label", type=int, default=0)
    parser.add_argument("--prefix-geometry-model", type=str, default="polygon_code")
    parser.add_argument("--max-polygons", type=int, default=8)
    parser.add_argument("--max-points-per-ring", type=int, default=128)
    parser.add_argument("--max-holes-per-polygon", type=int, default=8)
    parser.add_argument("--top-k-invalid", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {})
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


def _resolve_token_root(args: argparse.Namespace, checkpoint: dict) -> Path | None:
    if args.token_root is not None:
        return args.token_root
    train_config = checkpoint.get("train_config", {})
    value = train_config.get("val_token_root") or train_config.get("train_token_root")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    checkpoint, model, _ = load_checkpoint(args.checkpoint, map_location="cpu", load_optimizer=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    token_root = _resolve_token_root(args, checkpoint)
    source_rows = None
    if token_root is not None:
        source_rows = ManualSplitTokenSequenceDataset(token_root, sequence_kind="geometry").rows
    constraint_config = None
    if bool(args.constrained):
        constraint_config = GeometryConstrainedSamplerConfig(
            max_polygons=int(args.max_polygons),
            max_points_per_ring=int(args.max_points_per_ring),
            max_holes_per_polygon=int(args.max_holes_per_polygon),
        )
    rows = sample_model_geometry_rows(
        model,
        vocab,
        num_samples=int(args.num_samples),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if int(args.top_k) > 0 else None,
        device=device,
        source_rows=source_rows,
        prefix_role=str(args.prefix_role),
        prefix_label=int(args.prefix_label),
        prefix_geometry_model=str(args.prefix_geometry_model),
        constraint_config=constraint_config,
        progress_every=int(args.progress_every),
        progress_label="geometry_eval",
    )
    for row in rows:
        row["checkpoint"] = str(args.checkpoint.as_posix())
        row["sampling_config"] = {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "constrained": bool(args.constrained),
            "constraint_config": None if constraint_config is None else constraint_config.__dict__,
            "token_root": None if token_root is None else str(token_root.as_posix()),
        }
    summary = evaluate_geometry_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    summary.update(
        {
            "checkpoint": str(args.checkpoint.as_posix()),
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "sampling_mode": "constrained" if bool(args.constrained) else "unconstrained",
            "sampling_config": rows[0]["sampling_config"] if rows else {},
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_geometry_sample_evaluation_markdown(summary, args.summary_md)
    if args.output_samples is not None:
        write_geometry_sample_rows(args.output_samples, rows)
    print(
        f"evaluated geometry samples={summary['sample_count']} valid={summary['valid_count']} "
        f"valid_rate={summary['valid_rate']:.4f} hit_eos={summary['hit_eos_count']} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
