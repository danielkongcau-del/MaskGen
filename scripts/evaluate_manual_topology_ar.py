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
from partition_gen.manual_topology_constrained_sampling import TopologyConstrainedSamplerConfig  # noqa: E402
from partition_gen.manual_topology_evaluation import (  # noqa: E402
    evaluate_topology_sample_rows,
    sample_model_topology_rows,
    write_topology_sample_evaluation_markdown,
    write_topology_sample_rows,
)
from partition_gen.parse_graph_tokenizer import load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and evaluate a manual topology AR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-label", type=int, default=6)
    parser.add_argument("--max-children-per-group", type=int, default=128)
    parser.add_argument("--max-relation-pairs", type=int, default=512)
    parser.add_argument("--allow-other-relations", action="store_true")
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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    checkpoint, model, _ = load_checkpoint(args.checkpoint, map_location="cpu", load_optimizer=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    constraint_config = None
    if bool(args.constrained):
        constraint_config = TopologyConstrainedSamplerConfig(
            max_nodes=int(args.max_nodes),
            max_label=int(args.max_label),
            max_children_per_group=int(args.max_children_per_group),
            max_relation_pairs=int(args.max_relation_pairs),
            allow_other_relations=bool(args.allow_other_relations),
            max_other_relations=int(args.max_relation_pairs) if bool(args.allow_other_relations) else 0,
        )
    rows = sample_model_topology_rows(
        model,
        vocab,
        num_samples=int(args.num_samples),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if int(args.top_k) > 0 else None,
        device=device,
        constraint_config=constraint_config,
        progress_every=int(args.progress_every),
        progress_label="topology_eval",
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
        }
    summary = evaluate_topology_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    summary.update(
        {
            "checkpoint": str(args.checkpoint.as_posix()),
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "sampling_mode": "constrained" if bool(args.constrained) else "unconstrained",
            "sampling_config": {
                "num_samples": int(args.num_samples),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "seed": int(args.seed),
                "constrained": bool(args.constrained),
                "constraint_config": None if constraint_config is None else constraint_config.__dict__,
            },
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_topology_sample_evaluation_markdown(summary, args.summary_md)
    if args.output_samples is not None:
        write_topology_sample_rows(args.output_samples, rows)
    print(
        f"evaluated samples={summary['sample_count']} valid={summary['valid_count']} "
        f"valid_rate={summary['valid_rate']:.4f} hit_eos={summary['hit_eos_count']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
