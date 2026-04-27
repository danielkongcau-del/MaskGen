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
from partition_gen.manual_topology_constrained_sampling import (  # noqa: E402
    TopologyConstrainedSamplerConfig,
    sample_topology_constrained,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens  # noqa: E402
from partition_gen.parse_graph_tokenizer import load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grammar-constrained sampling for manual topology AR checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-label", type=int, default=6)
    parser.add_argument("--max-children-per-group", type=int, default=128)
    parser.add_argument("--max-relation-pairs", type=int, default=512)
    parser.add_argument("--allow-other-relations", action="store_true")
    parser.add_argument("--validate", action="store_true")
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
    constraint_config = TopologyConstrainedSamplerConfig(
        max_nodes=int(args.max_nodes),
        max_label=int(args.max_label),
        max_children_per_group=int(args.max_children_per_group),
        max_relation_pairs=int(args.max_relation_pairs),
        allow_other_relations=bool(args.allow_other_relations),
        max_other_relations=int(args.max_relation_pairs) if bool(args.allow_other_relations) else 0,
    )

    output = args.output or (args.checkpoint.parent / "samples_constrained.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    valid_count = 0
    with torch.no_grad():
        for index in range(int(args.num_samples)):
            sample = sample_topology_constrained(
                model,
                vocab,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                constraint_config=constraint_config,
                device=device,
            )
            validation = validate_topology_tokens(sample["tokens"]) if bool(args.validate) else None
            if validation and validation["valid"]:
                valid_count += 1
            row = {
                "format": "maskgen_manual_topology_ar_sample_v1",
                "sample_index": int(index),
                "checkpoint": str(args.checkpoint.as_posix()),
                "constrained": True,
                "length": int(sample["length"]),
                "hit_eos": bool(sample["hit_eos"]),
                "ids": [int(value) for value in sample["ids"]],
                "tokens": list(sample["tokens"]),
                "valid": None if validation is None else bool(validation["valid"]),
                "validation_errors": [] if validation is None else list(validation["errors"]),
                "constraint_diagnostics": sample["constraint_diagnostics"],
                "sampling_config": {
                    "max_new_tokens": int(args.max_new_tokens),
                    "temperature": float(args.temperature),
                    "top_k": int(args.top_k),
                    "seed": int(args.seed),
                    "constraint_config": constraint_config.__dict__,
                },
            }
            rows.append(row)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    if bool(args.validate):
        print(f"wrote samples={len(rows)} valid={valid_count} valid_rate={valid_count / max(1, len(rows)):.4f} output={output}")
    else:
        print(f"wrote samples={len(rows)} output={output}")


if __name__ == "__main__":
    main()
