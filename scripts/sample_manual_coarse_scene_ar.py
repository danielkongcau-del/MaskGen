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
from partition_gen.manual_coarse_scene_ar import (  # noqa: E402
    CoarseSceneSamplerConfig,
    evaluate_coarse_scene_sample_rows,
    sample_model_coarse_scene_rows,
    write_coarse_scene_sample_targets,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained sampling for parent-first coarse scene AR checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--max-label", type=int, default=6)
    parser.add_argument("--coarse-grid-bins", type=int, default=8)
    parser.add_argument("--coarse-size-bins", type=int, default=8)
    parser.add_argument("--coarse-aspect-bins", type=int, default=8)
    parser.add_argument("--coarse-angle-bins", type=int, default=8)
    parser.add_argument("--coarse-relation-bins", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--write-graphs", action="store_true")
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    checkpoint, model, _optimizer = load_checkpoint(args.checkpoint, map_location="cpu", load_optimizer=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    tokenizer_config = ParseGraphTokenizerConfig(
        coarse_grid_bins=int(args.coarse_grid_bins),
        coarse_size_bins=int(args.coarse_size_bins),
        coarse_aspect_bins=int(args.coarse_aspect_bins),
        coarse_angle_bins=int(args.coarse_angle_bins),
        coarse_relation_bins=int(args.coarse_relation_bins),
    )
    sampler_config = CoarseSceneSamplerConfig(
        tokenizer_config=tokenizer_config,
        max_actions=int(args.max_actions),
        max_label=int(args.max_label),
    )
    rows = sample_model_coarse_scene_rows(
        model,
        vocab,
        num_samples=int(args.num_samples),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if int(args.top_k) > 0 else None,
        device=device,
        sampler_config=sampler_config,
        progress_every=int(args.progress_every),
    )
    for row in rows:
        row["checkpoint"] = str(args.checkpoint.as_posix())
    output_samples = args.output_samples or (args.checkpoint.parent / "coarse_scene_samples.jsonl")
    _write_jsonl(output_samples, rows)
    summary = evaluate_coarse_scene_sample_rows(rows)
    summary["checkpoint"] = str(args.checkpoint.as_posix())
    summary["output_samples"] = str(output_samples.as_posix())
    if args.output_root is not None and bool(args.write_graphs):
        graph_summary = write_coarse_scene_sample_targets(rows, args.output_root, config=tokenizer_config)
        summary["graph_output"] = graph_summary
    summary_path = output_samples.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(
        f"wrote coarse scene samples={len(rows)} valid={summary['valid_count']} "
        f"valid_rate={summary['valid_rate']:.4f} output={output_samples}"
    )


if __name__ == "__main__":
    main()
