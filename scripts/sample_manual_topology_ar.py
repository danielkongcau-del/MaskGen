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
from partition_gen.parse_graph_tokenizer import ids_to_tokens, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample topology token sequences from a manual AR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    checkpoint, model, _ = load_checkpoint(args.checkpoint, map_location="cpu", load_optimizer=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if not vocab_path.exists():
        train_config = checkpoint.get("train_config", {})
        candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
        vocab_path = candidate if candidate.exists() else vocab_path
    vocab = load_vocabulary(vocab_path)
    bos_id = int(checkpoint.get("special_token_ids", {}).get("bos", vocab["<BOS>"]))
    eos_id = int(checkpoint.get("special_token_ids", {}).get("eos", vocab["<EOS>"]))

    output = args.output or (args.checkpoint.parent / "samples.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for index in range(int(args.num_samples)):
            start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            generated = model.generate(
                start,
                max_new_tokens=int(args.max_new_tokens),
                eos_id=eos_id,
                temperature=float(args.temperature),
                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            )[0].detach().cpu().tolist()
            hit_eos = bool(eos_id in generated[1:])
            row = {
                "format": "maskgen_manual_topology_ar_sample_v1",
                "sample_index": int(index),
                "checkpoint": str(args.checkpoint.as_posix()),
                "length": int(len(generated)),
                "hit_eos": hit_eos,
                "ids": [int(value) for value in generated],
                "tokens": ids_to_tokens(generated, vocab),
                "sampling_config": {
                    "max_new_tokens": int(args.max_new_tokens),
                    "temperature": float(args.temperature),
                    "top_k": int(args.top_k),
                    "seed": int(args.seed),
                },
            }
            rows.append(row)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote samples={len(rows)} output={output}")


if __name__ == "__main__":
    main()
