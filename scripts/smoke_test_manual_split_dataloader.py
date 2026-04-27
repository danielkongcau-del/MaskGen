from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_split_token_dataset import build_manual_split_token_dataloader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test manual split token PyTorch dataloader.")
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--sequence-kind", choices=["topology", "geometry"], default="topology")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = build_manual_split_token_dataloader(
        args.token_root,
        sequence_kind=args.sequence_kind,
        batch_size=int(args.batch_size),
        max_length=args.max_length,
    )
    batch = next(iter(loader))
    print(
        f"batch sequence_kind={batch['sequence_kind']} "
        f"input_shape={tuple(batch['input_ids'].shape)} "
        f"labels_shape={tuple(batch['labels'].shape)} "
        f"active_tokens={int(batch['attention_mask'].sum().item())}"
    )


if __name__ == "__main__":
    main()
