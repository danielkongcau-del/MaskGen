from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import append_jsonl, save_json  # noqa: E402
from partition_gen.manual_layout_frame import (  # noqa: E402
    ManualLayoutFrameDataset,
    ManualLayoutFrameMLP,
    ManualLayoutFrameMLPConfig,
    collate_layout_frame_examples,
    evaluate_layout_frame_model,
    layout_frame_loss,
    move_layout_batch_to_device,
    save_layout_frame_checkpoint,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train topology-conditioned manual layout/frame predictor.")
    parser.add_argument("--train-split-root", type=Path, required=True)
    parser.add_argument("--val-split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_layout_frame"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ManualLayoutFrameDataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_layout_frame_examples,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_config = ParseGraphTokenizerConfig()

    train_dataset = ManualLayoutFrameDataset(
        args.train_split_root,
        config=tokenizer_config,
        max_samples=args.max_train_samples,
    )
    val_dataset = ManualLayoutFrameDataset(
        args.val_split_root,
        config=tokenizer_config,
        max_samples=args.max_val_samples,
    )
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_config = ManualLayoutFrameMLPConfig(
        numeric_dim=int(train_dataset.numeric_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        position_bins=int(tokenizer_config.position_bins),
        scale_bins=int(tokenizer_config.scale_bins),
        angle_bins=int(tokenizer_config.angle_bins),
    )
    device = torch.device(args.device)
    model = ManualLayoutFrameMLP(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    train_config.update(
        {
            "resolved_output_dir": str(output_dir.as_posix()),
            "train_example_count": int(len(train_dataset)),
            "val_example_count": int(len(val_dataset)),
        }
    )
    save_json(
        output_dir / "config.json",
        {
            "model_config": asdict(model_config),
            "tokenizer_config": asdict(tokenizer_config),
            "train_config": train_config,
        },
    )

    best_metric = float("inf")
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            moved = move_layout_batch_to_device(batch, device)
            logits = model(moved)
            loss = layout_frame_loss(logits, moved)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        log_row = {"epoch": int(epoch), "train_loss": train_loss}
        if epoch % int(args.eval_every) == 0 or epoch == int(args.max_epochs):
            metrics = evaluate_layout_frame_model(model, val_loader, device=device, config=tokenizer_config)
            log_row.update({f"val_{key}": value for key, value in metrics.items() if key != "role_metrics"})
            current_metric = float(metrics.get("origin_mae") or float("inf"))
            save_layout_frame_checkpoint(
                output_dir / "ckpt_last.pt",
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                train_config=train_config,
                metrics=metrics,
                epoch=epoch,
            )
            if current_metric < best_metric:
                best_metric = current_metric
                save_layout_frame_checkpoint(
                    output_dir / "ckpt_best.pt",
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    tokenizer_config=tokenizer_config,
                    train_config=train_config,
                    metrics=metrics,
                    epoch=epoch,
                )
            save_json(output_dir / "eval_last.json", {"epoch": epoch, **metrics})
        append_jsonl(output_dir / "eval_log.jsonl", log_row)

    print(f"training complete epochs={args.max_epochs} best_origin_mae={best_metric:.4f} output={output_dir}")


if __name__ == "__main__":
    main()
