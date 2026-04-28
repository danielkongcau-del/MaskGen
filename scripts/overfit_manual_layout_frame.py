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
    parser = argparse.ArgumentParser(description="Run a small-sample overfit diagnostic for manual layout/frame MLP.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_layout_frame"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--target-origin-mae", type=float, default=2.0)
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


def write_summary_md(path: Path, payload: dict) -> None:
    lines = [
        "# Manual Layout Frame Overfit Diagnostic",
        "",
        f"- examples: {payload.get('example_count')}",
        f"- epoch: {payload.get('epoch')}",
        f"- train loss: {payload.get('train_loss')}",
        f"- origin MAE: {payload.get('origin_mae')}",
        f"- scale MAE: {payload.get('scale_mae')}",
        f"- orientation MAE: {payload.get('orientation_mae')}",
        f"- reached target: {payload.get('reached_target')}",
        "",
        "| head | accuracy | prediction unique |",
        "| --- | ---: | ---: |",
    ]
    for head, accuracy in (payload.get("head_accuracy", {}) or {}).items():
        hist = (payload.get("head_histograms", {}) or {}).get(head, {})
        lines.append(f"| {head} | {accuracy} | {hist.get('prediction_unique_count')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    run_name = args.run_name or time.strftime("overfit_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_config = ParseGraphTokenizerConfig()
    dataset = ManualLayoutFrameDataset(args.split_root, config=tokenizer_config, max_examples=args.max_examples)
    train_loader = make_loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model_config = ManualLayoutFrameMLPConfig(
        numeric_dim=int(dataset.numeric_dim),
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
    train_config.update({"resolved_output_dir": str(output_dir.as_posix()), "example_count": int(len(dataset))})
    save_json(
        output_dir / "config.json",
        {
            "model_config": asdict(model_config),
            "tokenizer_config": asdict(tokenizer_config),
            "train_config": train_config,
        },
    )

    best_origin_mae = float("inf")
    final_payload: dict = {}
    reached_target = False
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
        if epoch % int(args.eval_every) == 0 or epoch == int(args.max_epochs):
            metrics = evaluate_layout_frame_model(model, eval_loader, device=device, config=tokenizer_config)
            origin_mae = float(metrics.get("origin_mae") or float("inf"))
            final_payload = {
                "format": "maskgen_manual_layout_frame_overfit_eval_v1",
                "epoch": int(epoch),
                "train_loss": train_loss,
                "example_count": int(len(dataset)),
                "reached_target": bool(origin_mae <= float(args.target_origin_mae)),
                **metrics,
            }
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
            if origin_mae < best_origin_mae:
                best_origin_mae = origin_mae
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
            append_jsonl(
                output_dir / "eval_log.jsonl",
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "origin_mae": metrics.get("origin_mae"),
                    "scale_mae": metrics.get("scale_mae"),
                    "orientation_mae": metrics.get("orientation_mae"),
                    **{f"{head}_accuracy": value for head, value in (metrics.get("head_accuracy", {}) or {}).items()},
                },
            )
            if origin_mae <= float(args.target_origin_mae):
                reached_target = True
                break

    final_payload["reached_target"] = bool(reached_target or final_payload.get("reached_target", False))
    (output_dir / "eval_last.json").write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(output_dir / "eval_last.md", final_payload)
    print(
        f"overfit layout frames examples={len(dataset)} origin_mae={final_payload.get('origin_mae'):.4f} "
        f"reached_target={final_payload.get('reached_target')} output={output_dir}"
    )


if __name__ == "__main__":
    main()
