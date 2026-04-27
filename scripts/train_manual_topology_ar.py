from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import time
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import (  # noqa: E402
    append_jsonl,
    autocast_context,
    build_optimizer,
    build_scaler,
    estimate_loss,
    get_lr,
    move_batch_to_device,
    save_checkpoint,
    save_json,
)
from partition_gen.manual_split_token_dataset import (  # noqa: E402
    ManualSplitTokenSequenceDataset,
    collate_manual_split_token_sequences,
)
from partition_gen.models.manual_ar_transformer import ManualARTransformer, ManualARTransformerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train topology-only manual parse graph AR Transformer.")
    parser.add_argument("--train-token-root", type=Path, default=Path("data/remote_256_generator_tokens_manual_split_full/train"))
    parser.add_argument("--val-token-root", type=Path, default=Path("data/remote_256_generator_tokens_manual_split_full/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_topology_ar"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(token_root: Path, *, max_samples: int | None, block_size: int) -> ManualSplitTokenSequenceDataset:
    dataset = ManualSplitTokenSequenceDataset(token_root, sequence_kind="topology", max_length=block_size + 1)
    if max_samples is not None:
        dataset.rows = dataset.rows[: min(int(max_samples), len(dataset.rows))]
    return dataset


def make_loader(dataset: ManualSplitTokenSequenceDataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=lambda batch: collate_manual_split_token_sequences(batch, pad_id=dataset.pad_id),
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args.train_token_root, max_samples=args.max_train_samples, block_size=args.block_size)
    val_dataset = make_dataset(args.val_token_root, max_samples=args.max_val_samples, block_size=args.block_size)
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train and validation datasets must both be non-empty.")

    vocab = train_dataset.vocab
    special_token_ids = {
        "pad": int(vocab["<PAD>"]),
        "bos": int(vocab["<BOS>"]),
        "eos": int(vocab["<EOS>"]),
        "unk": int(vocab["<UNK>"]),
    }
    model_config = ManualARTransformerConfig(
        vocab_size=len(vocab),
        block_size=int(args.block_size),
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_embd=int(args.n_embd),
        dropout=float(args.dropout),
        bias=bool(args.bias),
    )
    device = torch.device(args.device)
    model = ManualARTransformer(model_config).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        device_type=device.type,
    )
    scaler = build_scaler(device, enabled=bool(args.amp))
    iter_num = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        iter_num = int(checkpoint.get("iter_num", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    if args.compile:
        model = torch.compile(model)

    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    train_config["resolved_output_dir"] = str(output_dir.as_posix())
    train_config["train_sample_count"] = int(len(train_dataset))
    train_config["val_sample_count"] = int(len(val_dataset))
    train_config["vocab_size"] = int(len(vocab))
    save_json(output_dir / "config.json", {"model_config": model_config.to_dict(), "train_config": train_config})

    lr_decay_iters = int(args.lr_decay_iters or args.max_iters)
    train_iter = itertools.cycle(train_loader)
    model.train()
    start_time = time.perf_counter()
    while iter_num < int(args.max_iters):
        lr = get_lr(
            iter_num,
            learning_rate=float(args.learning_rate),
            min_lr=float(args.min_lr),
            warmup_iters=int(args.warmup_iters),
            lr_decay_iters=lr_decay_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(int(args.grad_accum_steps)):
            batch = move_batch_to_device(next(train_iter), device)
            with autocast_context(device, enabled=bool(args.amp)):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = outputs["loss"] / float(args.grad_accum_steps)
            total_loss += float(loss.item())
            scaler.scale(loss).backward()
        if float(args.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        iter_num += 1

        if iter_num % int(args.log_every) == 0 or iter_num == 1:
            elapsed = time.perf_counter() - start_time
            row = {"type": "train", "iter": iter_num, "loss": total_loss, "lr": lr, "elapsed_seconds": elapsed}
            append_jsonl(output_dir / "train_log.jsonl", row)
            print(f"iter={iter_num} loss={total_loss:.4f} lr={lr:.6g} elapsed={elapsed:.1f}s")

        if iter_num % int(args.eval_every) == 0 or iter_num == int(args.max_iters):
            val_loss = estimate_loss(model, val_loader, device, eval_iters=int(args.eval_iters), amp=bool(args.amp))
            append_jsonl(output_dir / "train_log.jsonl", {"type": "eval", "iter": iter_num, "val_loss": val_loss})
            print(f"eval iter={iter_num} val_loss={val_loss:.4f}")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(
                output_dir / "ckpt_last.pt",
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                iter_num=iter_num,
                best_val_loss=min(best_val_loss, val_loss),
                model_config=model_config,
                train_config=train_config,
                vocab_path=args.train_token_root / "vocab.json",
                special_token_ids=special_token_ids,
            )
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                save_checkpoint(
                    output_dir / "ckpt_best.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    model_config=model_config,
                    train_config=train_config,
                    vocab_path=args.train_token_root / "vocab.json",
                    special_token_ids=special_token_ids,
                )

        if int(args.save_every) > 0 and iter_num % int(args.save_every) == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(
                output_dir / f"ckpt_iter_{iter_num}.pt",
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                model_config=model_config,
                train_config=train_config,
                vocab_path=args.train_token_root / "vocab.json",
                special_token_ids=special_token_ids,
            )


if __name__ == "__main__":
    main()
