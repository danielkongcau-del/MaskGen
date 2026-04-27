from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
from typing import Dict

import torch
from torch.utils.data import DataLoader

from partition_gen.models.manual_ar_transformer import ManualARTransformer, ManualARTransformerConfig


def autocast_context(device: torch.device, *, enabled: bool):
    if not enabled or device.type == "cpu":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def build_scaler(device: torch.device, *, enabled: bool):
    use_scaler = bool(enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported())
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)
    return torch.cuda.amp.GradScaler(enabled=use_scaler)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    for key in ("input_ids", "labels", "attention_mask", "lengths"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def get_lr(iter_num: int, *, learning_rate: float, min_lr: float, warmup_iters: int, lr_decay_iters: int) -> float:
    if iter_num < int(warmup_iters):
        return float(learning_rate) * float(iter_num + 1) / float(max(1, warmup_iters))
    if iter_num > int(lr_decay_iters):
        return float(min_lr)
    decay_ratio = float(iter_num - warmup_iters) / float(max(1, lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return float(min_lr + coeff * (learning_rate - min_lr))


@torch.no_grad()
def estimate_loss(
    model: ManualARTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    eval_iters: int,
    amp: bool,
) -> float:
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(int(eval_iters)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, enabled=amp):
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
        losses.append(float(outputs["loss"].item()))
    model.train()
    return float(sum(losses) / len(losses)) if losses else float("nan")


def build_optimizer(
    model: ManualARTransformer,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    device_type: str,
) -> torch.optim.Optimizer:
    return model.configure_optimizers(
        weight_decay=float(weight_decay),
        learning_rate=float(learning_rate),
        betas=(float(beta1), float(beta2)),
        device_type=device_type,
    )


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: ManualARTransformer,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    iter_num: int,
    best_val_loss: float,
    model_config: ManualARTransformerConfig,
    train_config: Dict[str, object],
    vocab_path: Path,
    special_token_ids: Dict[str, int],
    best_topology_valid_rate: float | None = None,
    metrics: Dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "maskgen_manual_topology_ar_checkpoint_v1",
        "model_config": model_config.to_dict(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "iter_num": int(iter_num),
        "best_val_loss": float(best_val_loss),
        "best_topology_valid_rate": None if best_topology_valid_rate is None else float(best_topology_valid_rate),
        "metrics": {} if metrics is None else metrics,
        "train_config": train_config,
        "vocab_path": str(vocab_path.as_posix()),
        "special_token_ids": special_token_ids,
        "git_commit": git_commit_hash(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    load_optimizer: bool = False,
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_config = ManualARTransformerConfig(**checkpoint["model_config"])
    model = ManualARTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    optimizer = None
    if load_optimizer and checkpoint.get("optimizer") is not None:
        optimizer = build_optimizer(
            model,
            learning_rate=float(checkpoint.get("train_config", {}).get("learning_rate", 3e-4)),
            weight_decay=float(checkpoint.get("train_config", {}).get("weight_decay", 0.1)),
            beta1=float(checkpoint.get("train_config", {}).get("beta1", 0.9)),
            beta2=float(checkpoint.get("train_config", {}).get("beta2", 0.95)),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint, model, optimizer
