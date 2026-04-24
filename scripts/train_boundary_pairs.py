from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.ar_dataset import load_binner_meta
from partition_gen.boundary_dataset import PairBoundaryGraphDataset, collate_pair_boundary_graphs
from partition_gen.models.pair_boundary_predictor import build_pair_boundary_model_from_metadata
from partition_gen.pair_boundary_training import compute_pair_boundary_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pair-level graph-conditioned boundary predictor.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/boundary_pairs"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=96)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--pair-hidden", type=int, default=128)
    parser.add_argument("--pair-token-channels", type=int, default=32)
    parser.add_argument("--decoder-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.25)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


def make_loader(
    *,
    dual_root: Path,
    boundary_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    max_faces: int,
    max_neighbors: int | None,
    max_pairs: int | None,
    max_samples: int | None,
    target_size: int,
    binners,
    shuffle: bool,
) -> DataLoader:
    dataset = PairBoundaryGraphDataset(
        dual_root=dual_root,
        boundary_root=boundary_root,
        split=split,
        binners=binners,
        max_faces=max_faces,
        max_neighbors=max_neighbors,
        max_pairs=max_pairs,
        target_size=target_size,
    )
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_boundary_graphs,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    tensor_keys = [
        "num_faces",
        "face_mask",
        "node_features",
        "neighbor_indices",
        "neighbor_tokens",
        "neighbor_mask",
        "centroid_ratios",
        "bbox_ratios",
        "labels",
        "seed_points",
        "seed_mask",
        "num_pairs",
        "pair_indices",
        "pair_features",
        "pair_masks",
        "pair_is_border",
        "pair_valid",
        "union_mask",
    ]
    for key in tensor_keys:
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def build_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        if torch.cuda.is_available():
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        return torch.amp.GradScaler("cpu", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def compute_pos_weight(dataset) -> float:
    positive = 0.0
    total = 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        mask = item["pair_masks"]
        valid = item["pair_valid"][:, None, None, None].to(mask.dtype)
        positive += float((mask * valid).sum().item())
        total += float(valid.sum().item()) * mask.shape[-1] * mask.shape[-2]
    negative = max(total - positive, 1.0)
    return negative / max(positive, 1.0)


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    *,
    pos_weight: torch.Tensor | None,
    count_loss_weight: float,
    overlap_loss_weight: float,
) -> Dict[str, float]:
    totals = {
        "total": 0.0,
        "pair_bce": 0.0,
        "pair_dice": 0.0,
        "union_bce": 0.0,
        "union_dice": 0.0,
        "count_l1": 0.0,
        "overlap_l1": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, use_amp):
                outputs = model(
                    node_features=batch["node_features"],
                    face_mask=batch["face_mask"],
                    neighbor_indices=batch["neighbor_indices"],
                    neighbor_tokens=batch["neighbor_tokens"],
                    neighbor_mask=batch["neighbor_mask"],
                    centroid_ratios=batch["centroid_ratios"],
                    bbox_ratios=batch["bbox_ratios"],
                    pair_indices=batch["pair_indices"],
                    pair_features=batch["pair_features"],
                    pair_valid=batch["pair_valid"],
                )
                losses = compute_pair_boundary_losses(
                    outputs,
                    batch,
                    pos_weight=pos_weight,
                    count_loss_weight=count_loss_weight,
                    overlap_loss_weight=overlap_loss_weight,
                )
            for key in totals:
                totals[key] += float(losses[key].item())
            count += 1
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, global_step: int, best_val_loss: float, args: argparse.Namespace) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    torch.save(checkpoint, path)


def optimizer_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(path: Path, model, optimizer, scaler, map_location: str = "cpu") -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def maybe_run_eval(
    *,
    model,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    output_dir: Path,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    optimizer,
    scaler,
    args: argparse.Namespace,
    pos_weight: torch.Tensor | None,
) -> float:
    metrics = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        pos_weight=pos_weight,
        count_loss_weight=args.count_loss_weight,
        overlap_loss_weight=args.overlap_loss_weight,
    )
    print(
        f"val epoch={epoch} step={global_step} "
        f"loss={metrics['total']:.4f} pair_bce={metrics['pair_bce']:.4f} "
        f"pair_dice={metrics['pair_dice']:.4f} union_bce={metrics['union_bce']:.4f} "
        f"union_dice={metrics['union_dice']:.4f} count_l1={metrics['count_l1']:.4f} "
        f"overlap_l1={metrics['overlap_l1']:.4f}"
    )
    append_jsonl(output_dir / "metrics.jsonl", {"type": "eval", "epoch": epoch, "step": global_step, **metrics})
    save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)
    if metrics["total"] < best_val_loss:
        best_val_loss = metrics["total"]
        save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)
    return best_val_loss


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    save_json(args.output_dir / "config.json", vars(args))

    binner_path = args.dual_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    neighbor_cap = args.max_neighbors if args.max_neighbors > 0 else None
    pair_cap = args.max_pairs if args.max_pairs > 0 else None
    train_loader = make_loader(
        dual_root=args.dual_root,
        boundary_root=args.boundary_root,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_neighbors=neighbor_cap,
        max_pairs=pair_cap,
        max_samples=args.max_train_samples,
        target_size=args.target_size,
        binners=binners,
        shuffle=True,
    )
    val_loader = make_loader(
        dual_root=args.dual_root,
        boundary_root=args.boundary_root,
        split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_neighbors=neighbor_cap,
        max_pairs=pair_cap,
        max_samples=args.max_val_samples,
        target_size=args.target_size,
        binners=binners,
        shuffle=False,
    )

    model = build_pair_boundary_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=args.max_faces,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        target_size=args.target_size,
        pair_hidden=args.pair_hidden,
        pair_token_channels=args.pair_token_channels,
        decoder_hidden=args.decoder_hidden,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = build_scaler(use_amp)

    pos_weight_value = None
    if not args.disable_pos_weight:
        pos_weight_value = compute_pos_weight(train_loader.dataset) * max(args.pos_weight_scale, 0.0)
    pos_weight = None if pos_weight_value is None else torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    save_json(
        args.output_dir / "pair_boundary_weights.json",
        {
            "disable_pos_weight": bool(args.disable_pos_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "pos_weight": float(pos_weight_value) if pos_weight_value is not None else None,
        },
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scaler, map_location="cpu")
        optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resumed checkpoint={args.resume} epoch={start_epoch} step={global_step} best_val={best_val_loss:.4f}")

    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, use_amp):
                outputs = model(
                    node_features=batch["node_features"],
                    face_mask=batch["face_mask"],
                    neighbor_indices=batch["neighbor_indices"],
                    neighbor_tokens=batch["neighbor_tokens"],
                    neighbor_mask=batch["neighbor_mask"],
                    centroid_ratios=batch["centroid_ratios"],
                    bbox_ratios=batch["bbox_ratios"],
                    pair_indices=batch["pair_indices"],
                    pair_features=batch["pair_features"],
                    pair_valid=batch["pair_valid"],
                )
                losses = compute_pair_boundary_losses(
                    outputs,
                    batch,
                    pos_weight=pos_weight,
                    count_loss_weight=args.count_loss_weight,
                    overlap_loss_weight=args.overlap_loss_weight,
                )
                scaled_loss = losses["total"] / max(1, args.grad_accum_steps)

            scaler.scale(scaled_loss).backward()
            should_step = (micro_step % args.grad_accum_steps == 0) or (micro_step == len(train_loader))
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            metrics = {
                "type": "train",
                "epoch": epoch,
                "step": global_step,
                "total": float(losses["total"].item()),
                "pair_bce": float(losses["pair_bce"].item()),
                "pair_dice": float(losses["pair_dice"].item()),
                "union_bce": float(losses["union_bce"].item()),
                "union_dice": float(losses["union_dice"].item()),
                "count_l1": float(losses["count_l1"].item()),
                "overlap_l1": float(losses["overlap_l1"].item()),
            }
            append_jsonl(args.output_dir / "metrics.jsonl", metrics)
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"step={global_step} epoch={epoch} loss={metrics['total']:.4f} "
                    f"pair_bce={metrics['pair_bce']:.4f} pair_dice={metrics['pair_dice']:.4f} "
                    f"union_bce={metrics['union_bce']:.4f} union_dice={metrics['union_dice']:.4f} "
                    f"count_l1={metrics['count_l1']:.4f} overlap_l1={metrics['overlap_l1']:.4f}"
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                best_val_loss = maybe_run_eval(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    use_amp=use_amp,
                    output_dir=args.output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    pos_weight=pos_weight,
                )
                model.train()

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop_training = True
                break

        best_val_loss = maybe_run_eval(
            model=model,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
            output_dir=args.output_dir,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            pos_weight=pos_weight,
        )
        save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)
        if stop_training:
            break


if __name__ == "__main__":
    main()
