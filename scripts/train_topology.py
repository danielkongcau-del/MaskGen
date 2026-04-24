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

from partition_gen.ar_dataset import SparseARDualGraphDataset, collate_sparse_ar, load_binner_meta
from partition_gen.models.topology_transformer import build_model_from_metadata
from partition_gen.topology_training import compute_topology_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sparse autoregressive dual-graph topology model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topology_ar"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=96)
    parser.add_argument("--max-prev-neighbors", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N optimizer steps. 0 disables step-wise eval.")
    parser.add_argument("--save-every", type=int, default=0, help="Save latest checkpoint every N optimizer steps. 0 disables step-wise save.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    data_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    max_faces: int,
    max_prev_neighbors: int,
    binners,
    shuffle: bool,
) -> DataLoader:
    dataset = SparseARDualGraphDataset(
        graph_root=data_root,
        split=split,
        max_faces=max_faces,
        max_prev_neighbors=max_prev_neighbors,
        binners=binners,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sparse_ar,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    tensor_keys = [
        "num_faces",
        "face_mask",
        "node_features",
        "prev_neighbor_indices",
        "prev_neighbor_tokens",
        "prev_neighbor_mask",
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


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "total": 0.0,
        "face_exists": 0.0,
        "node_features": 0.0,
        "prev_count": 0.0,
        "prev_neighbor_index": 0.0,
        "prev_neighbor_token": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, use_amp):
                outputs = model(node_features=batch["node_features"], face_mask=batch["face_mask"])
                losses = compute_topology_losses(outputs, batch)
            for key in totals:
                totals[key] += float(losses[key].item())
            count += 1
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "args": serialized_args,
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
    if "scaler" in checkpoint:
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
) -> float:
    metrics = evaluate(model, val_loader, device, use_amp)
    print(
        f"val epoch={epoch} step={global_step} "
        f"loss={metrics['total']:.4f} "
        f"exists={metrics['face_exists']:.4f} "
        f"node={metrics['node_features']:.4f} "
        f"count={metrics['prev_count']:.4f} "
        f"ptr={metrics['prev_neighbor_index']:.4f} "
        f"edge={metrics['prev_neighbor_token']:.4f}"
    )
    append_jsonl(
        output_dir / "metrics.jsonl",
        {
            "type": "eval",
            "epoch": epoch,
            "step": global_step,
            **metrics,
        },
    )
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

    binner_path = args.data_root / "meta" / "ar_binners.json"
    binners = load_binner_meta(binner_path)
    with binner_path.open("r", encoding="utf-8") as handle:
        binner_meta = json.load(handle)

    train_loader = make_loader(
        data_root=args.data_root,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_prev_neighbors=args.max_prev_neighbors,
        binners=binners,
        shuffle=True,
    )
    val_loader = make_loader(
        data_root=args.data_root,
        split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_prev_neighbors=args.max_prev_neighbors,
        binners=binners,
        shuffle=False,
    )

    model = build_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=args.max_faces,
        max_prev_neighbors=args.max_prev_neighbors,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = build_scaler(use_amp)

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
                outputs = model(node_features=batch["node_features"], face_mask=batch["face_mask"])
                losses = compute_topology_losses(outputs, batch)
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
            train_log = {
                "type": "train",
                "epoch": epoch,
                "step": global_step,
                "total": float(losses["total"].item()),
                "face_exists": float(losses["face_exists"].item()),
                "node_features": float(losses["node_features"].item()),
                "prev_count": float(losses["prev_count"].item()),
                "prev_neighbor_index": float(losses["prev_neighbor_index"].item()),
                "prev_neighbor_token": float(losses["prev_neighbor_token"].item()),
            }

            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={train_log['total']:.4f} "
                    f"exists={train_log['face_exists']:.4f} "
                    f"node={train_log['node_features']:.4f} "
                    f"count={train_log['prev_count']:.4f} "
                    f"ptr={train_log['prev_neighbor_index']:.4f} "
                    f"edge={train_log['prev_neighbor_token']:.4f}"
                )
            append_jsonl(args.output_dir / "metrics.jsonl", train_log)

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
        )
        save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)
        if stop_training:
            break


if __name__ == "__main__":
    main()
