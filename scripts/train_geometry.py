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
from partition_gen.geometry_dataset import GeometryGraphDataset, collate_geometry_graphs
from partition_gen.geometry_training import compute_geometry_losses
from partition_gen.models.geometry_decoder import build_geometry_model_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the graph-conditioned geometry decoder.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--geometry-root", type=Path, default=Path("data/remote_256_geometry"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_decoder"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=96)
    parser.add_argument("--max-neighbors", type=int, default=0, help="0 disables neighbor-cap filtering.")
    parser.add_argument("--max-vertices", type=int, default=32)
    parser.add_argument("--max-holes", type=int, default=0)
    parser.add_argument("--max-hole-vertices", type=int, default=0)
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
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--support-class-weighting", type=str, choices=["none", "inverse", "inverse_sqrt"], default="none")
    parser.add_argument("--vertex-count-class-weighting", type=str, choices=["none", "inverse", "inverse_sqrt"], default="none")
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


def make_loader(
    *,
    dual_root: Path,
    geometry_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    max_faces: int,
    max_neighbors: int | None,
    max_vertices: int,
    max_holes: int,
    max_hole_vertices: int,
    max_samples: int | None,
    binners,
    shuffle: bool,
) -> DataLoader:
    dataset = GeometryGraphDataset(
        dual_root=dual_root,
        geometry_root=geometry_root,
        split=split,
        binners=binners,
        max_faces=max_faces,
        max_neighbors=max_neighbors,
        max_vertices=max_vertices,
        max_holes=max_holes,
        max_hole_vertices=max_hole_vertices,
    )
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_geometry_graphs,
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
        "geometry_support",
        "vertex_counts",
        "vertices",
        "vertex_mask",
        "hole_counts",
        "hole_vertex_counts",
        "hole_vertices",
        "hole_mask",
        "hole_vertex_mask",
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


def _counts_to_weights(counts: torch.Tensor, strategy: str, *, active_mask: torch.Tensor | None = None) -> torch.Tensor | None:
    if strategy == "none":
        return None
    counts = counts.to(torch.float32)
    if active_mask is None:
        active_mask = counts > 0
    if not bool(active_mask.any()):
        return None
    safe = counts.clamp(min=1.0)
    if strategy == "inverse":
        weights = 1.0 / safe
    elif strategy == "inverse_sqrt":
        weights = torch.rsqrt(safe)
    else:
        raise ValueError(f"Unsupported weighting strategy: {strategy}")
    weights = weights * active_mask.to(weights.dtype)
    weights = weights / weights[active_mask].mean().clamp(min=1e-6)
    return weights


def build_geometry_class_weights(
    dataset,
    *,
    max_vertices: int,
    support_strategy: str,
    vertex_count_strategy: str,
) -> Dict[str, torch.Tensor | None]:
    support_counts = torch.zeros(2, dtype=torch.float32)
    vertex_count_counts = torch.zeros(max_vertices + 1, dtype=torch.float32)

    for index in range(len(dataset)):
        item = dataset[index]
        support = item["geometry_support"].to(torch.long)
        support_counts += torch.bincount(support, minlength=2).to(torch.float32)
        supported_counts = item["vertex_counts"][support.bool()].to(torch.long)
        if supported_counts.numel() > 0:
            vertex_count_counts += torch.bincount(supported_counts, minlength=max_vertices + 1).to(torch.float32)

    vertex_active_mask = vertex_count_counts > 0
    if vertex_active_mask.numel() > 0:
        vertex_active_mask[0] = False

    return {
        "support_counts": support_counts,
        "vertex_count_counts": vertex_count_counts,
        "support_weights": _counts_to_weights(support_counts, support_strategy),
        "vertex_count_weights": _counts_to_weights(
            vertex_count_counts,
            vertex_count_strategy,
            active_mask=vertex_active_mask,
        ),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


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


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    *,
    support_class_weights: torch.Tensor | None = None,
    vertex_count_class_weights: torch.Tensor | None = None,
) -> Dict[str, float]:
    totals = {
        "total": 0.0,
        "support": 0.0,
        "vertex_count": 0.0,
        "coords": 0.0,
        "hole_count": 0.0,
        "hole_vertex_count": 0.0,
        "hole_coords": 0.0,
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
                )
                losses = compute_geometry_losses(
                    outputs,
                    batch,
                    support_class_weights=support_class_weights,
                    vertex_count_class_weights=vertex_count_class_weights,
                )
            for key in totals:
                totals[key] += float(losses[key].item())
            count += 1
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


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
    support_class_weights: torch.Tensor | None = None,
    vertex_count_class_weights: torch.Tensor | None = None,
) -> float:
    metrics = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        support_class_weights=support_class_weights,
        vertex_count_class_weights=vertex_count_class_weights,
    )
    print(
        f"val epoch={epoch} step={global_step} "
        f"loss={metrics['total']:.4f} support={metrics['support']:.4f} "
        f"count={metrics['vertex_count']:.4f} coords={metrics['coords']:.4f} "
        f"hole_count={metrics['hole_count']:.4f} hole_vcount={metrics['hole_vertex_count']:.4f} "
        f"hole_coords={metrics['hole_coords']:.4f}"
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
    train_loader = make_loader(
        dual_root=args.dual_root,
        geometry_root=args.geometry_root,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_neighbors=neighbor_cap,
        max_vertices=args.max_vertices,
        max_holes=args.max_holes,
        max_hole_vertices=args.max_hole_vertices,
        max_samples=args.max_train_samples,
        binners=binners,
        shuffle=True,
    )
    val_loader = make_loader(
        dual_root=args.dual_root,
        geometry_root=args.geometry_root,
        split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_faces=args.max_faces,
        max_neighbors=neighbor_cap,
        max_vertices=args.max_vertices,
        max_holes=args.max_holes,
        max_hole_vertices=args.max_hole_vertices,
        max_samples=args.max_val_samples,
        binners=binners,
        shuffle=False,
    )

    model = build_geometry_model_from_metadata(
        binner_meta=binner_meta,
        max_faces=args.max_faces,
        max_neighbors=max(1, args.max_neighbors) if args.max_neighbors > 0 else 32,
        max_vertices=args.max_vertices,
        max_holes=args.max_holes,
        max_hole_vertices=args.max_hole_vertices,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = build_scaler(use_amp)

    class_weight_meta = build_geometry_class_weights(
        train_loader.dataset,
        max_vertices=args.max_vertices,
        support_strategy=args.support_class_weighting,
        vertex_count_strategy=args.vertex_count_class_weighting,
    )
    support_class_weights = class_weight_meta["support_weights"]
    vertex_count_class_weights = class_weight_meta["vertex_count_weights"]
    if support_class_weights is not None:
        support_class_weights = support_class_weights.to(device)
    if vertex_count_class_weights is not None:
        vertex_count_class_weights = vertex_count_class_weights.to(device)
    save_json(
        args.output_dir / "class_weights.json",
        {
            "support_class_weighting": args.support_class_weighting,
            "vertex_count_class_weighting": args.vertex_count_class_weighting,
            "support_counts": class_weight_meta["support_counts"].tolist(),
            "vertex_count_counts": class_weight_meta["vertex_count_counts"].tolist(),
            "support_weights": class_weight_meta["support_weights"].tolist() if class_weight_meta["support_weights"] is not None else None,
            "vertex_count_weights": class_weight_meta["vertex_count_weights"].tolist() if class_weight_meta["vertex_count_weights"] is not None else None,
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
                )
                losses = compute_geometry_losses(
                    outputs,
                    batch,
                    support_class_weights=support_class_weights,
                    vertex_count_class_weights=vertex_count_class_weights,
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
                "support": float(losses["support"].item()),
                "vertex_count": float(losses["vertex_count"].item()),
                "coords": float(losses["coords"].item()),
                "hole_count": float(losses["hole_count"].item()),
                "hole_vertex_count": float(losses["hole_vertex_count"].item()),
                "hole_coords": float(losses["hole_coords"].item()),
            }
            append_jsonl(args.output_dir / "metrics.jsonl", metrics)
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={metrics['total']:.4f} support={metrics['support']:.4f} "
                    f"count={metrics['vertex_count']:.4f} coords={metrics['coords']:.4f} "
                    f"hole_count={metrics['hole_count']:.4f} hole_vcount={metrics['hole_vertex_count']:.4f} "
                    f"hole_coords={metrics['hole_coords']:.4f}"
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
                    support_class_weights=support_class_weights,
                    vertex_count_class_weights=vertex_count_class_weights,
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
            support_class_weights=support_class_weights,
            vertex_count_class_weights=vertex_count_class_weights,
        )
        save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, args)
        if stop_training:
            break


if __name__ == "__main__":
    main()
