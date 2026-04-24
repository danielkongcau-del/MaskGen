from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.geometry_render import load_json


PALETTE = np.asarray(
    [
        [245, 245, 245],
        [233, 196, 106],
        [42, 157, 143],
        [38, 70, 83],
        [231, 111, 81],
        [244, 162, 97],
        [106, 76, 147],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pair-boundary prediction on one sample.")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/boundary_pairs_short_predictions_w005_t025_pairaware"),
    )
    parser.add_argument("--sample-id", type=str, default="0001")
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/pair_boundary_demo_0001.png"))
    return parser.parse_args()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def load_binary_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8) > 0


def upsample_float_map(array: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32))[None, None]
    up = F.interpolate(tensor, size=(size, size), mode="nearest")
    return up[0, 0].numpy()


def error_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    correct = pred == gt
    wrong = ~correct
    rgb[correct] = np.asarray([40, 40, 40], dtype=np.uint8)
    rgb[wrong] = np.asarray([220, 20, 60], dtype=np.uint8)
    return rgb


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report = load_json(args.prediction_root / "reports" / f"{args.sample_id}.json")
    dual_graph = load_json(Path(report["dual_path"]))
    split = Path(report["dual_path"]).parent.parent.name
    stem = Path(report["dual_path"]).stem

    gt_mask = np.array(Image.open(args.mask_root / str(dual_graph["source_mask"])), dtype=np.uint8)
    gt_boundary = load_binary_png(args.boundary_root / split / "boundary_masks" / f"{stem}.png")
    pred_union = load_binary_png(args.prediction_root / "union_masks" / f"{args.sample_id}_256.png")
    pred_mask = np.array(Image.open(args.prediction_root / "masks_id" / f"{args.sample_id}.png"), dtype=np.uint8)

    pair_paths = sorted((args.prediction_root / "pair_masks").glob(f"{args.sample_id}_*.png"))
    candidate_count_64 = None
    for pair_path in pair_paths:
        pair_mask = load_binary_png(pair_path).astype(np.float32)
        if candidate_count_64 is None:
            candidate_count_64 = np.zeros_like(pair_mask, dtype=np.float32)
        candidate_count_64 += pair_mask
    if candidate_count_64 is None:
        candidate_count_64 = np.zeros((64, 64), dtype=np.float32)
    candidate_count = upsample_float_map(candidate_count_64, gt_mask.shape[0])

    pixel_acc = float((pred_mask == gt_mask).mean())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].imshow(mask_to_rgb(gt_mask))
    axes[0, 0].set_title("A. Ground Truth Mask", fontsize=10)

    axes[0, 1].imshow(gt_boundary, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("B. GT Shared Boundary", fontsize=10)

    im = axes[0, 2].imshow(candidate_count, cmap="magma")
    axes[0, 2].set_title(
        f"C. Pair overlap count\nmax={int(candidate_count.max())}, ambiguous={report['ambiguous_boundary_pixels']}",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(pred_union, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"D. Predicted Union Boundary\nunion IoU={report['union_iou']:.3f}",
        fontsize=10,
    )

    axes[1, 1].imshow(mask_to_rgb(pred_mask))
    axes[1, 1].set_title(
        f"E. Pair-aware Rendered Mask\npixel acc={pixel_acc:.3f}",
        fontsize=10,
    )

    axes[1, 2].imshow(error_rgb(pred_mask, gt_mask))
    axes[1, 2].set_title("F. Error Map\nred = wrong", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Pair-level boundary supervision -> pair-aware rendering\n"
        f"sample={args.sample_id}, dual={Path(report['dual_path']).as_posix()}",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output.as_posix()))


if __name__ == "__main__":
    main()
