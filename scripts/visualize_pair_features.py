from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.boundary_dataset import (  # noqa: E402
    BOUNDARY_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    _build_pair_target_mask,
    build_graph_pair_specs,
)
from partition_gen.geometry_render import load_json  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Visualize the geometric features used by the pair-boundary model.")
    parser.add_argument("--dual-root", type=Path, default=Path("data/remote_256_dual"))
    parser.add_argument("--boundary-root", type=Path, default=Path("data/remote_256_boundary"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, default="10")
    parser.add_argument("--u", type=int, default=26)
    parser.add_argument("--v", type=int, default=27)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/pair_feature_demo_val10_26_27.png"),
    )
    return parser.parse_args()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def soft_box(grid_x: np.ndarray, grid_y: np.ndarray, bbox_ratio: np.ndarray, tau: float) -> np.ndarray:
    x0, y0, x1, y1 = bbox_ratio
    box = 1.0 / (1.0 + np.exp(-(grid_x - x0) / tau))
    box *= 1.0 / (1.0 + np.exp(-(x1 - grid_x) / tau))
    box *= 1.0 / (1.0 + np.exp(-(grid_y - y0) / tau))
    box *= 1.0 / (1.0 + np.exp(-(y1 - grid_y) / tau))
    return box


def seed_heat(grid_x: np.ndarray, grid_y: np.ndarray, centroid_ratio: np.ndarray, sigma: float) -> np.ndarray:
    cx, cy = centroid_ratio
    return np.exp(-0.5 * (((grid_x - cx) / sigma) ** 2 + ((grid_y - cy) / sigma) ** 2))


def draw_face_box(ax, face: dict, color: str, label: str) -> None:
    x0, y0, x1, y1 = face["bbox"]
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.plot(face["centroid"][0], face["centroid"][1], marker="x", color=color, markersize=8, markeredgewidth=2)
    ax.text(face["centroid"][0] + 4, face["centroid"][1] + 4, label, color=color, fontsize=9, weight="bold")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dual_graph = load_json(args.dual_root / args.split / "graphs" / f"{args.stem}.json")
    boundary_graph = load_json(args.boundary_root / args.split / "graphs" / f"{args.stem}.json")
    gt_mask = np.array(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)

    faces_by_id = {int(face["id"]): face for face in dual_graph["faces"]}
    face_u = faces_by_id[args.u]
    face_v = faces_by_id[args.v]

    pair_specs = build_graph_pair_specs(dual_graph)
    pair_spec = next(spec for spec in pair_specs if (int(spec["u"]), int(spec["v"])) == (args.u, args.v))
    pair_features = np.asarray(pair_spec["pair_features"], dtype=np.float32)

    edge_group = next(
        group
        for group in boundary_graph["edge_groups"]
        if tuple(sorted(int(value) for value in group["faces"])) == tuple(sorted((args.u, args.v)))
    )
    target_mask = _build_pair_target_mask(
        edge_group,
        source_size=int(dual_graph["size"][0]),
        target_size=args.target_size,
    )

    coords = (np.arange(args.target_size, dtype=np.float32) + 0.5) / float(args.target_size)
    grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
    tau = 1.5 / float(args.target_size)
    sigma = 1.25 / float(args.target_size)

    box_u = soft_box(grid_x, grid_y, np.asarray(face_u["bbox_ratio"], dtype=np.float32), tau=tau)
    box_v = soft_box(grid_x, grid_y, np.asarray(face_v["bbox_ratio"], dtype=np.float32), tau=tau)
    union_box = soft_box(grid_x, grid_y, pair_features[4:8], tau=tau)
    seed_u = seed_heat(grid_x, grid_y, np.asarray(face_u["centroid_ratio"], dtype=np.float32), sigma=sigma)
    seed_v = seed_heat(grid_x, grid_y, np.asarray(face_v["centroid_ratio"], dtype=np.float32), sigma=sigma)

    soft_box_rgb = np.stack([box_u, box_v, union_box], axis=-1)
    soft_box_rgb = np.clip(soft_box_rgb / np.maximum(soft_box_rgb.max(), 1e-6), 0.0, 1.0)
    seed_rgb = np.stack([seed_u, seed_v, np.maximum(seed_u, seed_v)], axis=-1)
    seed_rgb = np.clip(seed_rgb / np.maximum(seed_rgb.max(), 1e-6), 0.0, 1.0)

    face_feature_text = []
    for prefix, face in [("u", face_u), ("v", face_v)]:
        values = {
            "label": face["label"],
            "area_ratio": round(float(face["area_ratio"]), 4),
            "centroid_x": round(float(face["centroid_ratio"][0]), 3),
            "centroid_y": round(float(face["centroid_ratio"][1]), 3),
            "bbox_width_ratio": round(float(face["bbox_width_ratio"]), 3),
            "bbox_height_ratio": round(float(face["bbox_height_ratio"]), 3),
            "perimeter_ratio": round(float(face["perimeter_ratio"]), 4),
            "border_ratio": round(float(face["border_ratio"]), 4),
            "outer_vertices": int(face["outer_vertices"]),
            "hole_count": int(face["hole_count"]),
            "degree": int(face["degree"]),
            "touches_border": bool(face["touches_border"]),
        }
        face_feature_text.append(f"{prefix}=face {face['id']}")
        for key in BOUNDARY_FEATURE_NAMES:
            value = values[key]
            face_feature_text.append(f"  {key}: {value}")
        face_feature_text.append(
            f"  bbox_ratio(full): {tuple(round(float(v), 3) for v in face['bbox_ratio'])}"
        )

    pair_feature_text = ["", "pair features"]
    for name, value in zip(PAIR_FEATURE_NAMES, pair_features.tolist()):
        pair_feature_text.append(f"  {name}: {value:.4f}")

    text_block = "\n".join(face_feature_text + pair_feature_text)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    axes[0, 0].imshow(mask_to_rgb(gt_mask))
    draw_face_box(axes[0, 0], face_u, "#ff4d4d", "u")
    draw_face_box(axes[0, 0], face_v, "#2ecc71", "v")
    axes[0, 0].set_title("A. GT mask + selected faces", fontsize=10)

    axes[0, 1].imshow(mask_to_rgb(gt_mask))
    axes[0, 1].plot(face_u["centroid"][0], face_u["centroid"][1], "o", color="#ff4d4d", markersize=6)
    axes[0, 1].plot(face_v["centroid"][0], face_v["centroid"][1], "o", color="#2ecc71", markersize=6)
    axes[0, 1].plot(
        [face_u["centroid"][0], face_v["centroid"][0]],
        [face_u["centroid"][1], face_v["centroid"][1]],
        "--",
        color="#f1c40f",
        linewidth=2,
    )
    axes[0, 1].set_title("B. centroid delta / relative position", fontsize=10)

    axes[0, 2].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("C. GT pair boundary target (64x64)", fontsize=10)

    axes[1, 0].imshow(soft_box_rgb)
    axes[1, 0].set_title("D. soft boxes\nR=u bbox, G=v bbox, B=union bbox", fontsize=10)

    axes[1, 1].imshow(seed_rgb)
    axes[1, 1].set_title("E. seed heatmaps\nR=u seed, G=v seed, B=max", fontsize=10)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        text_block,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    axes[1, 2].set_title("F. raw feature values", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Pair-boundary model geometry features\nsample={args.split}/{args.stem}, pair=({args.u}, {args.v})",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output.as_posix()))


if __name__ == "__main__":
    main()
