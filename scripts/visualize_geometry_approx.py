from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    parser = argparse.ArgumentParser(description="Visualize old-base geometry approximation.")
    parser.add_argument("--approx-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def _ring_to_vertices_codes(ring):
    points = np.asarray(ring, dtype=np.float32)
    if len(points) == 0:
        return [], []
    vertices = points.tolist() + [points[0].tolist()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(points) - 1) + [mpath.Path.CLOSEPOLY]
    return vertices, codes


def draw_polygon(ax, geometry_block, *, facecolor, edgecolor, linewidth=1.5, alpha=0.35):
    vertices = []
    codes = []
    outer_vertices, outer_codes = _ring_to_vertices_codes(geometry_block["outer"])
    vertices.extend(outer_vertices)
    codes.extend(outer_codes)
    for ring in geometry_block.get("holes", []):
        hole_vertices, hole_codes = _ring_to_vertices_codes(ring)
        vertices.extend(hole_vertices)
        codes.extend(hole_codes)
    path = mpath.Path(np.asarray(vertices, dtype=np.float32), codes)
    patch = patches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(patch)


def annotate(ax, ring, text: str):
    polygon = np.asarray(ring, dtype=np.float32)
    centroid = polygon.mean(axis=0)
    ax.text(centroid[0], centroid[1], text, ha="center", va="center", fontsize=7, color="black", weight="bold")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.approx_json)
    mask = np.array(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(mask_to_rgb(mask))
    draw_polygon(axes[0], payload["original_geometry"], facecolor="#00ff66", edgecolor="red", linewidth=2.0, alpha=0.18)
    bbox = payload["bbox"]
    axes[0].add_patch(
        patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor="red", facecolor="none")
    )
    axes[0].set_title(
        f"A. Source face\nface={payload['face_id']}, label={payload['label']}, holes={payload['original_hole_count']}",
        fontsize=10,
    )

    axes[1].imshow(mask_to_rgb(mask), alpha=0.2)
    draw_polygon(axes[1], payload["original_geometry"], facecolor="none", edgecolor="black", linewidth=1.2, alpha=1.0)
    for primitive in payload["base_primitives"]:
        color = "#ff6b6b" if primitive["type"] == "triangle" else "#2ecc71"
        geom = {"outer": primitive["vertices"], "holes": []}
        draw_polygon(axes[1], geom, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.45)
        annotate(axes[1], primitive["vertices"], str(primitive["id"]))
    axes[1].set_title(
        f"B. Old base decomposition\ncount={payload['base_primitive_count']}, tri={payload['base_triangle_count']}, quad={payload['base_quad_count']}",
        fontsize=10,
    )

    axes[2].imshow(mask_to_rgb(mask), alpha=0.2)
    draw_polygon(axes[2], payload["approx_geometry"], facecolor="#3498db", edgecolor="black", linewidth=1.5, alpha=0.45)
    axes[2].set_title(
        "C. Geometry approximator\n"
        f"approx_vertices={payload['approx_vertex_count']}, IoU={payload['approx_iou']:.3f}",
        fontsize=10,
    )

    for axis in axes:
        axis.set_xlim(0, mask.shape[1])
        axis.set_ylim(mask.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
