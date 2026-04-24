from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Visualize a primitive decomposition for one partition face.")
    parser.add_argument("--primitive-root", type=Path, default=Path("data/remote_256_primitives"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, default="10")
    parser.add_argument("--face-id", type=int, default=26)
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "compressed", "strip_cover", "strip_refined", "composite_groups"],
        help="Which primitive set to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/face_primitives_val10_26.png"),
    )
    return parser.parse_args()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def _ring_to_vertices_codes(ring, *, reverse: bool = False):
    points = np.asarray(ring, dtype=np.float32)
    if reverse:
        points = points[::-1]
    if len(points) == 0:
        return [], []
    vertices = points.tolist() + [points[0].tolist()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(points) - 1) + [mpath.Path.CLOSEPOLY]
    return vertices, codes


def draw_component(
    ax,
    component,
    *,
    facecolor,
    edgecolor,
    linewidth=1.5,
    alpha=0.45,
    linestyle="-",
):
    vertices = []
    codes = []
    outer_vertices, outer_codes = _ring_to_vertices_codes(component["outer"], reverse=False)
    vertices.extend(outer_vertices)
    codes.extend(outer_codes)
    for hole_ring in component.get("holes", []):
        hole_vertices, hole_codes = _ring_to_vertices_codes(hole_ring, reverse=False)
        vertices.extend(hole_vertices)
        codes.extend(hole_codes)
    if not vertices:
        return
    path = mpath.Path(np.asarray(vertices, dtype=np.float32), codes)
    patch = patches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )
    ax.add_patch(patch)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    primitive_graph = load_json(args.primitive_root / args.split / "graphs" / f"{args.stem}.json")
    mask = np.array(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)
    face = next(item for item in primitive_graph["faces"] if int(item["face_id"]) == args.face_id)
    if args.variant == "base":
        primitive_block = face
    elif args.variant == "composite_groups":
        primitive_block = face["composite_groups"]
    else:
        primitive_block = face[args.variant]
    block_iou = primitive_block.get("approx_iou", face["approx_iou"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(mask_to_rgb(mask))
    bbox = face["bbox"]
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"A. Source mask\nface={args.face_id}, label={face['label']}", fontsize=10)

    axes[1].imshow(mask_to_rgb(mask), alpha=0.25)
    primitive_colors = {
        "triangle": "#ff4d4d",
        "quad": "#2ecc71",
        "convex": "#3498db",
    }
    if args.variant == "composite_groups":
        for group in primitive_block["groups"]:
            components = group.get("components")
            if components:
                for component in components:
                    draw_component(
                        axes[1],
                        component,
                        facecolor="#2ecc71",
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.45,
                    )
            else:
                for ring in group["polygons"]:
                    polygon = np.asarray(ring, dtype=np.float32)
                    patch = patches.Polygon(
                        polygon,
                        closed=True,
                        fill=True,
                        facecolor="#2ecc71",
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.45,
                    )
                    axes[1].add_patch(patch)
            centroid = np.asarray(group["centroid"], dtype=np.float32)
            axes[1].text(
                centroid[0],
                centroid[1],
                f"{group['id']}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
        axes[1].set_title(
            "B. Composite primitive groups\n"
            f"groups={primitive_block['group_count']}, mean atoms/group={primitive_block['mean_atom_count']:.2f}, "
            f"mean holes/group={primitive_block.get('mean_hole_count', 0.0):.2f}",
            fontsize=10,
        )
    else:
        for primitive in primitive_block["primitives"]:
            color = primitive_colors.get(primitive["type"], "#3498db")
            polygon = np.asarray(primitive["vertices"], dtype=np.float32)
            patch = patches.Polygon(
                polygon,
                closed=True,
                fill=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.45,
            )
            axes[1].add_patch(patch)
            centroid = polygon.mean(axis=0)
            axes[1].text(
                centroid[0],
                centroid[1],
                f"{primitive['id']}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
        axes[1].set_title(
            f"B. {args.variant.title()} tri/quad decomposition\n"
            f"count={primitive_block['primitive_count']}, tri={primitive_block['triangle_count']}, "
            f"quad={primitive_block['quad_count']}, convex={primitive_block.get('convex_count', 0)}, "
            f"IoU={block_iou:.3f}",
            fontsize=10,
        )

    for ax in axes:
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output.as_posix()))


if __name__ == "__main__":
    main()
