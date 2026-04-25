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
    parser = argparse.ArgumentParser(description="Visualize before/after global arc regularization.")
    parser.add_argument("--before-json", type=Path, required=True)
    parser.add_argument("--after-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_split_stem(payload: dict, args: argparse.Namespace) -> tuple[str, str]:
    if args.split and args.stem:
        return args.split, args.stem
    source_mask = str(payload.get("source_mask") or "")
    if source_mask:
        path = Path(source_mask)
        return path.parts[0], path.stem
    source_graph = str(payload.get("source_partition_graph") or "")
    path = Path(source_graph)
    if len(path.parts) >= 3:
        return path.parts[-3], path.stem
    raise ValueError("Could not infer split/stem; pass --split and --stem.")


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


def draw_polygon(ax, outer, holes, *, facecolor, edgecolor, linewidth=1.0, alpha=0.35):
    vertices = []
    codes = []
    outer_vertices, outer_codes = _ring_to_vertices_codes(outer)
    vertices.extend(outer_vertices)
    codes.extend(outer_codes)
    for ring in holes:
        hole_vertices, hole_codes = _ring_to_vertices_codes(ring)
        vertices.extend(hole_vertices)
        codes.extend(hole_codes)
    if not vertices:
        return
    ax.add_patch(
        patches.PathPatch(
            mpath.Path(np.asarray(vertices, dtype=np.float32), codes),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def draw_arcs(ax, arcs, *, color="#111111", linewidth=0.45, alpha=0.85):
    for arc in arcs:
        points = np.asarray(arc["points"], dtype=np.float32)
        if len(points) < 2:
            continue
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def draw_changed_arcs(ax, before: dict, after: dict) -> None:
    before_by_id = {int(arc["id"]): arc for arc in before["arcs"]}
    for arc in after["arcs"]:
        if not arc.get("regularized"):
            continue
        before_arc = before_by_id[int(arc["id"])]
        before_points = np.asarray(before_arc["points"], dtype=np.float32)
        after_points = np.asarray(arc["points"], dtype=np.float32)
        if len(before_points) >= 2:
            ax.plot(before_points[:, 0], before_points[:, 1], color="#c0392b", linewidth=1.0, alpha=0.55)
        if len(after_points) >= 2:
            ax.plot(after_points[:, 0], after_points[:, 1], color="#00a8e8", linewidth=1.25, alpha=0.95)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = load_json(args.before_json)
    after = load_json(args.after_json)
    split, stem = infer_split_stem(after, args)
    mask = np.asarray(Image.open(args.mask_root / split / "masks_id" / f"{stem}.png"), dtype=np.uint8)
    rgb = mask_to_rgb(mask)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    before_reg = before.get("arc_regularization", {})
    after_reg = after.get("arc_regularization", {})

    axes[0].imshow(rgb)
    axes[0].set_title(f"A. Original mask\n{split}/{stem}", fontsize=10)

    axes[1].imshow(rgb, alpha=0.18)
    draw_arcs(axes[1], before["arcs"], color="#111111", linewidth=0.45)
    axes[1].set_title("B. Before arc regularization", fontsize=10)

    axes[2].imshow(rgb, alpha=0.18)
    draw_arcs(axes[2], after["arcs"], color="#111111", linewidth=0.45)
    axes[2].set_title(
        f"C. After\naccepted={after_reg.get('accepted_count', 0)}, "
        f"vertices={after_reg.get('input_arc_vertex_count', 0)}->{after_reg.get('output_arc_vertex_count', 0)}",
        fontsize=10,
    )

    axes[3].imshow(rgb, alpha=0.16)
    for face in after["faces"]:
        draw_polygon(axes[3], face["outer"], face.get("holes", []), facecolor="none", edgecolor="black", linewidth=0.35, alpha=0.65)
    draw_changed_arcs(axes[3], before, after)
    validation = after["validation"]
    axes[3].set_title(
        f"D. Changed arcs\nvalid={validation['is_valid']}, iou={validation['union_iou']:.6f}",
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
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
