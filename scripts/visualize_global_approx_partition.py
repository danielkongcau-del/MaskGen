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
    parser = argparse.ArgumentParser(description="Visualize one full-image shared-arc approximation.")
    parser.add_argument("--global-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--stem", type=str, default=None)
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


def draw_arcs(ax, arcs, *, color_by_shared: bool = True, linewidth: float = 0.8):
    for arc in arcs:
        points = np.asarray(arc["points"], dtype=np.float32)
        if len(points) < 2:
            continue
        incident_faces = [face for face in arc["incident_faces"] if int(face) >= 0]
        color = "#111111"
        if color_by_shared:
            color = "#e74c3c" if len(incident_faces) == 2 else "#2c3e50"
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=0.9)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.global_json)
    split, stem = infer_split_stem(payload, args)
    mask = np.asarray(Image.open(args.mask_root / split / "masks_id" / f"{stem}.png"), dtype=np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    rgb = mask_to_rgb(mask)
    validation = payload["validation"]

    axes[0].imshow(rgb)
    axes[0].set_title(f"A. Original mask\n{split}/{stem}", fontsize=10)

    axes[1].imshow(rgb, alpha=0.22)
    draw_arcs(axes[1], payload["arcs"], linewidth=0.8)
    axes[1].set_title(
        f"B. Shared maximal arcs\narcs={validation['arc_count']}, shared={validation['shared_arc_count']}",
        fontsize=10,
    )

    axes[2].imshow(np.full_like(rgb, 255))
    for face in payload["faces"]:
        color = PALETTE[int(face["label"]) % PALETTE.shape[0]] / 255.0
        draw_polygon(axes[2], face["outer"], face["holes"], facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.7)
    axes[2].set_title(
        f"C. Reconstructed faces\nvalid={validation['is_valid']}, iou={validation['union_iou']:.6f}",
        fontsize=10,
    )

    axes[3].imshow(rgb, alpha=0.18)
    for face in payload["faces"]:
        draw_polygon(axes[3], face["outer"], face["holes"], facecolor="none", edgecolor="black", linewidth=0.45, alpha=0.85)
    draw_arcs(axes[3], payload["arcs"], color_by_shared=False, linewidth=0.45)
    axes[3].set_title(
        f"D. Approx overlay\naccepted={payload.get('reconciliation', {}).get('accepted_count', 0)}, "
        f"overlap={validation['overlap_area']:.4f}",
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
