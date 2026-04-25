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
    parser = argparse.ArgumentParser(description="Visualize tiny closed face absorption.")
    parser.add_argument("--before-json", type=Path, required=True)
    parser.add_argument("--after-json", type=Path, required=True)
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


def draw_polygon(ax, outer, holes, *, facecolor, edgecolor, linewidth=0.55, alpha=0.72):
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


def draw_faces(ax, faces, *, fill=True, linewidth=0.45, alpha=0.72):
    for face in faces:
        color = PALETTE[int(face["label"]) % PALETTE.shape[0]] / 255.0
        draw_polygon(
            ax,
            face["outer"],
            face.get("holes", []),
            facecolor=color if fill else "none",
            edgecolor="black",
            linewidth=linewidth,
            alpha=alpha if fill else 0.85,
        )


def draw_absorbed_before(ax, before, after):
    absorbed = after.get("micro_face_absorption", {}).get("absorbed", [])
    before_by_id = {int(face["id"]): face for face in before.get("faces", [])}
    for item in absorbed:
        face = before_by_id.get(int(item["face_id"]))
        if face is None:
            continue
        draw_polygon(
            ax,
            face["outer"],
            face.get("holes", []),
            facecolor="none",
            edgecolor="#e74c3c",
            linewidth=1.25,
            alpha=1.0,
        )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = load_json(args.before_json)
    after = load_json(args.after_json)
    mask = np.asarray(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)
    rgb = mask_to_rgb(mask)
    summary = after.get("micro_face_absorption", {})
    validation = after.get("validation", {})

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"A. Original mask\n{args.split}/{args.stem}", fontsize=10)

    axes[1].imshow(np.full_like(rgb, 255))
    draw_faces(axes[1], before.get("faces", []), fill=True)
    axes[1].set_title(f"B. Before absorption\nfaces={len(before.get('faces', []))}", fontsize=10)

    axes[2].imshow(np.full_like(rgb, 255))
    draw_faces(axes[2], after.get("faces", []), fill=True)
    axes[2].set_title(
        f"C. After absorption\nabsorbed={summary.get('absorbed_count', 0)}, "
        f"micro={summary.get('micro_candidate_count', 0)}, islands={summary.get('small_island_candidate_count', 0)}, "
        f"faces={summary.get('input_face_count', 0)}->{summary.get('output_face_count', 0)}",
        fontsize=10,
    )

    axes[3].imshow(rgb, alpha=0.18)
    draw_faces(axes[3], after.get("faces", []), fill=False, linewidth=0.45, alpha=0.9)
    draw_absorbed_before(axes[3], before, after)
    axes[3].set_title(
        f"D. Removed micro faces\nvalid={validation.get('is_valid')}, iou={validation.get('union_iou', 0):.6f}",
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
