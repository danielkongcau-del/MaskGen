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

ROLE_COLORS = {
    "support_region": "#8fd3b0",
    "divider_region": "#4fb3bf",
    "insert_object": "#f6c85f",
    "residual_region": "#ef8a62",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize label-pair relation analysis.")
    parser.add_argument("--pairwise-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ring_to_vertices_codes(ring):
    points = np.asarray(ring, dtype=np.float32)
    if len(points) == 0:
        return [], []
    vertices = points.tolist() + [points[0].tolist()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(points) - 1) + [mpath.Path.CLOSEPOLY]
    return vertices, codes


def draw_polygon(ax, outer, holes, *, facecolor, edgecolor, linewidth=0.7, alpha=0.65):
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


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def load_mask_rgb(args: argparse.Namespace, evidence: dict) -> np.ndarray:
    split = args.split
    stem = args.stem
    if not split or not stem:
        source_mask = str(evidence.get("source_mask") or "")
        if source_mask:
            path = Path(source_mask)
            split = split or (path.parts[0] if path.parts else None)
            stem = stem or path.stem
    if split and stem:
        for subdir in ("masks_id", "masks"):
            path = args.mask_root / split / subdir / f"{stem}.png"
            if path.exists():
                return mask_to_rgb(np.asarray(Image.open(path), dtype=np.uint8))
    height, width = evidence.get("size", [256, 256])
    return np.full((height, width, 3), 255, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    pairwise = load_json(args.pairwise_json)
    evidence = load_json(args.evidence_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgb = load_mask_rgb(args, evidence)

    preferred = {int(label): role for label, role in pairwise.get("preferred_role_by_label", {}).items()}
    pairs = pairwise.get("pairs", [])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("A. Source mask", fontsize=10)

    axes[1].imshow(np.full_like(rgb, 255))
    for face in evidence.get("faces", []):
        label = int(face["label"])
        role = preferred.get(label, "unassigned")
        color = ROLE_COLORS.get(role, "#dddddd")
        geom = face["geometry"]
        draw_polygon(axes[1], geom["outer"], geom["holes"], facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.7)
    axes[1].set_title("B. Preferred role by label pair voting", fontsize=10)

    axes[2].axis("off")
    lines = ["Selected label-pair explanations:"]
    for pair in pairs[:18]:
        selected = pair["selected"]
        lines.append(
            f"{pair['labels']}: {selected['template']} / {selected['fill_policy']} "
            f"{selected['roles']} cost={selected['cost']:.1f}"
        )
    if len(pairs) > 18:
        lines.append(f"... {len(pairs) - 18} more")
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")
    axes[2].set_title("C. Pairwise decisions", fontsize=10)

    height, width = rgb.shape[:2]
    for axis in axes[:2]:
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
