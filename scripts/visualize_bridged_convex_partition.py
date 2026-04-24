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
    parser = argparse.ArgumentParser(description="Visualize one bridged convex partition result.")
    parser.add_argument("--partition-json", type=Path, required=True)
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


def draw_polygon(ax, outer, holes, *, facecolor, edgecolor, linewidth=1.5, alpha=0.35, linestyle="-"):
    vertices = []
    codes = []
    outer_vertices, outer_codes = _ring_to_vertices_codes(outer)
    vertices.extend(outer_vertices)
    codes.extend(outer_codes)
    for ring in holes:
        hole_vertices, hole_codes = _ring_to_vertices_codes(ring)
        vertices.extend(hole_vertices)
        codes.extend(hole_codes)
    path = mpath.Path(np.asarray(vertices, dtype=np.float32), codes)
    ax.add_patch(
        patches.PathPatch(
            path,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )
    )


def annotate_centroid(ax, polygon_ring, text: str, *, color="black"):
    polygon = np.asarray(polygon_ring, dtype=np.float32)
    centroid = polygon.mean(axis=0)
    ax.text(centroid[0], centroid[1], text, ha="center", va="center", fontsize=8, color=color, weight="bold")


def selected_bridges(payload: dict) -> list[dict]:
    selected = payload.get("selected_bridge_set") or {}
    selected_ids = set(selected.get("bridge_ids", []))
    return [candidate for candidate in payload.get("bridge_candidates", []) if candidate["id"] in selected_ids]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.partition_json)
    mask = np.asarray(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    axes[0].imshow(mask_to_rgb(mask))
    draw_polygon(axes[0], payload["outer"], payload["holes"], facecolor="#00ff66", edgecolor="red", linewidth=2.0, alpha=0.16)
    bbox = payload["bbox"]
    axes[0].add_patch(
        patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor="red", facecolor="none")
    )
    axes[0].set_title(f"A. Approx geometry\nface={payload['face_id']}, holes={payload['hole_count']}", fontsize=10)

    axes[1].imshow(mask_to_rgb(mask), alpha=0.18)
    draw_polygon(axes[1], payload["outer"], payload["holes"], facecolor="none", edgecolor="black", linewidth=1.2, alpha=1.0)
    for bridge in selected_bridges(payload):
        p = np.asarray(bridge["p"], dtype=np.float32)
        q = np.asarray(bridge["q"], dtype=np.float32)
        axes[1].plot([p[0], q[0]], [p[1], q[1]], color="#e74c3c", linewidth=2.0)
        axes[1].scatter([p[0], q[0]], [p[1], q[1]], color="#e74c3c", s=18)
    axes[1].set_title(
        f"B. Selected bridges\npolicy={payload['bridge_policy']}, candidates={len(payload.get('bridge_candidates', []))}",
        fontsize=10,
    )

    axes[2].imshow(mask_to_rgb(mask), alpha=0.18)
    draw_polygon(axes[2], payload["outer"], payload["holes"], facecolor="none", edgecolor="black", linewidth=1.2, alpha=1.0)
    walk = payload.get("simple_polygon_boundary_walk", [])
    if len(walk) >= 2:
        walk_points = np.asarray(walk + [walk[0]], dtype=np.float32)
        axes[2].plot(walk_points[:, 0], walk_points[:, 1], color="#8e44ad", linewidth=1.5)
    axes[2].set_title(f"C. CGAL simple boundary\nvertices={payload['simple_polygon_vertex_count']}", fontsize=10)

    axes[3].imshow(mask_to_rgb(mask), alpha=0.18)
    draw_polygon(axes[3], payload["outer"], payload["holes"], facecolor="none", edgecolor="black", linewidth=1.2, alpha=1.0)
    for primitive in payload["primitives"]:
        color = "#2ecc71" if primitive["type"] == "quad" else ("#ff6b6b" if primitive["type"] == "triangle" else "#3498db")
        draw_polygon(axes[3], primitive["outer"], primitive["holes"], facecolor=color, edgecolor="black", linewidth=1.1, alpha=0.5)
        annotate_centroid(axes[3], primitive["outer"], str(primitive["id"]))
    backend = payload["backend_info"]["backend"]
    optimal = payload["backend_info"]["optimal"]
    axes[3].set_title(
        f"D. Convex pieces\nfinal={payload['final_primitive_count']}, iou={payload['validation']['iou']:.3f}, {backend}, optimal={optimal}",
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
