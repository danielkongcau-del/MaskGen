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
    parser = argparse.ArgumentParser(description="Visualize composite-group patch acceptance scores.")
    parser.add_argument("--primitive-root", type=Path, default=Path("data/remote_256_primitives_debug"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, default="10")
    parser.add_argument("--face-id", type=int, default=26)
    parser.add_argument("--group-id", type=int, default=1)
    parser.add_argument("--step", type=int, default=0, help="Patch history step index.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/composite_patch_scores_val10_face26_group1.png"),
    )
    return parser.parse_args()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def draw_ring(ax, ring, *, facecolor, edgecolor, linewidth=1.5, alpha=0.45, linestyle="-", zorder=2):
    polygon = np.asarray(ring, dtype=np.float32)
    patch = patches.Polygon(
        polygon,
        closed=True,
        fill=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return polygon


def _ring_to_vertices_codes(ring, *, reverse: bool = False):
    points = np.asarray(ring, dtype=np.float32)
    if reverse:
        points = points[::-1]
    if len(points) == 0:
        return [], []
    vertices = points.tolist() + [points[0].tolist()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(points) - 1) + [mpath.Path.CLOSEPOLY]
    return vertices, codes


def draw_component(ax, component, *, facecolor, edgecolor, linewidth=1.5, alpha=0.45, linestyle="-", zorder=2):
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
        return None
    path = mpath.Path(np.asarray(vertices, dtype=np.float32), codes)
    patch = patches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return np.asarray(component["outer"], dtype=np.float32)


def summarize_candidate(prefix: str, row: dict) -> str:
    return (
        f"{prefix} atom {row['atom_id']}: "
        f"score={row['total_score']:.3f}, "
        f"area+={row['area_gain_term']:.3f}, "
        f"edge-={row['edge_cost_term']:.3f}, "
        f"conn+={row['connectivity_term']:.3f}, "
        f"fit+={row['fit_gain_term']:.3f}, "
        f"hole-={row.get('hole_invasion_term', 0.0):.3f}, "
        f"holeloss-={row.get('hole_loss_term', 0.0):.3f}"
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    graph = load_json(args.primitive_root / args.split / "graphs" / f"{args.stem}.json")
    mask = np.array(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)
    face = next(item for item in graph["faces"] if int(item["face_id"]) == args.face_id)
    group = next(item for item in face["composite_groups"]["groups"] if int(item["id"]) == args.group_id)
    history = group["patch_history"]
    if not history:
        raise ValueError(f"group {args.group_id} has no patch history")
    step = history[min(max(args.step, 0), len(history) - 1)]

    atom_by_id = {int(atom["id"]): atom for atom in face["primitives"]}
    seed_ids = set(int(atom_id) for atom_id in group["seed_atom_ids"])
    added_ids = set(int(atom_id) for atom_id in group["added_atom_ids"])
    accepted = step.get("accepted")
    accepted_id = int(accepted["atom_id"]) if accepted else None
    top_rejected = step.get("top_rejected", [])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()

    for ax in axes[:3]:
        ax.imshow(mask_to_rgb(mask))
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    bbox = face["bbox"]
    bbox_patch = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(bbox_patch)
    components = group.get("components")
    if components:
        for component in components:
            draw_component(axes[0], component, facecolor="#2ecc71", edgecolor="black", linewidth=2.0, alpha=0.35)
    else:
        for ring in group["polygons"]:
            draw_ring(axes[0], ring, facecolor="#2ecc71", edgecolor="black", linewidth=2.0, alpha=0.35)
    axes[0].set_title(f"A. Final composite group {group['id']}", fontsize=11)

    # Seed / accepted / rejected atom layout
    for atom_id in group["atom_ids"]:
        atom = atom_by_id.get(int(atom_id))
        if atom is None:
            continue
        if atom_id in seed_ids:
            color = "#4C78A8"
        elif atom_id in added_ids:
            color = "#2CA02C"
        else:
            color = "#BBBBBB"
        polygon = draw_ring(axes[1], atom["vertices"], facecolor=color, edgecolor="black", alpha=0.45)
        centroid = polygon.mean(axis=0)
        axes[1].text(centroid[0], centroid[1], str(atom_id), ha="center", va="center", fontsize=8, weight="bold")

    for row in top_rejected:
        atom = atom_by_id.get(int(row["atom_id"]))
        if atom is None:
            continue
        polygon = draw_ring(
            axes[1],
            atom["vertices"],
            facecolor="none",
            edgecolor="#D62728",
            linewidth=2.0,
            alpha=1.0,
            linestyle="--",
            zorder=5,
        )
        centroid = polygon.mean(axis=0)
        axes[1].text(
            centroid[0],
            centroid[1],
            f"{row['atom_id']}\n{row['total_score']:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#D62728",
            weight="bold",
        )
    axes[1].set_title("B. Seed atoms / accepted patch / top rejected", fontsize=11)

    # Accepted patch focus
    if accepted_id is not None:
        if components:
            for component in components:
                draw_component(axes[2], component, facecolor="#2ecc71", edgecolor="black", linewidth=1.8, alpha=0.25)
        else:
            for ring in group["polygons"]:
                draw_ring(axes[2], ring, facecolor="#2ecc71", edgecolor="black", linewidth=1.8, alpha=0.25)
        atom = atom_by_id.get(int(accepted_id))
        if atom is not None:
            polygon = draw_ring(
                axes[2],
                atom["vertices"],
                facecolor="#2CA02C",
                edgecolor="black",
                linewidth=2.0,
                alpha=0.75,
                zorder=6,
            )
            centroid = polygon.mean(axis=0)
            axes[2].text(
                centroid[0],
                centroid[1],
                f"+{accepted_id}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                weight="bold",
            )
        axes[2].set_title(f"C. Accepted patch atom {accepted_id}", fontsize=11)
    else:
        axes[2].set_title("C. No patch accepted at this step", fontsize=11)

    axes[3].axis("off")
    lines = [
        f"group {group['id']} | seed atoms={group['seed_atom_ids']} | added atoms={group['added_atom_ids']}",
        f"component_count={group.get('component_count')} | hole_count={group.get('hole_count', 0)}",
        f"step status={step['status']} | current atoms={step['current_atom_ids']}",
    ]
    if accepted:
        lines.append("")
        lines.append(summarize_candidate("Accepted", accepted))
    if top_rejected:
        lines.append("")
        lines.append("Top rejected:")
        for row in top_rejected[:5]:
            lines.append(summarize_candidate("-", row))
    axes[3].text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output.as_posix()))


if __name__ == "__main__":
    main()
