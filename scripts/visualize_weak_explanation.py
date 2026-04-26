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
    parser = argparse.ArgumentParser(description="Visualize one weak convex-face-atom explanation.")
    parser.add_argument("--weak-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, default=None)
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


def draw_polygon(ax, outer, holes=None, *, facecolor, edgecolor, linewidth=0.6, alpha=0.55):
    holes = holes or []
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


def draw_arcs(ax, arcs, *, linewidth: float = 0.5):
    for arc in arcs:
        points = np.asarray(arc.get("points", []), dtype=np.float32)
        if len(points) < 2:
            continue
        color = "#d62728" if arc.get("is_shared") else "#333333"
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=0.75)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def infer_split_stem(evidence: dict, args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.split and args.stem:
        return args.split, args.stem
    source_mask = str(evidence.get("source_mask") or "")
    if source_mask:
        path = Path(source_mask)
        if len(path.parts) >= 2:
            return path.parts[0], path.stem
    source_graph = str(evidence.get("source_partition_graph") or "")
    if source_graph:
        path = Path(source_graph)
        if len(path.parts) >= 3:
            return path.parts[-3], path.stem
    return None, None


def load_mask_rgb(evidence: dict, args: argparse.Namespace, size: list[int]) -> np.ndarray:
    split, stem = infer_split_stem(evidence, args)
    if split and stem:
        for subdir in ("masks_id", "masks"):
            path = args.mask_root / split / subdir / f"{stem}.png"
            if path.exists():
                return mask_to_rgb(np.asarray(Image.open(path), dtype=np.uint8))
    height, width = [int(value) for value in size]
    return np.full((height, width, 3), 255, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    weak = load_json(args.weak_json)
    evidence_path = args.evidence_json
    if evidence_path is None:
        source = weak.get("source_evidence")
        evidence_path = Path(source) if source else None
    if evidence_path is None or not evidence_path.exists():
        raise ValueError("Pass --evidence-json; weak explanation source_evidence is missing or not a local file.")
    evidence = load_json(evidence_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    size = evidence.get("size", weak.get("generator_target", {}).get("size", [256, 256]))
    rgb = load_mask_rgb(evidence, args, size)
    face_by_id = {int(face["id"]): face for face in evidence.get("faces", [])}
    atom_count_by_face = {
        int(face_id): int(count)
        for face_id, count in weak.get("diagnostics", {}).get("atom_count_by_source_face", {}).items()
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("A. Source mask", fontsize=10)

    axes[1].imshow(rgb, alpha=0.18)
    draw_arcs(axes[1], evidence.get("arcs", []), linewidth=0.55)
    axes[1].set_title(f"B. Shared arcs\narcs={len(evidence.get('arcs', []))}", fontsize=10)

    axes[2].imshow(np.full_like(rgb, 255))
    for face in evidence.get("faces", []):
        label = int(face["label"])
        color = PALETTE[label % PALETTE.shape[0]] / 255.0
        geometry = face["geometry"]
        draw_polygon(axes[2], geometry["outer"], geometry["holes"], facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.6)
    axes[2].set_title(f"C. Semantic faces\nfaces={len(face_by_id)}", fontsize=10)

    axes[3].imshow(np.full_like(rgb, 255))
    for face in evidence.get("faces", []):
        label = int(face["label"])
        face_id = int(face["id"])
        face_color = PALETTE[label % PALETTE.shape[0]] / 255.0
        for atom in face.get("convex_partition", {}).get("atoms", []):
            draw_polygon(axes[3], atom.get("outer", []), [], facecolor=face_color, edgecolor="#1f1f1f", linewidth=0.35, alpha=0.42)
        centroid = face.get("features", {}).get("centroid", None)
        if centroid is not None:
            axes[3].text(centroid[0], centroid[1], str(atom_count_by_face.get(face_id, 0)), fontsize=4, ha="center", va="center")
    diagnostics = weak.get("diagnostics", {})
    axes[3].set_title(
        "D. Convex atoms per face\n"
        f"atoms={diagnostics.get('convex_atom_count', 0)}, residual_faces={diagnostics.get('residual_face_count', 0)}",
        fontsize=10,
    )

    height, width = rgb.shape[:2]
    for axis in axes:
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
