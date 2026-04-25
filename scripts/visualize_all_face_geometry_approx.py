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
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json  # noqa: E402
from partition_gen.geometry_approximator import (  # noqa: E402
    GeometryApproximationConfig,
    approximate_face_from_partition_graph,
)


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
    parser = argparse.ArgumentParser(description="Visualize independent per-face geometry approximations for one mask.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--area-epsilon", type=float, default=1e-3)
    parser.add_argument("--edge-linewidth", type=float, default=0.35)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    return rgb


def _ring_to_vertices_codes(ring: list[list[float]]) -> tuple[list[list[float]], list[int]]:
    points = np.asarray(ring, dtype=np.float32)
    if len(points) == 0:
        return [], []
    vertices = points.tolist() + [points[0].tolist()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(points) - 1) + [mpath.Path.CLOSEPOLY]
    return vertices, codes


def draw_polygon(
    ax,
    geometry: dict,
    *,
    facecolor,
    edgecolor,
    linewidth: float = 0.5,
    alpha: float = 0.6,
) -> None:
    outer = geometry.get("outer", [])
    if not outer:
        return
    vertices: list[list[float]] = []
    codes: list[int] = []
    outer_vertices, outer_codes = _ring_to_vertices_codes(outer)
    vertices.extend(outer_vertices)
    codes.extend(outer_codes)
    for ring in geometry.get("holes", []):
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


def polygon_from_geometry_payload(geometry: dict) -> Polygon:
    outer = geometry.get("outer", [])
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer, geometry.get("holes", [])), sign=1.0)
    except ValueError:
        return Polygon()


def build_payload(graph_data: dict, graph_path: Path, *, config: GeometryApproximationConfig) -> dict:
    faces = []
    approx_polygons = []
    original_polygons = []
    for face in graph_data["faces"]:
        payload = approximate_face_from_partition_graph(graph_data, face, config=config)
        faces.append(payload)
        approx = polygon_from_geometry_payload(payload["approx_geometry"])
        original = polygon_from_geometry_payload(payload["original_geometry"])
        if not approx.is_empty:
            approx_polygons.append(approx)
        if not original.is_empty:
            original_polygons.append(original)

    approx_area_sum = float(sum(poly.area for poly in approx_polygons))
    approx_union = unary_union(approx_polygons) if approx_polygons else None
    original_union = unary_union(original_polygons) if original_polygons else None
    approx_union_area = float(approx_union.area) if approx_union is not None and not approx_union.is_empty else 0.0
    original_union_area = float(original_union.area) if original_union is not None and not original_union.is_empty else 0.0
    overlap_area = max(0.0, approx_area_sum - approx_union_area)
    union_iou = 0.0
    if approx_union is not None and original_union is not None and not approx_union.is_empty and not original_union.is_empty:
        intersection = float(approx_union.intersection(original_union).area)
        union = float(approx_union.union(original_union).area)
        union_iou = intersection / union if union > 0 else 0.0

    return {
        "format": "all_face_independent_geometry_approx_v1",
        "source_partition_graph": str(graph_path.as_posix()),
        "source_mask": graph_data.get("source_mask"),
        "size": graph_data["size"],
        "config": {
            "simplify_tolerance": float(config.simplify_tolerance),
            "area_epsilon": float(config.area_epsilon),
            "trim_collinear_eps": float(config.trim_collinear_eps),
        },
        "summary": {
            "face_count": int(len(faces)),
            "approx_area_sum": float(approx_area_sum),
            "approx_union_area": float(approx_union_area),
            "original_union_area": float(original_union_area),
            "overlap_area": float(overlap_area),
            "union_iou": float(union_iou),
            "mean_face_iou": float(np.mean([face["approx_iou"] for face in faces])) if faces else 0.0,
            "mean_approx_vertex_count": float(np.mean([face["approx_vertex_count"] for face in faces])) if faces else 0.0,
            "total_base_primitive_count": int(sum(face["base_primitive_count"] for face in faces)),
            "total_approx_vertex_count": int(sum(face["approx_vertex_count"] for face in faces)),
        },
        "faces": faces,
    }


def draw_all_faces(ax, faces: list[dict], *, fill: bool, linewidth: float) -> None:
    for face in faces:
        color = PALETTE[int(face["label"]) % PALETTE.shape[0]] / 255.0
        draw_polygon(
            ax,
            face["approx_geometry"],
            facecolor=color if fill else "none",
            edgecolor="black",
            linewidth=linewidth,
            alpha=0.72 if fill else 0.9,
        )


def visualize(payload: dict, mask: np.ndarray, output: Path, *, edge_linewidth: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = mask_to_rgb(mask)
    faces = payload["faces"]
    summary = payload["summary"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("A. Original mask", fontsize=10)

    axes[1].imshow(np.full_like(rgb, 255))
    draw_all_faces(axes[1], faces, fill=True, linewidth=edge_linewidth)
    axes[1].set_title(
        f"B. Independent face approximations\nfaces={summary['face_count']}, vertices={summary['total_approx_vertex_count']}",
        fontsize=10,
    )

    axes[2].imshow(rgb, alpha=0.18)
    draw_all_faces(axes[2], faces, fill=False, linewidth=edge_linewidth)
    axes[2].set_title(
        f"C. Approx overlay\nmean_face_iou={summary['mean_face_iou']:.4f}",
        fontsize=10,
    )

    axes[3].imshow(np.full_like(rgb, 255))
    draw_all_faces(axes[3], faces, fill=True, linewidth=edge_linewidth)
    axes[3].imshow(rgb, alpha=0.18)
    axes[3].set_title(
        f"D. Diagnostics\nunion_iou={summary['union_iou']:.6f}, overlap={summary['overlap_area']:.4f}",
        fontsize=10,
    )

    for axis in axes:
        axis.set_xlim(0, mask.shape[1])
        axis.set_ylim(mask.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    graph_path = args.partition_root / args.split / "graphs" / f"{args.stem}.json"
    graph_data = load_json(graph_path)
    payload = build_payload(
        graph_data,
        graph_path,
        config=GeometryApproximationConfig(
            simplify_tolerance=float(args.simplify_tolerance),
            area_epsilon=float(args.area_epsilon),
        ),
    )
    mask = np.asarray(Image.open(args.mask_root / args.split / "masks_id" / f"{args.stem}.png"), dtype=np.uint8)
    visualize(payload, mask, args.output, edge_linewidth=float(args.edge_linewidth))
    if args.json_output is not None:
        dump_json(args.json_output, payload)
    summary = payload["summary"]
    print(
        f"visualized all-face geometry approx {args.split}/{args.stem}: "
        f"faces={summary['face_count']}, vertices={summary['total_approx_vertex_count']}, "
        f"mean_iou={summary['mean_face_iou']:.6f}, union_iou={summary['union_iou']:.6f}, "
        f"overlap={summary['overlap_area']:.6f}"
    )


if __name__ == "__main__":
    main()
