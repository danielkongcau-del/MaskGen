from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.geometry_render import (
    clamp_vertices,
    denormalize_local_vertices,
    load_json,
    remap_ids_to_values,
    render_geometry_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize face partitions and raster repair on one sample.")
    parser.add_argument(
        "--prediction-json",
        type=Path,
        default=Path("outputs/geometry_predictions_short/graphs/0000.json"),
        help="Predicted geometry json to visualize.",
    )
    parser.add_argument(
        "--dual-root",
        type=Path,
        default=Path("data/remote_256_dual"),
    )
    parser.add_argument(
        "--partition-root",
        type=Path,
        default=Path("data/remote_256_partition"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("data/remote_256_geometry"),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("data/remote_256"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/face_repair_demo.png"),
    )
    return parser.parse_args()


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
GAP_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)
OVERLAP_COLOR = np.asarray([255, 0, 255], dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray, unknown_value: int = -1) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = GAP_COLOR
    for label in range(PALETTE.shape[0]):
        rgb[mask == label] = PALETTE[label]
    if unknown_value >= 0:
        rgb[mask == unknown_value] = GAP_COLOR
    return rgb


def polygon_vertices_from_prediction(face_meta: Dict[str, object], face_pred: Dict[str, object], size: Tuple[int, int]) -> List[Tuple[float, float]]:
    vertex_count = int(face_pred.get("vertex_count_pred", 0))
    vertices_local = face_pred.get("vertices_local_pred", [])[:vertex_count]
    vertices = denormalize_local_vertices(face_meta, vertices_local)
    return clamp_vertices(vertices, size=size)


def hole_vertices_from_prediction(face_meta: Dict[str, object], face_pred: Dict[str, object], size: Tuple[int, int]) -> List[List[Tuple[float, float]]]:
    hole_count = int(face_pred.get("hole_count_pred", 0))
    hole_vertex_counts = face_pred.get("hole_vertex_counts_pred", [])
    hole_vertices_local = face_pred.get("hole_vertices_local_pred", [])
    holes: List[List[Tuple[float, float]]] = []
    for hole_index in range(min(hole_count, len(hole_vertex_counts), len(hole_vertices_local))):
        vertex_count = int(hole_vertex_counts[hole_index])
        if vertex_count < 3:
            continue
        vertices = denormalize_local_vertices(face_meta, hole_vertices_local[hole_index][:vertex_count])
        holes.append(clamp_vertices(vertices, size=size))
    return holes


def hole_vertices_from_geometry(face_meta: Dict[str, object], geometry_face: Dict[str, object], size: Tuple[int, int]) -> List[List[Tuple[float, float]]]:
    hole_count = int(geometry_face.get("hole_count", 0))
    hole_vertex_counts = geometry_face.get("hole_vertex_counts", [])
    hole_vertices_local = geometry_face.get("hole_vertices_local", [])
    holes: List[List[Tuple[float, float]]] = []
    for hole_index in range(min(hole_count, len(hole_vertex_counts), len(hole_vertices_local))):
        vertex_count = int(hole_vertex_counts[hole_index])
        if vertex_count < 3:
            continue
        vertices = denormalize_local_vertices(face_meta, hole_vertices_local[hole_index][:vertex_count])
        holes.append(clamp_vertices(vertices, size=size))
    return holes


def draw_polygon_with_holes(
    ax,
    outer: Sequence[Tuple[float, float]],
    holes: Sequence[Sequence[Tuple[float, float]]],
    *,
    facecolor,
    edgecolor: str = "black",
    linewidth: float = 0.5,
    alpha: float = 0.85,
) -> None:
    if len(outer) >= 3:
        ax.add_patch(
            MplPolygon(
                outer,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )
    for hole in holes:
        if len(hole) < 3:
            continue
        ax.add_patch(
            MplPolygon(
                hole,
                closed=True,
                facecolor="white",
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=1.0,
            )
        )


def build_raw_status_map(
    dual_graph: Dict[str, object],
    prediction: Dict[str, object],
    size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    height, width = size
    label_map = np.full((height, width), -1, dtype=np.int16)
    cover_count = np.zeros((height, width), dtype=np.int16)
    face_metas = {int(face["id"]): face for face in dual_graph["faces"]}

    from partition_gen.geometry_render import polygon_pixels, repair_polygon

    for face_pred in prediction["faces"]:
        if int(face_pred.get("support_pred", 0)) <= 0:
            continue
        face_id = int(face_pred["id"])
        if face_id not in face_metas:
            continue
        face_meta = face_metas[face_id]
        vertices = polygon_vertices_from_prediction(face_meta, face_pred, size=size)
        holes = hole_vertices_from_prediction(face_meta, face_pred, size=size)
        polygon = repair_polygon(vertices, holes=holes)
        if polygon is None:
            continue
        xs, ys, inside = polygon_pixels(polygon, size=size)
        if inside.size == 0:
            continue
        region = label_map[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        counts = cover_count[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        region[inside] = int(face_metas[face_id]["label"])
        counts[inside] += 1

    status = np.full((height, width), -1, dtype=np.int16)
    single = cover_count == 1
    gap = cover_count == 0
    overlap = cover_count > 1
    status[single] = label_map[single]
    status[gap] = -1
    status[overlap] = -2
    summary = {
        "gap_pixels": int(gap.sum()),
        "overlap_pixels": int(overlap.sum()),
        "single_pixels": int(single.sum()),
    }
    return status, summary


def status_to_rgb(status: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*status.shape, 3), dtype=np.uint8)
    rgb[status == -1] = GAP_COLOR
    rgb[status == -2] = OVERLAP_COLOR
    for label in range(PALETTE.shape[0]):
        rgb[status == label] = PALETTE[label]
    return rgb


def draw_partition_edges(ax, partition_graph: Dict[str, object], line_width: float = 0.6) -> None:
    vertices = [tuple(vertex) for vertex in partition_graph["vertices"]]
    for edge in partition_graph["edges"]:
        start, end = edge["vertices"]
        x0, y0 = vertices[start]
        x1, y1 = vertices[end]
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=line_width)


def annotate_face_ids(ax, dual_graph: Dict[str, object], max_labels: int = 12) -> None:
    faces = sorted(dual_graph["faces"], key=lambda face: float(face["area"]), reverse=True)[:max_labels]
    for face in faces:
        x, y = face["centroid"]
        ax.text(
            x,
            y,
            str(face["id"]),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.4, "linewidth": 0},
        )


def draw_geometry_faces(ax, dual_graph: Dict[str, object], geometry_graph: Dict[str, object], size: Tuple[int, int]) -> None:
    ax.imshow(np.ones((*size, 3), dtype=np.uint8) * 255)
    dual_faces = {int(face["id"]): face for face in dual_graph["faces"]}
    for face in geometry_graph["faces"]:
        if not bool(face["supported"]):
            meta = dual_faces[int(face["id"])]
            ax.scatter([meta["centroid"][0]], [meta["centroid"][1]], c="red", s=18, marker="x")
            continue
        meta = dual_faces[int(face["id"])]
        vertices = denormalize_local_vertices(meta, face["vertices_local"])
        holes = hole_vertices_from_geometry(meta, face, size=size)
        draw_polygon_with_holes(
            ax,
            vertices,
            holes,
            facecolor=PALETTE[int(meta["label"])] / 255.0,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )


def draw_predicted_polygons(ax, dual_graph: Dict[str, object], prediction: Dict[str, object], size: Tuple[int, int]) -> None:
    ax.imshow(np.ones((*size, 3), dtype=np.uint8) * 255)
    dual_faces = {int(face["id"]): face for face in dual_graph["faces"]}
    for face_pred in prediction["faces"]:
        if int(face_pred.get("support_pred", 0)) <= 0:
            continue
        meta = dual_faces[int(face_pred["id"])]
        vertices = polygon_vertices_from_prediction(meta, face_pred, size=size)
        holes = hole_vertices_from_prediction(meta, face_pred, size=size)
        if len(vertices) < 3:
            continue
        draw_polygon_with_holes(
            ax,
            vertices,
            holes,
            facecolor=PALETTE[int(meta["label"])] / 255.0,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.55,
        )


def finalize_axis(ax, title: str, size: Tuple[int, int]) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, size[1])
    ax.set_ylim(size[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    prediction = load_json(args.prediction_json)
    dual_graph = load_json(Path(prediction["source_dual_graph"]))
    partition_graph = load_json(args.partition_root / str(dual_graph["source_partition_graph"]))
    geometry_graph = load_json(args.geometry_root / str(prediction["source_dual_graph"]).replace("data/remote_256_dual", "").lstrip("/\\"))
    gt_mask_path = args.mask_root / str(dual_graph["source_mask"])
    gt_mask = np.array(Image.open(gt_mask_path))
    size = tuple(int(value) for value in dual_graph["size"])

    rendered_mask, render_meta = render_geometry_prediction(
        dual_graph=dual_graph,
        prediction=prediction,
        size=size,
    )
    raw_status, raw_meta = build_raw_status_map(dual_graph, prediction, size=size)
    pixel_acc = float((rendered_mask == gt_mask).mean())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].imshow(mask_to_rgb(gt_mask))
    finalize_axis(axes[0, 0], "A. Ground Truth Mask", size)

    axes[0, 1].imshow(mask_to_rgb(gt_mask))
    draw_partition_edges(axes[0, 1], partition_graph)
    annotate_face_ids(axes[0, 1], dual_graph)
    finalize_axis(axes[0, 1], "B. Faces = closed regions", size)

    draw_geometry_faces(axes[0, 2], dual_graph, geometry_graph, size)
    unsupported = sum(1 for face in geometry_graph["faces"] if not bool(face["supported"]))
    finalize_axis(axes[0, 2], f"C. Simplified face polygons\n(red x = unsupported holes: {unsupported})", size)

    draw_predicted_polygons(axes[1, 0], dual_graph, prediction, size)
    finalize_axis(axes[1, 0], "D. Predicted polygons", size)

    axes[1, 1].imshow(status_to_rgb(raw_status))
    finalize_axis(
        axes[1, 1],
        f"E. Raw raster\nblack=gaps {raw_meta['gap_pixels']}, magenta=overlap {raw_meta['overlap_pixels']}",
        size,
    )

    axes[1, 2].imshow(mask_to_rgb(rendered_mask))
    finalize_axis(
        axes[1, 2],
        f"F. After raster repair\npixel acc vs GT = {pixel_acc:.3f}",
        size,
    )

    fig.suptitle(
        "Face -> polygon -> raw raster -> repaired mask\n"
        f"sample: {args.prediction_json.as_posix()}",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output.as_posix()))


if __name__ == "__main__":
    main()
