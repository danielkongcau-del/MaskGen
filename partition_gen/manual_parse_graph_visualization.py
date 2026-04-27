from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence


LABEL_PALETTE = [
    "#d9d9d9",
    "#e76f51",
    "#2a9d8f",
    "#e9c46a",
    "#457b9d",
    "#9b5de5",
    "#f15bb5",
    "#00bbf9",
]

ROLE_EDGE_COLORS = {
    "support_region": "#264653",
    "divider_region": "#111111",
    "insert_object": "#8a1c7c",
    "residual_region": "#6c757d",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def local_to_world(point: Sequence[float], frame: dict) -> List[float]:
    origin = frame.get("origin", [0.0, 0.0])
    cx = float(origin[0])
    cy = float(origin[1])
    scale = max(float(frame.get("scale", 1.0)), 1e-8)
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x = float(point[0]) * scale
    y = float(point[1]) * scale
    return [cx + x * cos_theta - y * sin_theta, cy + x * sin_theta + y * cos_theta]


def _ring_to_world(ring: Sequence[Sequence[float]], frame: dict) -> List[List[float]]:
    return [local_to_world(point, frame) for point in ring]


def polygon_world_rings(node: dict) -> Iterable[tuple[List[List[float]], List[List[List[float]]]]]:
    geometry = node.get("geometry", {}) or {}
    frame = node.get("frame", {}) or {}
    polygons = geometry.get("polygons_local") or [
        {"outer_local": geometry.get("outer_local", []), "holes_local": geometry.get("holes_local", [])}
    ]
    for polygon in polygons:
        outer = _ring_to_world(polygon.get("outer_local", []) or [], frame)
        holes = [_ring_to_world(ring, frame) for ring in polygon.get("holes_local", []) or []]
        if len(outer) >= 3:
            yield outer, [hole for hole in holes if len(hole) >= 3]


def render_manual_parse_graph_target(
    target: dict,
    output_png: Path,
    *,
    dpi: int = 150,
    alpha: float = 0.55,
    annotate: bool = False,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    size = target.get("size", [256, 256])
    width = int(size[0]) if size else 256
    height = int(size[1]) if len(size) > 1 else width

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_facecolor("#ffffff")
    rendered_polygon_count = 0
    rendered_node_count = 0

    for node in nodes:
        if not bool(node.get("renderable", True)) or bool(node.get("is_reference_only", False)):
            continue
        if str(node.get("geometry_model", "none")) != "polygon_code":
            continue
        rings = list(polygon_world_rings(node))
        if not rings:
            continue
        rendered_node_count += 1
        label = int(node.get("label", 0))
        facecolor = LABEL_PALETTE[label % len(LABEL_PALETTE)]
        edgecolor = ROLE_EDGE_COLORS.get(str(node.get("role", "")), "#222222")
        for outer, holes in rings:
            rendered_polygon_count += 1
            ax.add_patch(
                patches.Polygon(
                    np.asarray(outer, dtype=np.float32),
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.1,
                    alpha=float(alpha),
                )
            )
            for hole in holes:
                ax.add_patch(
                    patches.Polygon(
                        np.asarray(hole, dtype=np.float32),
                        closed=True,
                        facecolor="#ffffff",
                        edgecolor=edgecolor,
                        linewidth=0.7,
                        alpha=1.0,
                    )
                )
            if annotate:
                points = np.asarray(outer, dtype=np.float32)
                centroid = points.mean(axis=0)
                ax.text(
                    float(centroid[0]),
                    float(centroid[1]),
                    str(node.get("id", "")),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#111111",
                )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output_png, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "output_png": str(output_png.as_posix()),
        "node_count": int(len(nodes)),
        "rendered_node_count": int(rendered_node_count),
        "rendered_polygon_count": int(rendered_polygon_count),
    }
