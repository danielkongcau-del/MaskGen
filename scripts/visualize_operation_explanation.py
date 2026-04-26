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
    "support_region": "#2ca02c",
    "divider_region": "#17becf",
    "insert_object": "#ffbf00",
    "insert_object_group": "#9467bd",
    "residual_region": "#7f7f7f",
}

ROLE_LABELS = {
    "support_region": "support",
    "divider_region": "divider",
    "insert_object": "insert",
    "insert_object_group": "insert group",
    "residual_region": "residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an operation-level explanation.")
    parser.add_argument("--operation-json", type=Path, required=True)
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


def draw_polygon(ax, outer, holes=None, *, facecolor, edgecolor="black", linewidth=0.5, alpha=0.55):
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


def _evidence_path(operation: dict, args: argparse.Namespace) -> Path | None:
    if args.evidence_json is not None:
        return args.evidence_json
    source = operation.get("source_evidence")
    if source:
        path = Path(str(source))
        if path.exists():
            return path
    return None


def _face_color(label: int):
    return PALETTE[label % PALETTE.shape[0]] / 255.0


def add_role_legend(ax, roles):
    handles = []
    labels = []
    for role in sorted(set(roles)):
        if role not in ROLE_COLORS:
            continue
        handles.append(patches.Patch(facecolor=ROLE_COLORS[role], edgecolor="black", alpha=0.45))
        labels.append(ROLE_LABELS.get(role, role))
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=6, framealpha=0.9)


def main() -> None:
    args = parse_args()
    operation = load_json(args.operation_json)
    evidence_path = _evidence_path(operation, args)
    if evidence_path is None:
        raise ValueError("Pass --evidence-json; operation source_evidence is missing or not a local file.")
    evidence = load_json(evidence_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    size = evidence.get("size", operation.get("generator_target", {}).get("size", [256, 256]))
    rgb = load_mask_rgb(evidence, args, size)
    face_by_id = {int(face["id"]): face for face in evidence.get("faces", [])}

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("A. Source / evidence faces", fontsize=10)
    for face in evidence.get("faces", []):
        geometry = face["geometry"]
        draw_polygon(axes[0], geometry["outer"], geometry["holes"], facecolor=_face_color(int(face["label"])), linewidth=0.35, alpha=0.18)

    axes[1].imshow(rgb, alpha=0.18)
    draw_arcs(axes[1], evidence.get("arcs", []), linewidth=0.5)
    axes[1].set_title("B. Evidence shared arcs", fontsize=10)

    axes[2].imshow(np.full_like(rgb, 255))
    if operation.get("format") == "maskgen_manual_rule_explanation_v1":
        relation_colors = {
            "INSERTED_IN": "#ffbf00",
            "DIVIDES": "#17becf",
            "PARALLEL": "#2ca02c",
        }
        for explanation in operation.get("selected_explanations", []):
            relation_type = str(explanation.get("relation_type", ""))
            color = relation_colors.get(relation_type, "#999999")
            face_ids = []
            for key in ("insert_face_ids", "support_face_ids", "divider_face_ids", "left_face_ids", "right_face_ids"):
                face_ids.extend(int(value) for value in explanation.get(key, []))
            for face_id in sorted(set(face_ids)):
                face = face_by_id.get(int(face_id))
                if not face:
                    continue
                geometry = face["geometry"]
                draw_polygon(axes[2], geometry["outer"], geometry["holes"], facecolor=color, edgecolor="black", linewidth=0.25, alpha=0.12)
        axes[2].set_title("C. Manual role-spec explanations", fontsize=10)
    else:
        for candidate in operation.get("candidate_summary", {}).get("top_candidates", [])[:12]:
            face_ids = candidate.get("covered_face_ids", [])
            color = "#2ca02c" if candidate.get("compression_gain", 0.0) > 0 else "#999999"
            for face_id in face_ids:
                face = face_by_id.get(int(face_id))
                if not face:
                    continue
                geometry = face["geometry"]
                draw_polygon(axes[2], geometry["outer"], geometry["holes"], facecolor=color, edgecolor="black", linewidth=0.25, alpha=0.08)
        axes[2].set_title("C. Top operation candidates", fontsize=10)

    axes[3].imshow(np.full_like(rgb, 255))
    graph = operation.get("generator_target", {}).get("parse_graph", {})
    false_cover_faces = set()
    for item in operation.get("selected_operations", []):
        false_cover = item.get("cost", {}).get("breakdown", {}).get("false_cover", {})
        if float(false_cover.get("area", 0.0) or 0.0) > 0.0:
            false_cover_faces.update(int(value) for value in item.get("evidence", {}).get("face_ids", []))
    drawn_roles = []
    for node in graph.get("nodes", []):
        role = str(node.get("role", ""))
        evidence_info = node.get("evidence", {})
        color = ROLE_COLORS.get(role, "#333333")
        for face_id in evidence_info.get("face_ids", []):
            face = face_by_id.get(int(face_id))
            if not face:
                continue
            geometry = face["geometry"]
            edgecolor = "red" if int(face_id) in false_cover_faces else "black"
            linewidth = 0.8 if int(face_id) in false_cover_faces else 0.35
            draw_polygon(axes[3], geometry["outer"], geometry["holes"], facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=0.35)
            drawn_roles.append(role)
    add_role_legend(axes[3], drawn_roles)
    diagnostics = operation.get("diagnostics", {})
    if operation.get("format") == "maskgen_manual_rule_explanation_v1":
        axes[3].set_title(
            "D. Manual parse graph\n"
            f"nodes={diagnostics.get('node_count')}, relations={diagnostics.get('relation_count')}, "
            f"residual={diagnostics.get('residual_area_ratio', 0.0):.3f}, duplicate_owned={diagnostics.get('duplicate_owned_face_count', 0)}\n"
            f"{diagnostics.get('selection_method')}, role_spec_rules={diagnostics.get('role_spec_relation_count')}",
            fontsize=10,
        )
    else:
        source_histogram = diagnostics.get("label_pair_relation_source_histogram", {})
        explicit_rules = int(source_histogram.get("explicit_role_spec", 0) or 0)
        axes[3].set_title(
            "D. Selected operations\n"
            f"ops={diagnostics.get('selected_operation_count')}, residual={diagnostics.get('residual_area_ratio', 0.0):.3f}, "
            f"gain={diagnostics.get('total_compression_gain', 0.0):.1f}, false={diagnostics.get('false_cover_ratio_max', 0.0):.3f}\n"
            f"{diagnostics.get('selection_method')}, optimal={diagnostics.get('global_optimal')}, explicit_rules={explicit_rules}",
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
