from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, encode_generator_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose token length drivers for generator target parse_graph JSON files.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_target_paths(root: Path, split: str) -> Iterable[Path]:
    graph_root = root / split / "graphs"
    if graph_root.exists():
        yield from sorted(graph_root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    split_root = root / split
    if split_root.exists():
        yield from sorted(split_root.glob("*.json"), key=lambda path: (len(path.stem), path.stem))
        return
    yield from sorted(root.rglob("*.json"), key=lambda path: (str(path.parent), len(path.stem), path.stem))


def _polygon_stats(geometry: dict) -> dict:
    polygons = geometry.get("polygons_local")
    if not polygons:
        polygons = [{"outer_local": geometry.get("outer_local", []), "holes_local": geometry.get("holes_local", [])}]
    component_count = 0
    outer_vertex_count = 0
    hole_count = 0
    hole_vertex_count = 0
    max_component_outer_vertices = 0
    for polygon in polygons:
        outer = polygon.get("outer_local", []) or []
        holes = polygon.get("holes_local", []) or []
        component_count += 1
        outer_vertex_count += len(outer)
        max_component_outer_vertices = max(max_component_outer_vertices, len(outer))
        hole_count += len(holes)
        hole_vertex_count += sum(len(hole) for hole in holes)
    return {
        "component_count": int(component_count),
        "outer_vertex_count": int(outer_vertex_count),
        "hole_count": int(hole_count),
        "hole_vertex_count": int(hole_vertex_count),
        "max_component_outer_vertices": int(max_component_outer_vertices),
    }


def _node_geometry_stats(node: dict) -> dict:
    geometry_model = str(node.get("geometry_model", "none"))
    if geometry_model == "polygon_code":
        stats = _polygon_stats(node.get("geometry", {}))
        return {
            **stats,
            "atom_count": 0,
            "atom_vertex_count": 0,
            "geometry_vertex_count": int(stats["outer_vertex_count"] + stats["hole_vertex_count"]),
        }
    if geometry_model == "convex_atoms":
        atoms = node.get("atoms", []) or []
        atom_vertices = sum(int(atom.get("vertex_count", len(atom.get("outer_local", [])))) for atom in atoms)
        return {
            "component_count": 0,
            "outer_vertex_count": 0,
            "hole_count": 0,
            "hole_vertex_count": 0,
            "max_component_outer_vertices": 0,
            "atom_count": int(len(atoms)),
            "atom_vertex_count": int(atom_vertices),
            "geometry_vertex_count": int(atom_vertices),
        }
    return {
        "component_count": 0,
        "outer_vertex_count": 0,
        "hole_count": 0,
        "hole_vertex_count": 0,
        "max_component_outer_vertices": 0,
        "atom_count": 0,
        "atom_vertex_count": 0,
        "geometry_vertex_count": 0,
    }


def diagnose_target(path: Path, config: ParseGraphTokenizerConfig) -> dict:
    target = load_json(path)
    graph = target.get("parse_graph", {})
    nodes = graph.get("nodes", []) or []
    relations = graph.get("relations", []) or []
    tokens = encode_generator_target(target, config=config)
    role_histogram: dict[str, int] = {}
    geometry_model_histogram: dict[str, int] = {}
    relation_histogram: dict[str, int] = {}
    node_rows = []
    totals = {
        "component_count": 0,
        "outer_vertex_count": 0,
        "hole_count": 0,
        "hole_vertex_count": 0,
        "atom_count": 0,
        "atom_vertex_count": 0,
        "geometry_vertex_count": 0,
    }
    for node in nodes:
        role = str(node.get("role", "unknown"))
        geometry_model = str(node.get("geometry_model", "none"))
        role_histogram[role] = int(role_histogram.get(role, 0) + 1)
        geometry_model_histogram[geometry_model] = int(geometry_model_histogram.get(geometry_model, 0) + 1)
        stats = _node_geometry_stats(node)
        for key in totals:
            totals[key] += int(stats.get(key, 0))
        node_rows.append(
            {
                "id": node.get("id"),
                "role": role,
                "label": node.get("label"),
                "geometry_model": geometry_model,
                "is_reference_only": bool(node.get("is_reference_only", False)),
                **stats,
            }
        )
    for relation in relations:
        relation_type = str(relation.get("type", "unknown"))
        relation_histogram[relation_type] = int(relation_histogram.get(relation_type, 0) + 1)

    top_nodes = sorted(node_rows, key=lambda item: int(item["geometry_vertex_count"]), reverse=True)[:10]
    return {
        "source_target": str(path.as_posix()),
        "stem": path.stem,
        "token_length": int(len(tokens)),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "residual_count": int(len(graph.get("residuals", []) or [])),
        "reference_only_count": int(sum(1 for node in nodes if bool(node.get("is_reference_only", False)))),
        "role_histogram": dict(sorted(role_histogram.items())),
        "geometry_model_histogram": dict(sorted(geometry_model_histogram.items())),
        "relation_histogram": dict(sorted(relation_histogram.items())),
        **totals,
        "top_geometry_nodes": top_nodes,
    }


def _table(rows: List[dict], columns: List[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(rows: List[dict]) -> str:
    lengths = [int(row["token_length"]) for row in rows]
    vertices = [int(row["geometry_vertex_count"]) for row in rows]
    lines = [
        "# Generator Target Token Diagnostics",
        "",
        f"- samples: {len(rows)}",
        f"- mean_token_length: {mean(lengths):.2f}" if lengths else "- mean_token_length: n/a",
        f"- max_token_length: {max(lengths) if lengths else 'n/a'}",
        f"- mean_geometry_vertex_count: {mean(vertices):.2f}" if vertices else "- mean_geometry_vertex_count: n/a",
        f"- max_geometry_vertex_count: {max(vertices) if vertices else 'n/a'}",
        "",
        "## Longest Samples",
        "",
        _table(
            sorted(rows, key=lambda item: int(item["token_length"]), reverse=True)[:20],
            [
                "stem",
                "token_length",
                "node_count",
                "relation_count",
                "geometry_vertex_count",
                "outer_vertex_count",
                "hole_count",
                "hole_vertex_count",
                "component_count",
                "reference_only_count",
            ],
        ),
        "",
        "## Most Geometry Vertices",
        "",
        _table(
            sorted(rows, key=lambda item: int(item["geometry_vertex_count"]), reverse=True)[:20],
            [
                "stem",
                "token_length",
                "node_count",
                "relation_count",
                "geometry_vertex_count",
                "outer_vertex_count",
                "hole_count",
                "hole_vertex_count",
                "component_count",
            ],
        ),
        "",
        "## Top Geometry Nodes In Longest Samples",
    ]
    for row in sorted(rows, key=lambda item: int(item["token_length"]), reverse=True)[:5]:
        lines.extend(
            [
                "",
                f"### {row['stem']} length={row['token_length']}",
                "",
                _table(
                    row.get("top_geometry_nodes", [])[:10],
                    [
                        "id",
                        "role",
                        "label",
                        "geometry_model",
                        "is_reference_only",
                        "geometry_vertex_count",
                        "outer_vertex_count",
                        "hole_count",
                        "hole_vertex_count",
                        "component_count",
                    ],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = ParseGraphTokenizerConfig()
    rows = []
    for index, path in enumerate(iter_target_paths(args.target_root, args.split)):
        if args.max_samples is not None and index >= int(args.max_samples):
            break
        target = load_json(path)
        if target.get("format") != "maskgen_generator_target_v1":
            continue
        rows.append(diagnose_target(path, config))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(summarize(rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output_jsonl}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
