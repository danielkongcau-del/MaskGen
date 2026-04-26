from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Sequence

from partition_gen.parse_graph_relations import divides_target, inserted_in_container
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, encode_generator_target


def _manual_relation_endpoint_ids(relation: Dict[str, object], node_ids: set[str]) -> List[str]:
    relation_type = str(relation.get("type"))
    if relation_type == "inserted_in":
        endpoints = [relation.get("object"), inserted_in_container(relation)]
    elif relation_type == "contains":
        endpoints = [relation.get("parent"), relation.get("child")]
    elif relation_type == "divides":
        endpoints = [relation.get("divider"), divides_target(relation)]
    elif relation_type == "adjacent_to":
        endpoints = list(relation.get("faces", []) or [])
    else:
        endpoints = []
    return [str(value) for value in endpoints if value is not None and str(value) in node_ids]


def polygon_payload_stats(geometry: Dict[str, object]) -> Dict[str, int]:
    polygons = geometry.get("polygons_local")
    if not polygons:
        polygons = [{"outer_local": geometry.get("outer_local", []), "holes_local": geometry.get("holes_local", [])}]
    component_count = 0
    outer_vertex_count = 0
    hole_count = 0
    hole_vertex_count = 0
    max_component_outer_vertices = 0
    for polygon in polygons or []:
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
        "polygon_vertex_count": int(outer_vertex_count + hole_vertex_count),
    }


def _polygon_payload_token_count(geometry: Dict[str, object]) -> int:
    polygons = geometry.get("polygons_local")
    if not polygons:
        polygons = [{"outer_local": geometry.get("outer_local", []), "holes_local": geometry.get("holes_local", [])}]
    total = 2  # POLYS count
    for polygon in polygons or []:
        outer = polygon.get("outer_local", []) or []
        holes = polygon.get("holes_local", []) or []
        total += 1  # POLY
        total += 2 + 2 * len(outer)  # PTS count coords
        total += 2  # HOLES count
        for hole in holes:
            total += 1  # HOLE
            total += 2 + 2 * len(hole)
            total += 1  # END_HOLE
        total += 1  # END_POLY
    return int(total)


def _convex_atoms_token_count(atoms: Sequence[Dict[str, object]]) -> tuple[int, int, int]:
    total = 2  # ATOMS count
    atom_count = 0
    atom_vertex_count = 0
    for atom in atoms or []:
        outer = atom.get("outer_local", []) or []
        vertex_count = int(atom.get("vertex_count", len(outer)))
        atom_count += 1
        atom_vertex_count += vertex_count
        total += 4  # ATOM type AREA value
        total += 2 + 2 * len(outer)  # PTS count coords
        total += 1  # END_ATOM
    return int(total), int(atom_count), int(atom_vertex_count)


def _node_token_stats(node: Dict[str, object]) -> Dict[str, object]:
    role = str(node.get("role", "unknown"))
    geometry_model = str(node.get("geometry_model", "none"))
    header_tokens = 7  # NODE role LABEL label REF_ONLY value geometry_model
    if role == "insert_object_group":
        header_tokens += 2  # COUNT value
    end_tokens = 1
    frame_tokens = 0
    polygon_tokens = 0
    convex_atom_tokens = 0
    atom_count = 0
    atom_vertex_count = 0
    geometry_stats = {
        "component_count": 0,
        "outer_vertex_count": 0,
        "hole_count": 0,
        "hole_vertex_count": 0,
        "max_component_outer_vertices": 0,
        "polygon_vertex_count": 0,
    }
    if geometry_model == "polygon_code":
        frame_tokens = 5
        geometry_stats = polygon_payload_stats(node.get("geometry", {}) or {})
        polygon_tokens = _polygon_payload_token_count(node.get("geometry", {}) or {})
    elif geometry_model == "convex_atoms":
        frame_tokens = 5
        convex_atom_tokens, atom_count, atom_vertex_count = _convex_atoms_token_count(node.get("atoms", []) or [])
    total = header_tokens + frame_tokens + polygon_tokens + convex_atom_tokens + end_tokens
    return {
        "id": node.get("id"),
        "role": role,
        "label": node.get("label"),
        "geometry_model": geometry_model,
        "renderable": bool(node.get("renderable", True)),
        "is_reference_only": bool(node.get("is_reference_only", False)),
        "token_count": int(total),
        "node_header_token_count": int(header_tokens + end_tokens),
        "frame_token_count": int(frame_tokens),
        "polygon_token_count": int(polygon_tokens),
        "convex_atom_token_count": int(convex_atom_tokens),
        "geometry_token_count": int(frame_tokens + polygon_tokens + convex_atom_tokens),
        "vertex_count": int(geometry_stats["polygon_vertex_count"] + atom_vertex_count),
        "polygon_vertex_count": int(geometry_stats["polygon_vertex_count"]),
        "atom_count": int(atom_count),
        "atom_vertex_count": int(atom_vertex_count),
        **geometry_stats,
    }


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(int(value) for value in values)
    index = int(round((len(sorted_values) - 1) * percentile))
    return int(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def summarize_manual_token_stat_rows(rows: Sequence[Dict[str, object]], *, top_k: int = 20) -> str:
    lengths = [int(row["total_tokens"]) for row in rows]
    polygon_vertices = [int(row["polygon_vertex_count"]) for row in rows]
    node_counts = [int(row["node_count"]) for row in rows]
    relation_counts = [int(row["relation_count"]) for row in rows]
    role_totals: Dict[str, int] = {}
    geometry_totals: Dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("tokens_by_role") or {}).items():
            role_totals[key] = int(role_totals.get(key, 0) + int(value))
        for key, value in (row.get("tokens_by_geometry_model") or {}).items():
            geometry_totals[key] = int(geometry_totals.get(key, 0) + int(value))

    def table(items: Iterable[Dict[str, object]], columns: Sequence[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for item in items:
            lines.append("| " + " | ".join(str(item.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    longest = sorted(rows, key=lambda item: int(item["total_tokens"]), reverse=True)[:top_k]
    most_vertices = sorted(rows, key=lambda item: int(item["polygon_vertex_count"]), reverse=True)[:top_k]
    most_relations = sorted(rows, key=lambda item: int(item["relation_count"]), reverse=True)[:top_k]
    lines = [
        "# Manual Target Token Length Summary",
        "",
        f"- samples: {len(rows)}",
        f"- mean_tokens: {mean(lengths):.2f}" if lengths else "- mean_tokens: n/a",
        f"- median_tokens: {_percentile(lengths, 0.50)}",
        f"- p90_tokens: {_percentile(lengths, 0.90)}",
        f"- p95_tokens: {_percentile(lengths, 0.95)}",
        f"- p99_tokens: {_percentile(lengths, 0.99)}",
        f"- max_tokens: {max(lengths) if lengths else 'n/a'}",
        f"- mean_polygon_vertex_count: {mean(polygon_vertices):.2f}" if polygon_vertices else "- mean_polygon_vertex_count: n/a",
        f"- max_polygon_vertex_count: {max(polygon_vertices) if polygon_vertices else 'n/a'}",
        f"- mean_node_count: {mean(node_counts):.2f}" if node_counts else "- mean_node_count: n/a",
        f"- max_node_count: {max(node_counts) if node_counts else 'n/a'}",
        f"- mean_relation_count: {mean(relation_counts):.2f}" if relation_counts else "- mean_relation_count: n/a",
        f"- max_relation_count: {max(relation_counts) if relation_counts else 'n/a'}",
        "",
        "## Longest Samples",
        "",
        table(longest, ["stem", "total_tokens", "node_count", "relation_count", "polygon_vertex_count", "polygon_token_count"]),
        "",
        "## Most Polygon Vertices",
        "",
        table(most_vertices, ["stem", "total_tokens", "polygon_vertex_count", "polygon_component_count", "hole_count", "hole_vertex_count"]),
        "",
        "## Most Relations",
        "",
        table(most_relations, ["stem", "total_tokens", "relation_count", "relation_token_count", "node_count"]),
        "",
        "## Tokens By Role",
        "",
        table([{"role": key, "tokens": value} for key, value in sorted(role_totals.items())], ["role", "tokens"]),
        "",
        "## Tokens By Geometry Model",
        "",
        table([{"geometry_model": key, "tokens": value} for key, value in sorted(geometry_totals.items())], ["geometry_model", "tokens"]),
    ]
    return "\n".join(lines) + "\n"


def analyze_manual_target_token_stats(
    target: Dict[str, object],
    *,
    tokenizer_config: ParseGraphTokenizerConfig | None = None,
) -> Dict[str, object]:
    config = tokenizer_config or ParseGraphTokenizerConfig()
    full_token_count = len(encode_generator_target(target, config=config))
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    residuals = list(graph.get("residuals", []) or [])
    node_ids = {str(node.get("id")) for node in nodes}

    node_rows = [_node_token_stats(node) for node in nodes]
    relation_type_histogram: Dict[str, int] = {}
    relation_token_count = 2
    for relation in relations:
        relation_type = str(relation.get("type", "unknown"))
        relation_type_histogram[relation_type] = int(relation_type_histogram.get(relation_type, 0) + 1)
        relation_token_count += 5 + len(_manual_relation_endpoint_ids(relation, node_ids))

    tokens_by_role: Dict[str, int] = {}
    tokens_by_geometry_model: Dict[str, int] = {}
    for row in node_rows:
        tokens_by_role[str(row["role"])] = int(tokens_by_role.get(str(row["role"]), 0) + int(row["token_count"]))
        tokens_by_geometry_model[str(row["geometry_model"])] = int(
            tokens_by_geometry_model.get(str(row["geometry_model"]), 0) + int(row["token_count"])
        )

    node_header_token_count = sum(int(row["node_header_token_count"]) for row in node_rows)
    polygon_token_count = sum(int(row["polygon_token_count"]) for row in node_rows)
    convex_atom_token_count = sum(int(row["convex_atom_token_count"]) for row in node_rows)
    frame_token_count = sum(int(row["frame_token_count"]) for row in node_rows)
    geometry_token_count = sum(int(row["geometry_token_count"]) for row in node_rows)
    attributed_token_count = 5 + 2 + 3 + sum(int(row["token_count"]) for row in node_rows) + relation_token_count
    top_nodes = sorted(node_rows, key=lambda item: int(item["token_count"]), reverse=True)[:20]
    return {
        "total_tokens": int(full_token_count),
        "full_token_count": int(full_token_count),
        "attributed_token_count": int(attributed_token_count),
        "attribution_gap": int(full_token_count - attributed_token_count),
        "node_count": int(len(nodes)),
        "renderable_node_count": int(sum(1 for node in nodes if bool(node.get("renderable", True)))),
        "non_renderable_node_count": int(sum(1 for node in nodes if not bool(node.get("renderable", True)))),
        "reference_only_node_count": int(sum(1 for node in nodes if bool(node.get("is_reference_only", False)))),
        "relation_count": int(len(relations)),
        "residual_count": int(len(residuals)),
        "geometry_token_count": int(geometry_token_count),
        "frame_token_count": int(frame_token_count),
        "polygon_token_count": int(polygon_token_count),
        "convex_atom_token_count": int(convex_atom_token_count),
        "node_header_token_count": int(node_header_token_count),
        "relation_token_count": int(relation_token_count),
        "residual_token_count": 3,
        "polygon_vertex_count": int(sum(int(row["polygon_vertex_count"]) for row in node_rows)),
        "polygon_component_count": int(sum(int(row["component_count"]) for row in node_rows)),
        "hole_count": int(sum(int(row["hole_count"]) for row in node_rows)),
        "hole_vertex_count": int(sum(int(row["hole_vertex_count"]) for row in node_rows)),
        "tokens_by_role": dict(sorted(tokens_by_role.items())),
        "tokens_by_geometry_model": dict(sorted(tokens_by_geometry_model.items())),
        "relation_type_histogram": dict(sorted(relation_type_histogram.items())),
        "max_node_token_count": int(max([int(row["token_count"]) for row in node_rows] or [0])),
        "max_node_vertex_count": int(max([int(row["vertex_count"]) for row in node_rows] or [0])),
        "top_nodes_by_token_count": top_nodes,
    }
