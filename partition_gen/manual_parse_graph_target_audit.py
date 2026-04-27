from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

from partition_gen.parse_graph_compact_tokenizer import encode_generator_target_compact
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, encode_generator_target


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return path


def iter_manual_parse_graph_target_paths(root: Path) -> List[Path]:
    """Resolve full parse-graph JSON paths from a file, graph dir, or placeholder output root."""

    root = Path(root)
    if root.is_file():
        return [root]

    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        paths: List[Path] = []
        for row in iter_jsonl(manifest_path):
            value = row.get("output_path") or row.get("target_path") or row.get("path")
            if value is not None:
                paths.append(_resolve_path(value, base=manifest_path.parent))
        return paths

    graph_dir = root / "graphs"
    if graph_dir.exists():
        return sorted(graph_dir.glob("*.json"))
    return sorted(root.glob("*.json"))


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(int(value) for value in values)
    index = int(math.ceil(float(percentile) * len(sorted_values))) - 1
    return int(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _stats(values: Sequence[int]) -> dict:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": float(mean(int(value) for value in values)),
        "p95": _percentile(values, 0.95),
        "max": int(max(values)),
    }


def _sort_histogram(counter: Counter) -> dict:
    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        key, _value = item
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return dict(sorted(counter.items(), key=sort_key))


def _geometry_payload_missing(node: dict) -> bool:
    geometry_model = str(node.get("geometry_model", "none"))
    if geometry_model == "polygon_code":
        return "frame" not in node or "geometry" not in node
    if geometry_model == "convex_atoms":
        return "frame" not in node or "atoms" not in node
    return False


def audit_manual_parse_graph_target(
    target: dict,
    *,
    source: str | None = None,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    graph = target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    errors: List[str] = []

    if target.get("format") != "maskgen_generator_target_v1":
        errors.append("format_not_maskgen_generator_target_v1")
    if target.get("target_type") != "parse_graph":
        errors.append("target_type_not_parse_graph")

    renderable_nodes = [
        node
        for node in nodes
        if bool(node.get("renderable", True)) and not bool(node.get("is_reference_only", False))
    ]
    geometry_nodes = [
        node
        for node in nodes
        if str(node.get("geometry_model", "none")) in {"polygon_code", "convex_atoms"}
    ]
    missing_geometry_nodes = [
        str(node.get("id", ""))
        for node in geometry_nodes
        if _geometry_payload_missing(node)
    ]
    role_histogram = Counter(str(node.get("role", "")) for node in nodes)
    label_histogram = Counter(str(node.get("label", 0)) for node in nodes)

    old_token_count = None
    compact_token_count = None
    old_encodable = False
    compact_encodable = False
    try:
        old_token_count = len(encode_generator_target(target, config=config))
        old_encodable = True
    except Exception as exc:  # pragma: no cover - exact encoder failures are data-dependent
        errors.append(f"old_tokenizer_error:{type(exc).__name__}:{exc}")
    try:
        compact_token_count = len(encode_generator_target_compact(target, config=config))
        compact_encodable = True
    except Exception as exc:  # pragma: no cover - exact encoder failures are data-dependent
        errors.append(f"compact_tokenizer_error:{type(exc).__name__}:{exc}")

    return {
        "source": source,
        "target_type": target.get("target_type"),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "renderable_node_count": int(len(renderable_nodes)),
        "geometry_node_count": int(len(geometry_nodes)),
        "missing_geometry_payload_count": int(len(missing_geometry_nodes)),
        "missing_geometry_node_ids": missing_geometry_nodes,
        "placeholder_geometry_source_count": int(
            sum(1 for node in nodes if "placeholder_geometry_source_node_id" in node)
        ),
        "old_token_count": old_token_count,
        "compact_token_count": compact_token_count,
        "old_encodable": bool(old_encodable),
        "compact_encodable": bool(compact_encodable),
        "encodable": bool(old_encodable and compact_encodable),
        "errors": errors,
        "role_histogram": dict(sorted(role_histogram.items())),
        "label_histogram": _sort_histogram(label_histogram),
    }


def audit_manual_parse_graph_targets(
    paths: Sequence[Path],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    rows: List[dict] = []
    load_errors: List[dict] = []
    config = config or ParseGraphTokenizerConfig()
    for path in paths:
        try:
            target = load_json(path)
            rows.append(audit_manual_parse_graph_target(target, source=str(Path(path).as_posix()), config=config))
        except Exception as exc:
            load_errors.append({"source": str(Path(path).as_posix()), "error": f"{type(exc).__name__}:{exc}"})

    node_counts = [int(row["node_count"]) for row in rows]
    relation_counts = [int(row["relation_count"]) for row in rows]
    old_token_counts = [int(row["old_token_count"]) for row in rows if row["old_token_count"] is not None]
    compact_token_counts = [int(row["compact_token_count"]) for row in rows if row["compact_token_count"] is not None]
    role_histogram = Counter()
    label_histogram = Counter()
    for row in rows:
        role_histogram.update(row.get("role_histogram", {}))
        label_histogram.update(row.get("label_histogram", {}))

    return {
        "format": "maskgen_manual_parse_graph_target_audit_v1",
        "input_path_count": int(len(paths)),
        "loaded_count": int(len(rows)),
        "load_error_count": int(len(load_errors)),
        "encodable_count": int(sum(1 for row in rows if row["encodable"])),
        "old_encodable_count": int(sum(1 for row in rows if row["old_encodable"])),
        "compact_encodable_count": int(sum(1 for row in rows if row["compact_encodable"])),
        "missing_geometry_payload_count": int(sum(int(row["missing_geometry_payload_count"]) for row in rows)),
        "placeholder_geometry_source_count": int(sum(int(row["placeholder_geometry_source_count"]) for row in rows)),
        "node_counts": _stats(node_counts),
        "relation_counts": _stats(relation_counts),
        "old_token_counts": _stats(old_token_counts),
        "compact_token_counts": _stats(compact_token_counts),
        "role_histogram": dict(sorted(role_histogram.items())),
        "label_histogram": _sort_histogram(label_histogram),
        "load_errors": load_errors,
        "rows": rows,
    }
