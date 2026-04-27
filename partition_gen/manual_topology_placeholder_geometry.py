from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from partition_gen.manual_topology_sample_validation import validate_topology_tokens
from partition_gen.parse_graph_tokenizer import token_int


ROLE_TOKEN_TO_MANUAL_ROLE = {
    "ROLE_SUPPORT": "support_region",
    "ROLE_DIVIDER": "divider_region",
    "ROLE_INSERT": "insert_object",
    "ROLE_INSERT_GROUP": "insert_object_group",
    "ROLE_RESIDUAL": "residual_region",
    "ROLE_UNKNOWN": "unknown",
}

MANUAL_ROLE_TO_ID_PREFIX = {
    "support_region": "support",
    "divider_region": "divider",
    "insert_object": "insert",
    "insert_object_group": "insert_group",
    "residual_region": "residual",
    "unknown": "node",
}

GEOMETRY_TOKEN_TO_MODEL = {
    "GEOM_NONE": "none",
    "GEOM_POLYGON_CODE": "polygon_code",
    "GEOM_CONVEX_ATOMS": "convex_atoms",
    "GEOM_UNKNOWN": "unknown",
}

RELATION_TOKEN_TO_TYPE = {
    "REL_INSERTED_IN": "inserted_in",
    "REL_CONTAINS": "contains",
    "REL_DIVIDES": "divides",
    "REL_ADJACENT_TO": "adjacent_to",
    "REL_UNKNOWN": "unknown",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _node_id_for_index(node_ids: Sequence[str], index: int) -> str:
    if index < 0 or index >= len(node_ids):
        raise ValueError(f"Node index {index} out of range for {len(node_ids)} decoded nodes")
    return str(node_ids[int(index)])


def _next_node_id(role: str, counters: Dict[str, int]) -> str:
    prefix = MANUAL_ROLE_TO_ID_PREFIX.get(str(role), "node")
    index = int(counters.get(prefix, 0))
    counters[prefix] = index + 1
    return f"{prefix}_{index}"


class _TokenReader:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = [str(token) for token in tokens]
        self.index = 0

    def next(self) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of topology token sequence")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, expected: str) -> None:
        actual = self.next()
        if actual != expected:
            raise ValueError(f"Expected {expected} at {self.index - 1}, got {actual}")

    def next_int(self) -> int:
        return int(token_int(self.next()))

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None


def decode_topology_tokens_to_target(tokens: Sequence[str]) -> dict:
    """Decode validated MANUAL_TOPOLOGY_V1 tokens into a topology target JSON."""

    reader = _TokenReader(tokens)
    reader.expect("<BOS>")
    reader.expect("MANUAL_TOPOLOGY_V1")
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]
    reader.expect("NODE_BLOCK")
    node_count = reader.next_int()

    nodes: List[dict] = []
    node_ids: List[str] = []
    children_by_node_id: Dict[str, List[int]] = {}
    counters: Dict[str, int] = {}

    for _node_index in range(int(node_count)):
        reader.expect("NODE")
        role = ROLE_TOKEN_TO_MANUAL_ROLE.get(reader.next(), "unknown")
        node_id = _next_node_id(role, counters)
        label = reader.next_int()
        renderable = bool(reader.next_int())
        is_reference_only = bool(reader.next_int())
        geometry_model = GEOMETRY_TOKEN_TO_MODEL.get(reader.next(), "unknown")
        geometry_ref = bool(reader.next_int())
        node = {
            "id": node_id,
            "role": role,
            "label": int(label),
            "renderable": renderable,
            "is_reference_only": is_reference_only,
            "geometry_model": geometry_model,
        }
        if geometry_ref:
            node["geometry_ref"] = node_id
        if reader.peek() == "CHILDREN":
            reader.expect("CHILDREN")
            child_count = reader.next_int()
            children_by_node_id[node_id] = [reader.next_int() for _ in range(int(child_count))]
        reader.expect("END_NODE")
        nodes.append(node)
        node_ids.append(node_id)

    for node in nodes:
        child_indices = children_by_node_id.get(str(node["id"]))
        if child_indices is not None:
            node["children"] = [_node_id_for_index(node_ids, index) for index in child_indices]
            node["count"] = int(len(node["children"]))

    relations: List[dict] = []
    for node in nodes:
        if str(node.get("role")) != "insert_object_group":
            continue
        for child_id in node.get("children", []) or []:
            relations.append({"type": "contains", "parent": str(node["id"]), "child": str(child_id)})

    for block_name, relation_type in (
        ("REL_BLOCK_INSERTED_IN", "inserted_in"),
        ("REL_BLOCK_DIVIDES", "divides"),
        ("REL_BLOCK_ADJACENT_TO", "adjacent_to"),
    ):
        reader.expect(block_name)
        pair_count = reader.next_int()
        for _pair_index in range(int(pair_count)):
            left = _node_id_for_index(node_ids, reader.next_int())
            right = _node_id_for_index(node_ids, reader.next_int())
            if relation_type == "inserted_in":
                relations.append({"type": relation_type, "object": left, "container": right, "support": right})
            elif relation_type == "divides":
                relations.append({"type": relation_type, "divider": left, "target": right, "support": right})
            else:
                relations.append({"type": relation_type, "faces": [left, right]})
        reader.expect("END_BLOCK")

    reader.expect("REL_BLOCK_OTHER")
    other_count = reader.next_int()
    for _relation_index in range(int(other_count)):
        relation_type = RELATION_TOKEN_TO_TYPE.get(reader.next(), "unknown")
        ref_count = reader.next_int()
        refs = [_node_id_for_index(node_ids, reader.next_int()) for _ in range(int(ref_count))]
        relations.append({"type": relation_type, "refs": refs})
    reader.expect("END_BLOCK")

    reader.expect("RESIDUALS")
    residual_count = reader.next_int()
    residuals = [{"reason": "tokenized_residual_placeholder"} for _ in range(int(residual_count))]
    reader.expect("<EOS>")
    if reader.index != len(reader.tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(reader.tokens) - reader.index}")

    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_topology_v1",
        "size": size,
        "parse_graph": {"nodes": nodes, "relations": relations, "residuals": residuals},
        "metadata": {"decoded_from_tokens": True, "tokenizer": "manual_topology_v1"},
    }


def _resolve_path(value: object, *, split_root: Path, manifest_parent: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    for base in (manifest_parent, split_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


class GeometryPlaceholderLibrary:
    def __init__(self, geometry_targets: Sequence[dict], *, seed: int = 1337) -> None:
        self.targets = [copy.deepcopy(target) for target in geometry_targets]
        self.rng = random.Random(int(seed))
        self.exact: Dict[tuple[str, int, str], List[dict]] = defaultdict(list)
        self.role_model: Dict[tuple[str, str], List[dict]] = defaultdict(list)
        self.model: Dict[str, List[dict]] = defaultdict(list)
        for target in self.targets:
            role = str(target.get("role", ""))
            label = int(target.get("label", 0))
            geometry_model = str(target.get("geometry_model", "none"))
            self.exact[(role, label, geometry_model)].append(target)
            self.role_model[(role, geometry_model)].append(target)
            self.model[geometry_model].append(target)

    @classmethod
    def from_split_manifest(cls, split_root: Path, *, seed: int = 1337, max_geometry_targets: int | None = None) -> "GeometryPlaceholderLibrary":
        manifest_path = split_root / "manifest.jsonl"
        rows = list(iter_jsonl(manifest_path))
        geometry_targets: List[dict] = []
        for row in rows:
            for value in row.get("geometry_paths", []) or []:
                path = _resolve_path(value, split_root=split_root, manifest_parent=manifest_path.parent)
                geometry_targets.append(load_json(path))
                if max_geometry_targets is not None and len(geometry_targets) >= int(max_geometry_targets):
                    return cls(geometry_targets, seed=seed)
        return cls(geometry_targets, seed=seed)

    def choose(self, *, role: str, label: int, geometry_model: str) -> tuple[dict | None, str]:
        exact = self.exact.get((str(role), int(label), str(geometry_model)), [])
        if exact:
            return copy.deepcopy(self.rng.choice(exact)), "exact"
        role_model = self.role_model.get((str(role), str(geometry_model)), [])
        if role_model:
            return copy.deepcopy(self.rng.choice(role_model)), "role_model"
        model = self.model.get(str(geometry_model), [])
        if model:
            return copy.deepcopy(self.rng.choice(model)), "geometry_model"
        if self.targets:
            return copy.deepcopy(self.rng.choice(self.targets)), "any"
        return None, "missing"


def attach_placeholder_geometry(
    topology_target: dict,
    library: GeometryPlaceholderLibrary,
) -> tuple[dict, dict]:
    graph = topology_target.get("parse_graph", {}) or {}
    nodes: List[dict] = []
    attach_modes: Counter[str] = Counter()
    missing_nodes: List[str] = []

    for node in graph.get("nodes", []) or []:
        output_node = copy.deepcopy(node)
        geometry_ref = output_node.pop("geometry_ref", None)
        if geometry_ref:
            target, mode = library.choose(
                role=str(output_node.get("role", "")),
                label=int(output_node.get("label", 0)),
                geometry_model=str(output_node.get("geometry_model", "none")),
            )
            attach_modes[mode] += 1
            if target is None:
                missing_nodes.append(str(output_node.get("id")))
            else:
                output_node["geometry_model"] = copy.deepcopy(target.get("geometry_model", output_node.get("geometry_model")))
                if "frame" in target:
                    output_node["frame"] = copy.deepcopy(target["frame"])
                if "geometry" in target:
                    output_node["geometry"] = copy.deepcopy(target["geometry"])
                if "atoms" in target:
                    output_node["atoms"] = copy.deepcopy(target["atoms"])
                output_node["placeholder_geometry_source_node_id"] = str(target.get("source_node_id"))
                output_node["placeholder_geometry_attach_mode"] = mode
        nodes.append(output_node)

    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": copy.deepcopy(topology_target.get("size", [0, 0])),
        "parse_graph": {
            "nodes": nodes,
            "relations": copy.deepcopy(list(graph.get("relations", []) or [])),
            "residuals": copy.deepcopy(list(graph.get("residuals", []) or [])),
        },
        "metadata": {
            "placeholder_geometry": True,
            "source_topology_metadata": copy.deepcopy(topology_target.get("metadata", {})),
            "attached_geometry_count": int(sum(attach_modes.values()) - attach_modes.get("missing", 0)),
            "missing_geometry_count": int(len(missing_nodes)),
            "attach_modes": dict(attach_modes),
        },
    }
    diagnostics = {
        "node_count": int(len(nodes)),
        "relation_count": int(len(target["parse_graph"]["relations"])),
        "attached_geometry_count": int(target["metadata"]["attached_geometry_count"]),
        "missing_geometry_count": int(len(missing_nodes)),
        "attach_modes": dict(attach_modes),
    }
    return target, diagnostics


def build_placeholder_targets_from_sample_rows(
    rows: Sequence[dict],
    library: GeometryPlaceholderLibrary,
    *,
    include_invalid: bool = False,
) -> tuple[List[dict], dict]:
    targets: List[dict] = []
    diagnostics_rows: List[dict] = []
    skipped_invalid = 0
    for fallback_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not include_invalid:
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        topology_target["metadata"].update(
            {
                "sample_index": sample_index,
                "semantic_valid": bool(validation["semantic_valid"]),
                "checkpoint": row.get("checkpoint"),
            }
        )
        target, diagnostics = attach_placeholder_geometry(topology_target, library)
        target["metadata"]["sample_index"] = sample_index
        targets.append(target)
        diagnostics_rows.append({"sample_index": sample_index, **diagnostics})

    node_counts = [int(row["node_count"]) for row in diagnostics_rows]
    attached_counts = [int(row["attached_geometry_count"]) for row in diagnostics_rows]
    attach_modes = Counter()
    for row in diagnostics_rows:
        attach_modes.update(row.get("attach_modes", {}))
    summary = {
        "format": "maskgen_placeholder_geometry_summary_v1",
        "input_count": int(len(rows)),
        "output_count": int(len(targets)),
        "skipped_invalid_count": int(skipped_invalid),
        "node_count_mean": float(mean(node_counts)) if node_counts else None,
        "attached_geometry_mean": float(mean(attached_counts)) if attached_counts else None,
        "attach_modes": dict(attach_modes),
        "rows": diagnostics_rows,
    }
    return targets, summary
