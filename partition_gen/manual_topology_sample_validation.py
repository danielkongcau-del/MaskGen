from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from partition_gen.parse_graph_tokenizer import token_int


class TopologyTokenParser:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = [str(token) for token in tokens]
        self.index = 0
        self.errors: List[str] = []
        self.semantic_errors: List[str] = []
        self.node_count_declared = 0
        self.node_count_actual = 0
        self.relation_counts: Dict[str, int] = {}
        self.nodes: List[Dict[str, object]] = []
        self.relation_pairs: Dict[str, List[tuple[int, int]]] = {}

    def current(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def consume(self, expected: str) -> bool:
        if self.current() != expected:
            self.errors.append(f"expected_{expected}_at_{self.index}_got_{self.current()}")
            return False
        self.index += 1
        return True

    def consume_int(self, *, name: str) -> int | None:
        token = self.current()
        if token is None:
            self.errors.append(f"missing_int_{name}_at_{self.index}")
            return None
        try:
            value = token_int(token)
        except Exception:
            self.errors.append(f"expected_int_{name}_at_{self.index}_got_{token}")
            return None
        self.index += 1
        return int(value)

    def validate(self) -> Dict[str, object]:
        self._parse()
        hit_eos = bool("<EOS>" in self.tokens)
        valid = bool(not self.errors and hit_eos and self.index == len(self.tokens))
        if valid:
            self._validate_semantics()
        semantic_valid = bool(valid and not self.semantic_errors)
        return {
            "valid": valid,
            "semantic_valid": semantic_valid,
            "errors": self.errors,
            "semantic_errors": self.semantic_errors,
            "hit_eos": hit_eos,
            "length": int(len(self.tokens)),
            "node_count_declared": int(self.node_count_declared),
            "node_count_actual": int(self.node_count_actual),
            "relation_counts": self.relation_counts,
        }

    def _parse(self) -> None:
        if not self.consume("<BOS>"):
            return
        if not self.consume("MANUAL_TOPOLOGY_V1"):
            return
        if not self.consume("SIZE"):
            return
        self.consume_int(name="height")
        self.consume_int(name="width")
        if not self.consume("NODE_BLOCK"):
            return
        declared = self.consume_int(name="node_count")
        if declared is None:
            return
        self.node_count_declared = int(declared)
        for node_index in range(self.node_count_declared):
            if not self._parse_node(node_index):
                return
        if self.node_count_actual != self.node_count_declared:
            self.errors.append(f"node_count_mismatch_declared_{self.node_count_declared}_actual_{self.node_count_actual}")
            return
        for block_name in ("REL_BLOCK_INSERTED_IN", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO"):
            if not self._parse_pair_block(block_name):
                return
        if not self._parse_other_block():
            return
        if not self.consume("RESIDUALS"):
            return
        residual_count = self.consume_int(name="residual_count")
        if residual_count is None:
            return
        if residual_count != 0:
            self.errors.append(f"nonzero_residual_count_{residual_count}")
        if not self.consume("<EOS>"):
            return
        if self.index != len(self.tokens):
            self.errors.append(f"trailing_tokens_after_eos_{len(self.tokens) - self.index}")

    def _parse_node(self, node_index: int) -> bool:
        if not self.consume("NODE"):
            return False
        node: Dict[str, object] = {"index": int(node_index), "children": []}
        for name in ("role", "label", "renderable", "reference_only", "geometry_model", "geometry_ref"):
            if self.current() is None:
                self.errors.append(f"truncated_node_{node_index}_missing_{name}")
                return False
            if name in {"label", "renderable", "reference_only", "geometry_ref"}:
                value = self.consume_int(name=f"node_{node_index}_{name}")
                if value is None:
                    return False
                node[name] = int(value)
            else:
                node[name] = str(self.current())
                self.index += 1
        if self.current() == "CHILDREN":
            self.index += 1
            child_count = self.consume_int(name=f"node_{node_index}_child_count")
            if child_count is None:
                return False
            children: List[int] = []
            for child_slot in range(int(child_count)):
                child_index = self.consume_int(name=f"node_{node_index}_child_{child_slot}")
                if child_index is None:
                    return False
                children.append(int(child_index))
                if child_index < 0 or child_index >= self.node_count_declared:
                    self.errors.append(f"child_index_out_of_range_node_{node_index}_child_{child_index}")
            node["children"] = children
            node["has_children_field"] = True
        else:
            node["has_children_field"] = False
        if not self.consume("END_NODE"):
            return False
        self.nodes.append(node)
        self.node_count_actual += 1
        return True

    def _parse_pair_block(self, block_name: str) -> bool:
        if not self.consume(block_name):
            return False
        count = self.consume_int(name=f"{block_name}_count")
        if count is None:
            return False
        self.relation_counts[block_name] = int(count)
        pairs: List[tuple[int, int]] = []
        for pair_index in range(int(count)):
            left = self.consume_int(name=f"{block_name}_{pair_index}_left")
            right = self.consume_int(name=f"{block_name}_{pair_index}_right")
            if left is None or right is None:
                return False
            pairs.append((int(left), int(right)))
            for side, value in (("left", left), ("right", right)):
                if value < 0 or value >= self.node_count_declared:
                    self.errors.append(f"{block_name}_{pair_index}_{side}_endpoint_out_of_range_{value}")
        self.relation_pairs[block_name] = pairs
        return self.consume("END_BLOCK")

    def _parse_other_block(self) -> bool:
        if not self.consume("REL_BLOCK_OTHER"):
            return False
        count = self.consume_int(name="other_relation_count")
        if count is None:
            return False
        self.relation_counts["REL_BLOCK_OTHER"] = int(count)
        for relation_index in range(int(count)):
            if self.current() is None:
                self.errors.append(f"truncated_other_relation_{relation_index}_type")
                return False
            self.index += 1
            ref_count = self.consume_int(name=f"other_relation_{relation_index}_ref_count")
            if ref_count is None:
                return False
            for ref_index in range(int(ref_count)):
                endpoint = self.consume_int(name=f"other_relation_{relation_index}_ref_{ref_index}")
                if endpoint is None:
                    return False
                if endpoint < 0 or endpoint >= self.node_count_declared:
                    self.errors.append(f"other_relation_{relation_index}_endpoint_out_of_range_{endpoint}")
        return self.consume("END_BLOCK")

    def _validate_semantics(self) -> None:
        known_roles = {"ROLE_SUPPORT", "ROLE_DIVIDER", "ROLE_INSERT", "ROLE_INSERT_GROUP"}
        for node in self.nodes:
            index = int(node["index"])
            role = str(node.get("role"))
            renderable = int(node.get("renderable", -1))
            reference_only = int(node.get("reference_only", -1))
            geometry_model = str(node.get("geometry_model"))
            geometry_ref = int(node.get("geometry_ref", -1))
            has_children_field = bool(node.get("has_children_field", False))
            children = [int(value) for value in node.get("children", [])]

            if role not in known_roles:
                self.semantic_errors.append(f"node_{index}_unknown_role_{role}")
            if reference_only != 0:
                self.semantic_errors.append(f"node_{index}_reference_only_nonzero_{reference_only}")
            if role == "ROLE_INSERT_GROUP":
                if renderable != 0:
                    self.semantic_errors.append(f"insert_group_{index}_renderable_{renderable}")
                if geometry_model != "GEOM_NONE":
                    self.semantic_errors.append(f"insert_group_{index}_geometry_{geometry_model}")
                if geometry_ref != 0:
                    self.semantic_errors.append(f"insert_group_{index}_geometry_ref_{geometry_ref}")
                if not has_children_field:
                    self.semantic_errors.append(f"insert_group_{index}_missing_children")
                if not children:
                    self.semantic_errors.append(f"insert_group_{index}_empty_children")
            else:
                if has_children_field:
                    self.semantic_errors.append(f"node_{index}_{role}_unexpected_children")
                if renderable != 1:
                    self.semantic_errors.append(f"node_{index}_{role}_renderable_{renderable}")
                if geometry_model != "GEOM_POLYGON_CODE":
                    self.semantic_errors.append(f"node_{index}_{role}_geometry_{geometry_model}")
                if geometry_ref != 1:
                    self.semantic_errors.append(f"node_{index}_{role}_geometry_ref_{geometry_ref}")

        child_parent_count: Counter[int] = Counter()
        for node in self.nodes:
            parent_index = int(node["index"])
            role = str(node.get("role"))
            children = [int(value) for value in node.get("children", [])]
            if len(children) != len(set(children)):
                self.semantic_errors.append(f"insert_group_{parent_index}_duplicate_children")
            for child_index in children:
                if child_index == parent_index:
                    self.semantic_errors.append(f"insert_group_{parent_index}_self_child")
                    continue
                if 0 <= child_index < len(self.nodes):
                    child_role = str(self.nodes[child_index].get("role"))
                    if role != "ROLE_INSERT_GROUP":
                        self.semantic_errors.append(f"node_{parent_index}_{role}_child_{child_index}")
                    if child_role != "ROLE_INSERT":
                        self.semantic_errors.append(f"insert_group_{parent_index}_child_{child_index}_role_{child_role}")
                    child_parent_count[child_index] += 1
        for node in self.nodes:
            index = int(node["index"])
            if str(node.get("role")) == "ROLE_INSERT" and child_parent_count[index] != 1:
                self.semantic_errors.append(f"insert_{index}_parent_count_{child_parent_count[index]}")

        self._validate_pair_block_semantics(
            "REL_BLOCK_INSERTED_IN",
            allowed_left={"ROLE_INSERT_GROUP"},
            allowed_right={"ROLE_SUPPORT", "ROLE_INSERT_GROUP"},
            require_unique_left=True,
        )
        self._validate_pair_block_semantics(
            "REL_BLOCK_DIVIDES",
            allowed_left={"ROLE_DIVIDER"},
            allowed_right={"ROLE_SUPPORT", "ROLE_INSERT_GROUP"},
        )
        self._validate_pair_block_semantics(
            "REL_BLOCK_ADJACENT_TO",
            allowed_left={"ROLE_SUPPORT", "ROLE_INSERT_GROUP", "ROLE_INSERT"},
            allowed_right={"ROLE_SUPPORT", "ROLE_INSERT_GROUP", "ROLE_INSERT"},
            undirected=True,
        )

        inserted_left_counts = Counter(left for left, _right in self.relation_pairs.get("REL_BLOCK_INSERTED_IN", []))
        for node in self.nodes:
            index = int(node["index"])
            if str(node.get("role")) == "ROLE_INSERT_GROUP" and inserted_left_counts[index] != 1:
                self.semantic_errors.append(f"insert_group_{index}_inserted_in_count_{inserted_left_counts[index]}")
        if int(self.relation_counts.get("REL_BLOCK_OTHER", 0)) != 0:
            self.semantic_errors.append(f"nonzero_other_relation_count_{self.relation_counts.get('REL_BLOCK_OTHER')}")

    def _validate_pair_block_semantics(
        self,
        block_name: str,
        *,
        allowed_left: set[str],
        allowed_right: set[str],
        require_unique_left: bool = False,
        undirected: bool = False,
    ) -> None:
        pairs = self.relation_pairs.get(block_name, [])
        seen_pairs: set[tuple[int, int]] = set()
        left_counts: Counter[int] = Counter()
        for pair_index, (left, right) in enumerate(pairs):
            if left == right:
                self.semantic_errors.append(f"{block_name}_{pair_index}_self_relation_{left}")
            key = tuple(sorted((left, right))) if undirected else (left, right)
            if key in seen_pairs:
                self.semantic_errors.append(f"{block_name}_{pair_index}_duplicate_pair_{left}_{right}")
            seen_pairs.add(key)
            left_counts[left] += 1
            if 0 <= left < len(self.nodes):
                left_role = str(self.nodes[left].get("role"))
                if left_role not in allowed_left:
                    self.semantic_errors.append(f"{block_name}_{pair_index}_left_role_{left_role}")
            if 0 <= right < len(self.nodes):
                right_role = str(self.nodes[right].get("role"))
                if right_role not in allowed_right:
                    self.semantic_errors.append(f"{block_name}_{pair_index}_right_role_{right_role}")
        if require_unique_left:
            for left, count in left_counts.items():
                if count > 1:
                    self.semantic_errors.append(f"{block_name}_left_{left}_count_{count}")


def validate_topology_tokens(tokens: Sequence[str]) -> Dict[str, object]:
    return TopologyTokenParser(tokens).validate()


def validate_sample_row(row: dict) -> Dict[str, object]:
    result = validate_topology_tokens(row.get("tokens", []) or [])
    result.update(
        {
            "sample_index": row.get("sample_index"),
            "checkpoint": row.get("checkpoint"),
            "reported_hit_eos": bool(row.get("hit_eos", False)),
        }
    )
    return result


def validate_topology_sample_file(samples_path: Path, *, top_k: int = 20) -> Dict[str, object]:
    rows = []
    with Path(samples_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    results = [validate_sample_row(row) for row in rows]
    failures = [row for row in results if not row["valid"]]
    semantic_failures = [row for row in results if row["valid"] and not row["semantic_valid"]]
    failure_histogram: Counter[str] = Counter()
    for row in failures:
        if row["errors"]:
            failure_histogram[str(row["errors"][0])] += 1
        else:
            failure_histogram["unknown"] += 1
    semantic_failure_histogram: Counter[str] = Counter()
    for row in semantic_failures:
        if row["semantic_errors"]:
            semantic_failure_histogram[str(row["semantic_errors"][0])] += 1
        else:
            semantic_failure_histogram["unknown"] += 1
    lengths = [int(row["length"]) for row in results]
    return {
        "format": "maskgen_manual_topology_sample_validation_v1",
        "samples_path": str(Path(samples_path).as_posix()),
        "sample_count": int(len(results)),
        "hit_eos_count": int(sum(1 for row in results if row["hit_eos"])),
        "reported_hit_eos_count": int(sum(1 for row in results if row["reported_hit_eos"])),
        "valid_count": int(sum(1 for row in results if row["valid"])),
        "valid_rate": float(sum(1 for row in results if row["valid"]) / len(results)) if results else 0.0,
        "semantic_valid_count": int(sum(1 for row in results if row["semantic_valid"])),
        "semantic_valid_rate": float(sum(1 for row in results if row["semantic_valid"]) / len(results)) if results else 0.0,
        "length_mean": float(mean(lengths)) if lengths else 0.0,
        "length_max": int(max(lengths)) if lengths else 0,
        "failure_reason_histogram": dict(failure_histogram),
        "semantic_failure_reason_histogram": dict(semantic_failure_histogram),
        "invalid_samples": failures[: int(top_k)],
        "semantic_invalid_samples": semantic_failures[: int(top_k)],
    }


def write_topology_sample_validation_markdown(payload: dict, output_path: Path) -> None:
    lines = [
        "# Manual Topology Sample Validation",
        "",
        f"- samples: {payload['sample_count']}",
        f"- valid: {payload['valid_count']}",
        f"- valid_rate: {payload['valid_rate']:.4f}",
        f"- semantic_valid: {payload.get('semantic_valid_count', 0)}",
        f"- semantic_valid_rate: {float(payload.get('semantic_valid_rate', 0.0)):.4f}",
        f"- hit_eos: {payload['hit_eos_count']}",
        f"- reported_hit_eos: {payload['reported_hit_eos_count']}",
        f"- length mean / max: {payload['length_mean']:.2f} / {payload['length_max']}",
        "",
        "## Failure Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(payload["failure_reason_histogram"].items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Semantic Failure Reasons", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in sorted(
        (payload.get("semantic_failure_reason_histogram") or {}).items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Invalid Samples", "", "| sample_index | length | first_error |", "| ---: | ---: | --- |"])
    for row in payload.get("invalid_samples", []):
        error = row.get("errors", [""])[0] if row.get("errors") else ""
        lines.append(f"| {row.get('sample_index')} | {row.get('length')} | `{error}` |")
    lines.extend(
        ["", "## Semantic Invalid Samples", "", "| sample_index | length | first_semantic_error |", "| ---: | ---: | --- |"]
    )
    for row in payload.get("semantic_invalid_samples", []):
        error = row.get("semantic_errors", [""])[0] if row.get("semantic_errors") else ""
        lines.append(f"| {row.get('sample_index')} | {row.get('length')} | `{error}` |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
