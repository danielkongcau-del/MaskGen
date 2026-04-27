from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * float(q)))
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _length_stats(values: Sequence[int]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": int(max(values)),
    }


def _token_prefix(token: str) -> str:
    if token.startswith("I_"):
        return "I"
    if token.startswith("Q_"):
        return "Q"
    if token.startswith("ROLE_"):
        return "ROLE"
    if token.startswith("REL_"):
        return "REL"
    if token.startswith("GEOM_"):
        return "GEOM"
    return token


def _sequence_summary(row: dict) -> Dict[str, object]:
    tokens = row.get("tokens") or []
    return {
        "stem": row.get("stem"),
        "source_target": row.get("source_target"),
        "source_node_id": row.get("source_node_id"),
        "tokenizer": row.get("tokenizer"),
        "length": int(row.get("length", len(tokens))),
    }


def _collect_rows(path: Path, *, top_k: int) -> Dict[str, object]:
    lengths: List[int] = []
    token_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    top_sequences: List[Dict[str, object]] = []
    unknown_token_count = 0
    rows = list(_read_jsonl(path))
    for row in rows:
        tokens = [str(token) for token in row.get("tokens", [])]
        length = int(row.get("length", len(tokens)))
        lengths.append(length)
        token_counts.update(tokens)
        prefix_counts.update(_token_prefix(token) for token in tokens)
        unknown_token_count += sum(1 for token in tokens if token == "<UNK>")
        top_sequences.append(_sequence_summary(row))
    top_sequences.sort(key=lambda item: int(item["length"]), reverse=True)
    return {
        "path": str(path.as_posix()),
        "sequence_count": int(len(rows)),
        "lengths": _length_stats(lengths),
        "over_512": int(sum(1 for value in lengths if int(value) > 512)),
        "over_1024": int(sum(1 for value in lengths if int(value) > 1024)),
        "over_2048": int(sum(1 for value in lengths if int(value) > 2048)),
        "over_4096": int(sum(1 for value in lengths if int(value) > 4096)),
        "unknown_token_count": int(unknown_token_count),
        "top_sequences": top_sequences[: int(top_k)],
        "top_tokens": [{"token": token, "count": int(count)} for token, count in token_counts.most_common(int(top_k))],
        "token_prefix_histogram": {key: int(value) for key, value in sorted(prefix_counts.items())},
    }


def audit_manual_split_tokens(token_root: Path, *, top_k: int = 20) -> Dict[str, object]:
    token_root = Path(token_root)
    topology = _collect_rows(token_root / "topology_sequences.jsonl", top_k=top_k)
    geometry = _collect_rows(token_root / "geometry_sequences.jsonl", top_k=top_k)
    manifest_rows = list(_read_jsonl(token_root / "manifest.jsonl"))
    summary_path = token_root / "summary.json"
    tokenized_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    topology_max = topology["lengths"]["max"] or 0
    geometry_max = geometry["lengths"]["max"] or 0
    return {
        "format": "maskgen_manual_split_token_audit_v1",
        "token_root": str(token_root.as_posix()),
        "tokenized_summary": tokenized_summary,
        "manifest_sample_count": int(len(manifest_rows)),
        "topology": topology,
        "geometry": geometry,
        "max_single_sequence_tokens": int(max(int(topology_max), int(geometry_max))),
        "sequences_over_1024": int(topology["over_1024"] + geometry["over_1024"]),
    }


def write_manual_split_token_audit_markdown(audit: dict, output_path: Path) -> None:
    def section(kind: str) -> List[str]:
        item = audit[kind]
        lengths = item["lengths"]
        lines = [
            f"## {kind.title()}",
            "",
            f"- sequences: {item['sequence_count']}",
            f"- mean / p95 / max: {lengths['mean']} / {lengths['p95']} / {lengths['max']}",
            f"- over 512 / 1024 / 2048 / 4096: {item['over_512']} / {item['over_1024']} / {item['over_2048']} / {item['over_4096']}",
            f"- unknown tokens: {item['unknown_token_count']}",
            "",
            "### Longest sequences",
            "",
            "| rank | stem | source_node_id | length |",
            "| ---: | --- | --- | ---: |",
        ]
        for index, row in enumerate(item["top_sequences"], start=1):
            lines.append(f"| {index} | {row.get('stem')} | {row.get('source_node_id') or ''} | {row.get('length')} |")
        return lines

    lines = [
        "# Manual Split Token Audit",
        "",
        f"- token_root: `{audit['token_root']}`",
        f"- manifest samples: {audit['manifest_sample_count']}",
        f"- max single sequence tokens: {audit['max_single_sequence_tokens']}",
        "",
    ]
    lines.extend(section("topology"))
    lines.append("")
    lines.extend(section("geometry"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
