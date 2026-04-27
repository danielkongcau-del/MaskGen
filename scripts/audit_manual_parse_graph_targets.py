from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_parse_graph_target_audit import (  # noqa: E402
    audit_manual_parse_graph_targets,
    iter_manual_parse_graph_target_paths,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit full manual parse-graph targets produced by downstream smoke tests.")
    parser.add_argument("--target-root", type=Path, required=True, help="A target JSON, graphs directory, or output root with manifest.jsonl.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-int", type=int, default=4096)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_summary_md(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Parse Graph Target Audit",
        "",
        f"- loaded: {payload['loaded_count']} / {payload['input_path_count']}",
        f"- encodable: {payload['encodable_count']}",
        f"- old encodable: {payload['old_encodable_count']}",
        f"- compact encodable: {payload['compact_encodable_count']}",
        f"- missing geometry payloads: {payload['missing_geometry_payload_count']}",
        f"- placeholder geometry sources: {payload['placeholder_geometry_source_count']}",
        "",
        "| metric | mean | p95 | max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("node_counts", "relation_counts", "old_token_counts", "compact_token_counts"):
        stats = payload[key]
        lines.append(f"| {key} | {stats['mean']} | {stats['p95']} | {stats['max']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]
    config = ParseGraphTokenizerConfig(max_int=int(args.max_int))
    payload = audit_manual_parse_graph_targets(paths, config=config)
    payload["target_root"] = str(args.target_root.as_posix())
    dump_json(args.output_json, payload)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"audited targets={payload['loaded_count']} "
        f"encodable={payload['encodable_count']} "
        f"missing_geometry={payload['missing_geometry_payload_count']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
