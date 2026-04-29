from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_parse_graph_relation_spatial_audit import (  # noqa: E402
    audit_manual_parse_graph_targets_relation_spatial,
)
from partition_gen.manual_parse_graph_target_audit import iter_manual_parse_graph_target_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether parse-graph semantic relations are spatially plausible in rendered geometry."
    )
    parser.add_argument("--target-root", type=Path, required=True, help="A target JSON, graphs directory, or output root with manifest.jsonl.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-containment-ratio", type=float, default=0.8)
    parser.add_argument("--max-adjacent-gap", type=float, default=4.0)
    parser.add_argument("--max-adjacent-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--min-divider-target-intersection-ratio", type=float, default=0.3)
    parser.add_argument("--max-divider-target-area-ratio", type=float, default=1.25)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _stats_text(stats: dict) -> str:
    if int(stats.get("count", 0)) == 0:
        return ""
    return f"{stats.get('median')} / {stats.get('p90')} / {stats.get('max')}"


def write_summary_md(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(payload["relation_pair_count"])
    passed = int(payload["passed_relation_pair_count"])
    failed = int(payload["failed_relation_pair_count"])
    lines = [
        "# Manual Parse Graph Relation Spatial Audit",
        "",
        f"- target root: `{payload['target_root']}`",
        f"- loaded: {payload['loaded_count']} / {payload['input_path_count']}",
        f"- relation pairs: {total}",
        f"- passed: {passed} ({_ratio(passed, total):.4f})",
        f"- failed: {failed} ({_ratio(failed, total):.4f})",
        f"- thresholds: `{json.dumps(payload['thresholds'], ensure_ascii=False)}`",
        "",
        "## Relation Types",
        "",
        "| type | count | passed | failed | pass_ratio | overlap median/p90/max | gap median/p90/max |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for relation_type, metrics in payload["relation_type_metrics"].items():
        lines.append(
            f"| {relation_type} | {metrics['count']} | {metrics['passed_count']} | "
            f"{metrics['failed_count']} | {metrics['pass_ratio']:.4f} | "
            f"{_stats_text(metrics['left_intersection_ratio_stats'])} | {_stats_text(metrics['bbox_gap_stats'])} |"
        )

    lines.extend(["", "## Failure Reasons", "", "| reason | count |", "| --- | ---: |"])
    for key, value in payload["failure_reason_histogram"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Role Pair Failures", "", "| pair | count |", "| --- | ---: |"])
    for key, value in payload["role_pair_failure_histogram"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Top Failures",
            "",
            "| sample | type | left | right | reasons | overlap | gap | severity | source |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["top_failures"]:
        lines.append(
            f"| {row.get('sample_index')} | {row.get('relation_type')} | {row.get('left_id')} | "
            f"{row.get('right_id')} | {','.join(row.get('failure_reasons', []))} | "
            f"{row.get('left_intersection_ratio')} | {row.get('bbox_gap')} | "
            f"{row.get('severity')} | {row.get('source')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]
    payload = audit_manual_parse_graph_targets_relation_spatial(
        paths,
        min_containment_ratio=float(args.min_containment_ratio),
        max_adjacent_gap=float(args.max_adjacent_gap),
        max_adjacent_overlap_ratio=float(args.max_adjacent_overlap_ratio),
        min_divider_target_intersection_ratio=float(args.min_divider_target_intersection_ratio),
        max_divider_target_area_ratio=float(args.max_divider_target_area_ratio),
        top_k=int(args.top_k),
    )
    payload["target_root"] = str(args.target_root.as_posix())
    dump_json(args.output_json, payload)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"relation-spatial-audited targets={payload['loaded_count']} "
        f"pairs={payload['relation_pair_count']} passed={payload['passed_relation_pair_count']} "
        f"failed={payload['failed_relation_pair_count']} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
