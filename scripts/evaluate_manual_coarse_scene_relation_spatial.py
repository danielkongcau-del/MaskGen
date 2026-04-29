from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_manual_parse_graph_relation_spatial import write_summary_md  # noqa: E402
from partition_gen.manual_parse_graph_relation_spatial_audit import audit_manual_parse_graph_targets_relation_spatial  # noqa: E402
from partition_gen.manual_parse_graph_target_audit import iter_manual_parse_graph_target_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coarse-scene outputs with the relation spatial audit.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline-overall-pass-ratio", type=float, default=0.4048)
    parser.add_argument("--baseline-adjacent-pass-ratio", type=float, default=0.0479)
    parser.add_argument("--baseline-divides-pass-ratio", type=float, default=0.1623)
    parser.add_argument("--baseline-inserted-pass-ratio", type=float, default=0.4909)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _pass_ratio(payload: dict, relation_type: str) -> float:
    metrics = (payload.get("relation_type_metrics", {}) or {}).get(relation_type, {}) or {}
    return float(metrics.get("pass_ratio", 0.0))


def main() -> None:
    args = parse_args()
    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]
    payload = audit_manual_parse_graph_targets_relation_spatial(paths, top_k=int(args.top_k))
    total = int(payload.get("relation_pair_count", 0))
    passed = int(payload.get("passed_relation_pair_count", 0))
    overall = float(passed / total) if total else 0.0
    payload["target_root"] = str(args.target_root.as_posix())
    payload["coarse_scene_baseline_comparison"] = {
        "overall_pass_ratio": overall,
        "overall_delta": float(overall - float(args.baseline_overall_pass_ratio)),
        "adjacent_to_delta": float(_pass_ratio(payload, "adjacent_to") - float(args.baseline_adjacent_pass_ratio)),
        "divides_delta": float(_pass_ratio(payload, "divides") - float(args.baseline_divides_pass_ratio)),
        "inserted_in_delta": float(_pass_ratio(payload, "inserted_in") - float(args.baseline_inserted_pass_ratio)),
        "success_thresholds": {
            "overall_delta_min": 0.15,
            "adjacent_to_delta_min": 0.20,
            "divides_delta_min": 0.20,
            "inserted_in_delta_min": 0.15,
        },
    }
    comparison = payload["coarse_scene_baseline_comparison"]
    comparison["passes_first_round_threshold"] = bool(
        comparison["overall_delta"] >= 0.15
        and comparison["adjacent_to_delta"] >= 0.20
        and comparison["divides_delta"] >= 0.20
        and comparison["inserted_in_delta"] >= 0.15
    )
    dump_json(args.output_json, payload)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"coarse-scene relation-spatial targets={payload['loaded_count']} "
        f"pairs={payload['relation_pair_count']} passed={payload['passed_relation_pair_count']} "
        f"overall={overall:.4f} threshold_pass={comparison['passes_first_round_threshold']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
