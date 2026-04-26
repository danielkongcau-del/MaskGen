from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.operation_explainer import build_operation_explanation_payload  # noqa: E402
from partition_gen.operation_role_spec import load_role_spec  # noqa: E402
from partition_gen.operation_types import OperationExplainerConfig  # noqa: E402
from partition_gen.manual_rule_explainer import (  # noqa: E402
    ManualRuleExplainerConfig,
    build_manual_rule_explanation_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one operation-level explanation JSON.")
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-spec", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="operation", choices=["operation", "manual_rule"])
    parser.add_argument("--max-patch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-patch", type=int, default=16)
    parser.add_argument("--min-compression-gain", type=float, default=0.0)
    parser.add_argument("--ortools-time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--cost-profile", type=str, default="token_length_v1", choices=["heuristic_v1", "token_length_v1"])
    parser.add_argument("--disable-overlay-insert", action="store_true")
    parser.add_argument("--disable-divide-by-region", action="store_true")
    parser.add_argument("--disable-parallel-supports", action="store_true")
    parser.add_argument("--disable-residual", action="store_true")
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument("--allow-greedy-fallback", dest="allow_greedy_fallback", action="store_true", default=True)
    fallback.add_argument("--disable-greedy-fallback", dest="allow_greedy_fallback", action="store_false")
    parser.add_argument("--manual-include-all-faces-of-support-labels", action="store_true")
    parser.add_argument("--manual-include-all-faces-of-divider-labels", action="store_true")
    parser.add_argument("--manual-min-shared-length", type=float, default=0.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    evidence = load_json(args.evidence_json)
    role_spec = load_role_spec(args.role_spec) if args.role_spec else None
    if args.mode == "manual_rule":
        if role_spec is None:
            raise ValueError("--mode manual_rule requires --role-spec")
        payload = build_manual_rule_explanation_payload(
            evidence,
            role_spec,
            config=ManualRuleExplainerConfig(
                include_all_faces_of_support_labels=args.manual_include_all_faces_of_support_labels,
                include_all_faces_of_divider_labels=args.manual_include_all_faces_of_divider_labels,
                min_shared_length=args.manual_min_shared_length,
            ),
            source_tag=str(args.evidence_json.as_posix()),
        )
    else:
        payload = build_operation_explanation_payload(
            evidence,
            role_spec_payload=role_spec,
            config=OperationExplainerConfig(
                max_patch_size=args.max_patch_size,
                max_candidates_per_patch=args.max_candidates_per_patch,
                min_compression_gain=args.min_compression_gain,
                ortools_time_limit_seconds=args.ortools_time_limit_seconds,
                cost_profile=args.cost_profile,
                enable_overlay_insert=not args.disable_overlay_insert,
                enable_divide_by_region=not args.disable_divide_by_region,
                enable_parallel_supports=not args.disable_parallel_supports,
                enable_residual=not args.disable_residual,
                allow_greedy_fallback=args.allow_greedy_fallback,
            ),
            source_tag=str(args.evidence_json.as_posix()),
        )
    dump_json(args.output, payload)
    diagnostics = payload["diagnostics"]
    validation = payload["validation"]
    if args.mode == "manual_rule":
        print(
            "built manual rule explanation: "
            f"faces={diagnostics['face_count']}, "
            f"nodes={diagnostics['node_count']}, "
            f"relations={diagnostics['relation_count']}, "
            f"residual_faces={diagnostics['residual_face_count']}, "
            f"residual_area_ratio={diagnostics['residual_area_ratio']:.6f}, "
            f"duplicate_owned_faces={diagnostics['duplicate_owned_face_count']}, "
            f"valid={validation['is_valid']}"
        )
    else:
        print(
            "built operation explanation: "
            f"faces={diagnostics['face_count']}, "
            f"patches={diagnostics['patch_count']}, "
            f"candidates={diagnostics['candidate_count']}, "
            f"selected_operations={diagnostics['selected_operation_count']}, "
            f"residual_faces={diagnostics['residual_face_count']}, "
            f"residual_area_ratio={diagnostics['residual_area_ratio']:.6f}, "
            f"total_compression_gain={diagnostics['total_compression_gain']:.3f}, "
            f"selection_method={diagnostics['selection_method']}, "
            f"global_optimal={diagnostics['global_optimal']}, "
            f"valid={validation['is_valid']}"
        )


if __name__ == "__main__":
    main()
