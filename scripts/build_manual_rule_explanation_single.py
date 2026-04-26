from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_rule_explainer import (  # noqa: E402
    ManualRuleExplainerConfig,
    build_manual_rule_explanation_payload,
)
from partition_gen.operation_role_spec import load_role_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one manual role-spec parse graph explanation JSON.")
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--role-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-all-faces-of-support-labels", action="store_true")
    parser.add_argument("--include-all-faces-of-divider-labels", action="store_true")
    parser.add_argument("--min-shared-length", type=float, default=0.0)
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
    role_spec = load_role_spec(args.role_spec)
    payload = build_manual_rule_explanation_payload(
        evidence,
        role_spec,
        config=ManualRuleExplainerConfig(
            include_all_faces_of_support_labels=args.include_all_faces_of_support_labels,
            include_all_faces_of_divider_labels=args.include_all_faces_of_divider_labels,
            min_shared_length=args.min_shared_length,
        ),
        source_tag=str(args.evidence_json.as_posix()),
    )
    dump_json(args.output, payload)
    diagnostics = payload["diagnostics"]
    validation = payload["validation"]
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


if __name__ == "__main__":
    main()
