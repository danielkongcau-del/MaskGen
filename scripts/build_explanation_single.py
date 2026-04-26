from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.explainer import ExplainerConfig, build_explanation_payload  # noqa: E402
from partition_gen.explanation_evidence import (  # noqa: E402
    ExplanationEvidenceConfig,
    build_explanation_evidence_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one initial explanation JSON.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence-json", type=Path)
    source.add_argument("--global-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-backend", type=str, default="auto", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=256)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--max-role-candidates-per-face", type=int, default=4)
    parser.add_argument("--disable-pairwise-label-relations", action="store_true")
    parser.add_argument("--disable-label-role-consistency", action="store_true")
    parser.add_argument("--label-consistency-min-faces", type=int, default=2)
    parser.add_argument("--label-consistency-penalty", type=float, default=None)
    parser.add_argument("--pair-relation-min-shared-length", type=float, default=1e-6)
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
    if args.evidence_json:
        evidence_payload = load_json(args.evidence_json)
        source_tag = str(args.evidence_json.as_posix())
    else:
        global_payload = load_json(args.global_json)
        evidence_payload = build_explanation_evidence_payload(
            global_payload,
            config=ExplanationEvidenceConfig(
                convex_backend=args.convex_backend,
                convex_cgal_cli=args.convex_cgal_cli,
                convex_max_bridge_sets=args.convex_max_bridge_sets,
                convex_cut_slit_scale=args.convex_cut_slit_scale,
            ),
            source_tag=str(args.global_json.as_posix()),
        )
        source_tag = str(args.global_json.as_posix())

    payload = build_explanation_payload(
        evidence_payload,
        config=ExplainerConfig(
            max_role_candidates_per_face=args.max_role_candidates_per_face,
            enable_pairwise_label_relations=not args.disable_pairwise_label_relations,
            enable_label_role_consistency=not args.disable_label_role_consistency,
            label_consistency_min_faces=args.label_consistency_min_faces,
            label_consistency_penalty=args.label_consistency_penalty,
            pair_relation_min_shared_length=args.pair_relation_min_shared_length,
        ),
        source_tag=source_tag,
    )
    dump_json(args.output, payload)
    graph = payload["generator_target"]["parse_graph"]
    diagnostics = payload["diagnostics"]
    validation = payload["validation"]
    print(
        "built explanation: "
        f"nodes={len(graph['nodes'])}, relations={len(graph['relations'])}, "
        f"residuals={len(graph['residuals'])}, "
        f"code_length={diagnostics['total_code_length']:.3f}, "
        f"valid={validation['is_valid']}, "
        f"selection={diagnostics['selection_method']}"
    )


if __name__ == "__main__":
    main()
