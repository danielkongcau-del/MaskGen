from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.pairwise_relation_explainer import (  # noqa: E402
    PairwiseRelationConfig,
    build_pairwise_relation_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze binary label-pair relation explanations for one evidence JSON.")
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-backend", type=str, default="fallback_cdt_greedy", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=128)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--min-shared-length", type=float, default=1e-6)
    parser.add_argument("--max-false-cover-ratio", type=float, default=0.12)
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
    payload = build_pairwise_relation_payload(
        evidence,
        config=PairwiseRelationConfig(
            convex_backend=args.convex_backend,
            convex_cgal_cli=args.convex_cgal_cli,
            convex_max_bridge_sets=args.convex_max_bridge_sets,
            convex_cut_slit_scale=args.convex_cut_slit_scale,
            min_shared_length=args.min_shared_length,
            max_false_cover_ratio=args.max_false_cover_ratio,
        ),
        source_tag=str(args.evidence_json.as_posix()),
    )
    dump_json(args.output, payload)
    print(
        "built pairwise relations: "
        f"pairs={payload['statistics']['pair_count']}, "
        f"labels={payload['statistics']['label_count']}, "
        f"preferred={payload['preferred_role_by_label']}"
    )
    for pair in payload["pairs"]:
        selected = pair["selected"]
        print(
            f"  labels={pair['labels']} shared={pair['shared_length']:.3f} "
            f"template={selected['template']} fill={selected['fill_policy']} "
            f"roles={selected['roles']} cost={selected['cost']:.3f}"
        )


if __name__ == "__main__":
    main()
