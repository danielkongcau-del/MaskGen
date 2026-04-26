from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.explanation_evidence import (  # noqa: E402
    ExplanationEvidenceConfig,
    build_explanation_evidence_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one explanation evidence JSON from a global approximation JSON.")
    parser.add_argument("--global-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-backend", type=str, default="auto", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=256)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--thin-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--compactness-threshold", type=float, default=0.45)
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
    global_payload = load_json(args.global_json)
    payload = build_explanation_evidence_payload(
        global_payload,
        config=ExplanationEvidenceConfig(
            convex_backend=args.convex_backend,
            convex_cgal_cli=args.convex_cgal_cli,
            convex_max_bridge_sets=args.convex_max_bridge_sets,
            convex_cut_slit_scale=args.convex_cut_slit_scale,
            thin_aspect_ratio=args.thin_aspect_ratio,
            compactness_threshold=args.compactness_threshold,
        ),
        source_tag=str(args.global_json.as_posix()),
    )
    dump_json(args.output, payload)
    validation = payload["evidence_validation"]
    stats = payload["statistics"]
    print(
        "built evidence: "
        f"faces={validation['face_count']}, arcs={validation['arc_count']}, "
        f"adjacency={validation['adjacency_count']}, "
        f"convex_success={validation['convex_success_count']}, "
        f"convex_failure={validation['convex_failure_count']}, "
        f"atoms={stats['total_convex_atom_count']}, "
        f"usable_for_explainer={validation['usable_for_explainer']}"
    )


if __name__ == "__main__":
    main()
