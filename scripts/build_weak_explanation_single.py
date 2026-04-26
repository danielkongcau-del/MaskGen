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
from partition_gen.weak_explainer import WeakExplainerConfig, build_weak_explanation_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one weak convex-face-atom explanation JSON.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence-json", type=Path)
    source.add_argument("--global-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convex-backend", type=str, default="auto", choices=["auto", "cgal", "fallback_cdt_greedy", "fallback_hm"])
    parser.add_argument("--convex-cgal-cli", type=str, default=None)
    parser.add_argument("--convex-max-bridge-sets", type=int, default=256)
    parser.add_argument("--convex-cut-slit-scale", type=float, default=1e-6)
    parser.add_argument("--no-label-groups", action="store_true")
    parser.add_argument("--no-atom-nodes", action="store_true")
    parser.add_argument("--no-boundary-arcs", action="store_true")
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

    payload = build_weak_explanation_payload(
        evidence_payload,
        config=WeakExplainerConfig(
            include_label_groups=not args.no_label_groups,
            include_convex_atom_nodes=not args.no_atom_nodes,
            use_boundary_arcs_when_available=not args.no_boundary_arcs,
        ),
        source_tag=source_tag,
    )
    dump_json(args.output, payload)
    diagnostics = payload["diagnostics"]
    validation = payload["validation"]
    print(
        "built weak explanation: "
        f"faces={diagnostics['semantic_face_count']}, "
        f"atoms={diagnostics['convex_atom_count']}, "
        f"label_groups={diagnostics['label_group_count']}, "
        f"relations={diagnostics['relation_count']}, "
        f"residual_faces={diagnostics['residual_face_count']}, "
        f"code_length={diagnostics['total_code_length']:.3f}, "
        f"valid={validation['is_valid']}"
    )


if __name__ == "__main__":
    main()
