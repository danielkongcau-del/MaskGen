from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.operation_explainer import build_operation_explanation_payload  # noqa: E402
from partition_gen.operation_types import OperationExplainerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark operation-level explainer over evidence JSON files.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ortools-time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--max-patch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-patch", type=int, default=16)
    parser.add_argument("--cost-profile", type=str, default="token_length_v1", choices=["heuristic_v1", "token_length_v1"])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_evidence_paths(root: Path, split: str | None = None) -> Iterable[Path]:
    search_root = root / split if split and (root / split).exists() else root
    for path in sorted(search_root.rglob("*.json")):
        yield path


def benchmark_one(path: Path, config: OperationExplainerConfig) -> dict | None:
    payload = load_json(path)
    if payload.get("format") != "maskgen_explanation_evidence_v1":
        return None
    started = time.perf_counter()
    try:
        explanation = build_operation_explanation_payload(payload, config=config, source_tag=str(path.as_posix()))
        runtime_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = explanation["diagnostics"]
        validation = explanation["validation"]
        return {
            "source": str(path.as_posix()),
            "cost_profile": diagnostics.get("cost_profile"),
            "face_count": diagnostics.get("face_count"),
            "patch_count": diagnostics.get("patch_count"),
            "candidate_count": diagnostics.get("candidate_count"),
            "raw_candidate_count": diagnostics.get("raw_candidate_count"),
            "deduplicated_candidate_count": diagnostics.get("deduplicated_candidate_count"),
            "dropped_duplicate_count": diagnostics.get("dropped_duplicate_count"),
            "valid_candidate_count": explanation.get("candidate_summary", {}).get("valid_candidate_count"),
            "selected_operation_count": diagnostics.get("selected_operation_count"),
            "operation_histogram": diagnostics.get("operation_histogram", {}),
            "role_histogram": diagnostics.get("role_histogram", {}),
            "residual_face_count": diagnostics.get("residual_face_count"),
            "residual_area_ratio": diagnostics.get("residual_area_ratio"),
            "total_independent_cost": diagnostics.get("total_independent_cost"),
            "total_operation_cost": diagnostics.get("total_operation_cost"),
            "total_compression_gain": diagnostics.get("total_compression_gain"),
            "selection_method": diagnostics.get("selection_method"),
            "solver_status": diagnostics.get("solver_status"),
            "global_optimal": diagnostics.get("global_optimal"),
            "validation_is_valid": validation.get("is_valid"),
            "false_cover_area_total": diagnostics.get("false_cover_area_total"),
            "false_cover_ratio_max": diagnostics.get("false_cover_ratio_max"),
            "hard_false_cover_candidate_count": sum(
                1
                for candidate in explanation.get("candidate_summary", {}).get("top_candidates", [])
                if candidate.get("false_cover", {}).get("hard_invalid")
            ),
            "failure_reasons": diagnostics.get("failure_reasons", []),
            "runtime_ms": runtime_ms,
        }
    except Exception as exc:
        return {
            "source": str(path.as_posix()),
            "error": str(exc),
            "validation_is_valid": False,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        }


def run_benchmark(
    evidence_root: Path,
    *,
    split: str | None = None,
    max_samples: int | None = None,
    config: OperationExplainerConfig,
) -> List[dict]:
    rows: List[dict] = []
    for path in iter_evidence_paths(evidence_root, split=split):
        row = benchmark_one(path, config)
        if row is None:
            continue
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def main() -> None:
    args = parse_args()
    config = OperationExplainerConfig(
        max_patch_size=args.max_patch_size,
        max_candidates_per_patch=args.max_candidates_per_patch,
        ortools_time_limit_seconds=args.ortools_time_limit_seconds,
        cost_profile=args.cost_profile,
    )
    rows = run_benchmark(args.evidence_root, split=args.split, max_samples=args.max_samples, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
