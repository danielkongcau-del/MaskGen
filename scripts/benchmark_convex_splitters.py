from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    build_bridged_convex_partition_from_geometry_payload,
)
from partition_gen.convex_partition import ConvexMergeConfig, build_convex_partition_from_geometry_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CDT-greedy and bridged CGAL convex splitters.")
    parser.add_argument("--approx-root", type=Path, default=Path("data/remote_256_geometry_approx_debug"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/convex_splitter_benchmark_val.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cgal", "fallback_hm", "fallback_cdt_greedy"])
    parser.add_argument("--cgal-cli", type=str, default=None)
    parser.add_argument("--cut-slit-scales", type=float, nargs="+", default=[1e-7, 1e-6, 1e-5])
    parser.add_argument("--max-bridge-sets", type=int, default=256)
    parser.add_argument("--vertex-round-digits", type=int, default=8)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--validity-eps", type=float, default=1e-7)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_approx_jsons(root: Path, split: str) -> Iterable[Path]:
    preferred = root / split / "graphs"
    if preferred.exists():
        yield from sorted(preferred.glob("*.json"))
        return
    split_root = root / split
    if split_root.exists():
        yield from sorted(split_root.rglob("*.json"))
        return
    yield from sorted(root.rglob("*.json"))


def is_geometry_payload(payload: dict) -> bool:
    approx = payload.get("approx_geometry")
    return isinstance(approx, dict) and "outer" in approx and "face_id" in payload


def infer_stem(path: Path, payload: dict) -> str | None:
    name = path.stem
    if "_face" in name:
        return name.split("_face", 1)[0]
    source_mask = str(payload.get("source_mask") or "")
    if source_mask:
        return Path(source_mask).stem
    source_graph = str(payload.get("source_partition_graph") or "")
    if source_graph:
        return Path(source_graph).stem
    return name or None


def approx_vertex_count(payload: dict) -> int:
    if "approx_vertex_count" in payload:
        return int(payload["approx_vertex_count"])
    approx = payload.get("approx_geometry") or {}
    return int(len(approx.get("outer") or []))


def count_holes(payload: dict) -> int:
    approx = payload.get("approx_geometry") or {}
    return int(len(approx.get("holes") or []))


def runtime_ms_since(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def safe_get(mapping: dict, key: str, default=None):
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def make_row_base(path: Path, payload: dict, split: str) -> dict:
    return {
        "source_file": str(path.as_posix()),
        "split": split,
        "stem": infer_stem(path, payload),
        "face_id": int(payload.get("face_id", -1)),
        "label": int(payload.get("label", -1)),
        "hole_count": count_holes(payload),
        "approx_vertex_count": approx_vertex_count(payload),
    }


def benchmark_one_scale(
    path: Path,
    payload: dict,
    *,
    split: str,
    baseline_payload: dict | None,
    baseline_runtime_ms: float | None,
    baseline_failure: str | None,
    args: argparse.Namespace,
    cut_slit_scale: float,
) -> dict:
    row = make_row_base(path, payload, split)
    row.update(
        {
            "cut_slit_scale": float(cut_slit_scale),
            "baseline_piece_count": None,
            "baseline_runtime_ms": baseline_runtime_ms,
            "bridged_piece_count": None,
            "piece_reduction": None,
            "backend": None,
            "valid_partition": False,
            "simple_polygon_optimal": False,
            "global_optimal": False,
            "optimal_scope": None,
            "bridge_policy": None,
            "bridge_candidate_count": None,
            "available_bridge_set_count": None,
            "evaluated_bridge_set_count": None,
            "rejected_bridge_set_count": None,
            "simple_polygon_vertex_count": None,
            "validation_iou": None,
            "overlap_area": None,
            "all_convex": False,
            "runtime_ms": None,
            "failure_reason": baseline_failure,
        }
    )

    if baseline_payload is not None:
        row["baseline_piece_count"] = int(baseline_payload.get("final_primitive_count", 0))

    start = time.perf_counter()
    try:
        bridged = build_bridged_convex_partition_from_geometry_payload(
            payload,
            config=BridgedPartitionConfig(
                max_bridge_sets=args.max_bridge_sets,
                vertex_round_digits=args.vertex_round_digits,
                area_eps=args.area_eps,
                validity_eps=args.validity_eps,
                backend=args.backend,
                cgal_cli=args.cgal_cli,
                cut_slit_scale=float(cut_slit_scale),
            ),
            source_tag=str(path.as_posix()),
        )
    except Exception as error:  # noqa: BLE001 - benchmark rows must capture failures.
        row["runtime_ms"] = runtime_ms_since(start)
        row["failure_reason"] = "; ".join(value for value in [baseline_failure, f"bridged: {error}"] if value)
        return row

    row["runtime_ms"] = runtime_ms_since(start)
    backend_info = bridged.get("backend_info") or {}
    validation = bridged.get("validation") or {}
    row.update(
        {
            "bridged_piece_count": int(bridged.get("final_primitive_count", 0)),
            "backend": safe_get(backend_info, "backend"),
            "valid_partition": bool(validation.get("is_valid", False)),
            "simple_polygon_optimal": bool(safe_get(backend_info, "simple_polygon_optimal", False)),
            "global_optimal": bool(safe_get(backend_info, "global_optimal", False)),
            "optimal_scope": safe_get(backend_info, "optimal_scope"),
            "bridge_policy": bridged.get("bridge_policy"),
            "bridge_candidate_count": int(len(bridged.get("bridge_candidates") or [])),
            "available_bridge_set_count": int(bridged.get("available_bridge_set_count", 0)),
            "evaluated_bridge_set_count": int(safe_get(backend_info, "evaluated_bridge_set_count", 0) or 0),
            "rejected_bridge_set_count": int(safe_get(backend_info, "rejected_bridge_set_count", 0) or 0),
            "simple_polygon_vertex_count": int(bridged.get("simple_polygon_vertex_count", 0)),
            "validation_iou": float(validation.get("iou", 0.0)),
            "overlap_area": float(validation.get("overlap_area", 0.0)),
            "all_convex": bool(validation.get("all_convex", False)),
        }
    )
    if row["baseline_piece_count"] is not None:
        row["piece_reduction"] = int(row["baseline_piece_count"]) - int(row["bridged_piece_count"])
    if row["failure_reason"] is None and not row["valid_partition"]:
        row["failure_reason"] = "invalid bridged partition"
    return row


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in iter_approx_jsons(args.approx_root, args.split):
            try:
                payload = load_json(path)
            except Exception:
                continue
            if not is_geometry_payload(payload):
                continue
            if args.max_samples is not None and written >= args.max_samples * len(args.cut_slit_scales):
                break

            baseline_payload = None
            baseline_failure = None
            baseline_start = time.perf_counter()
            try:
                baseline_payload = build_convex_partition_from_geometry_payload(
                    payload,
                    config=ConvexMergeConfig(),
                    source_tag=str(path.as_posix()),
                )
            except Exception as error:  # noqa: BLE001 - benchmark rows must capture failures.
                baseline_failure = f"baseline: {error}"
            baseline_runtime_ms = runtime_ms_since(baseline_start)

            for scale in args.cut_slit_scales:
                row = benchmark_one_scale(
                    path,
                    payload,
                    split=args.split,
                    baseline_payload=baseline_payload,
                    baseline_runtime_ms=baseline_runtime_ms,
                    baseline_failure=baseline_failure,
                    args=args,
                    cut_slit_scale=scale,
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                print(
                    f"{row['source_file']} scale={scale:g} baseline={row['baseline_piece_count']} "
                    f"bridged={row['bridged_piece_count']} backend={row['backend']} "
                    f"iou={row['validation_iou']} failure={row['failure_reason']}"
                )

    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
