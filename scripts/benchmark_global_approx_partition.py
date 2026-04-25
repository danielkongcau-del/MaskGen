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

from partition_gen.bridged_convex_partition import (  # noqa: E402
    BridgedPartitionConfig,
    build_bridged_convex_partition_from_geometry_payload,
)
from partition_gen.dual_graph import load_json  # noqa: E402
from partition_gen.global_approx_partition import (  # noqa: E402
    GlobalApproxConfig,
    build_global_approx_partition_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full-image shared-arc geometry approximation.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/global_approx_benchmark_val.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--face-simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--face-area-epsilon", type=float, default=1e-3)
    parser.add_argument("--validity-eps", type=float, default=1e-6)
    parser.add_argument("--convex-smoke", action="store_true")
    parser.add_argument("--convex-smoke-max-faces", type=int, default=None)
    return parser.parse_args()


def iter_graph_paths(root: Path, split: str) -> Iterable[Path]:
    graph_dir = root / split / "graphs"
    yield from sorted(graph_dir.glob("*.json"), key=lambda path: (len(path.stem), path.stem))


def runtime_ms_since(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def convex_smoke(payload: dict, *, max_faces: int | None = None) -> dict:
    ok = 0
    failures = []
    faces = payload.get("faces", [])
    if max_faces is not None:
        faces = faces[: max(0, int(max_faces))]
    start = time.perf_counter()
    for face in faces:
        geometry_payload = {
            "source_partition_graph": payload.get("source_partition_graph"),
            "face_id": int(face["id"]),
            "label": int(face["label"]),
            "bbox": [int(value) for value in face["bbox"]],
            "approx_geometry": {"outer": face["outer"], "holes": face.get("holes", [])},
        }
        try:
            result = build_bridged_convex_partition_from_geometry_payload(
                geometry_payload,
                config=BridgedPartitionConfig(backend="fallback_cdt_greedy"),
                source_tag=str(payload.get("source_partition_graph") or ""),
            )
            if result["validation"]["is_valid"]:
                ok += 1
            else:
                failures.append({"face_id": int(face["id"]), "reason": "invalid", "validation": result["validation"]})
        except Exception as error:  # noqa: BLE001 - benchmark rows must capture failures.
            failures.append({"face_id": int(face["id"]), "reason": type(error).__name__, "message": str(error)})
    return {
        "enabled": True,
        "checked_face_count": int(len(faces)),
        "success_count": int(ok),
        "failure_count": int(len(failures)),
        "success": bool(ok == len(faces) and not failures),
        "runtime_ms": runtime_ms_since(start),
        "failures": failures[:16],
    }


def label_counts(payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for face in payload.get("faces", []):
        key = str(int(face["label"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def make_row(path: Path, split: str, payload: dict, runtime_ms: float, failure_reason: str | None = None) -> dict:
    validation = payload.get("validation") or {}
    reconciliation = payload.get("reconciliation") or {}
    arcs = payload.get("arcs") or []
    original_vertices = int(sum(int(arc.get("original_vertex_count", 0)) for arc in arcs))
    final_vertices = int(sum(int(arc.get("vertex_count", 0)) for arc in arcs))
    compression_ratio = float(final_vertices / original_vertices) if original_vertices > 0 else None
    accepted = int(reconciliation.get("accepted_count", 0) or 0)
    candidates = int(reconciliation.get("candidate_count", 0) or 0)
    return {
        "source_file": str(path.as_posix()),
        "split": split,
        "stem": path.stem,
        "format": payload.get("format"),
        "valid_partition": bool(validation.get("is_valid", False)),
        "failure_reason": failure_reason,
        "runtime_ms": float(runtime_ms),
        "face_count": int(validation.get("face_count", len(payload.get("faces", [])))),
        "arc_count": int(validation.get("arc_count", len(arcs))),
        "shared_arc_count": int(validation.get("shared_arc_count", 0)),
        "junction_count": int(validation.get("junction_count", 0)),
        "original_arc_vertex_count": original_vertices,
        "final_arc_vertex_count": final_vertices,
        "arc_vertex_reduction": int(original_vertices - final_vertices),
        "compression_ratio": compression_ratio,
        "union_iou": float(validation.get("union_iou", 0.0)),
        "overlap_area": float(validation.get("overlap_area", 0.0)),
        "missing_adjacency_count": int(len(validation.get("missing_adjacency", []) or [])),
        "extra_adjacency_count": int(len(validation.get("extra_adjacency", []) or [])),
        "candidate_count": candidates,
        "accepted_owner_arc_count": accepted,
        "rejected_owner_arc_count": int(reconciliation.get("rejected_count", 0) or 0),
        "owner_acceptance_rate": float(accepted / candidates) if candidates > 0 else None,
        "label_counts": label_counts(payload),
        "convex_smoke": payload.get("convex_smoke"),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = GlobalApproxConfig(
        face_simplify_tolerance=float(args.face_simplify_tolerance),
        face_area_epsilon=float(args.face_area_epsilon),
        validity_eps=float(args.validity_eps),
    )

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in iter_graph_paths(args.partition_root, args.split):
            if args.max_samples is not None and written >= int(args.max_samples):
                break
            start = time.perf_counter()
            try:
                graph_data = load_json(path)
                payload = build_global_approx_partition_payload(graph_data, config=config, source_tag=str(path.as_posix()))
                if args.convex_smoke:
                    payload["convex_smoke"] = convex_smoke(payload, max_faces=args.convex_smoke_max_faces)
                row = make_row(path, args.split, payload, runtime_ms_since(start))
            except Exception as error:  # noqa: BLE001 - benchmark rows must capture failures.
                row = {
                    "source_file": str(path.as_posix()),
                    "split": args.split,
                    "stem": path.stem,
                    "valid_partition": False,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "runtime_ms": runtime_ms_since(start),
                }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            print(
                f"{path.name}: valid={row.get('valid_partition')} faces={row.get('face_count')} "
                f"arcs={row.get('arc_count')} vertices={row.get('original_arc_vertex_count')}->{row.get('final_arc_vertex_count')} "
                f"accepted={row.get('accepted_owner_arc_count')}/{row.get('candidate_count')} failure={row.get('failure_reason')}"
            )

    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
