from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.dual_graph import load_json
from partition_gen.geometry_approximator import GeometryApproximationConfig, approximate_face_from_partition_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one geometry approximation payload from old base primitives.")
    parser.add_argument("--partition-root", type=Path, default=Path("data/remote_256_partition"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--face-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplify-tolerance", type=float, default=1.5)
    parser.add_argument("--area-epsilon", type=float, default=1e-3)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), indent=2)


def main() -> None:
    args = parse_args()
    graph_path = args.partition_root / args.split / "graphs" / f"{args.stem}.json"
    graph_data = load_json(graph_path)
    face_data = next(face for face in graph_data["faces"] if int(face["id"]) == int(args.face_id))
    payload = approximate_face_from_partition_graph(
        graph_data,
        face_data,
        config=GeometryApproximationConfig(
            simplify_tolerance=float(args.simplify_tolerance),
            area_epsilon=float(args.area_epsilon),
        ),
    )
    payload.update(
        {
            "source_partition_graph": str(graph_path.as_posix()),
            "source_mask": graph_data.get("source_mask"),
            "size": graph_data["size"],
        }
    )
    dump_json(args.output, payload)
    print(
        f"built approx face {args.face_id} from {graph_path.name}: "
        f"base={payload['base_primitive_count']}, approx_vertices={payload['approx_vertex_count']}, "
        f"holes={len(payload['approx_geometry']['holes'])}, iou={payload['approx_iou']:.6f}"
    )


if __name__ == "__main__":
    main()
