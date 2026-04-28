from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_layout_frame import (  # noqa: E402
    attach_predicted_frames_to_topology_sample_rows,
    load_layout_frame_checkpoint,
)
from partition_gen.manual_topology_placeholder_geometry import GeometryPlaceholderLibrary, iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach predicted layout frames and placeholder local shapes to topology samples.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shape-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shape-targets", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    _checkpoint, model, tokenizer_config = load_layout_frame_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    shape_library = GeometryPlaceholderLibrary.from_split_manifest(
        args.shape_split_root,
        seed=int(args.seed),
        max_geometry_targets=args.max_shape_targets,
    )
    targets = attach_predicted_frames_to_topology_sample_rows(
        rows,
        model=model,
        tokenizer_config=tokenizer_config,
        device=device,
        shape_library=shape_library,
        include_invalid=bool(args.include_invalid),
    )
    manifest_rows = []
    for index, target in enumerate(targets):
        sample_index = int(target.get("metadata", {}).get("sample_index", index))
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append(
            {
                "format": "maskgen_layout_frame_topology_manifest_row_v1",
                "sample_index": int(sample_index),
                "output_path": str(output_path.as_posix()),
                "node_count": int(len(target.get("parse_graph", {}).get("nodes", []) or [])),
                "relation_count": int(len(target.get("parse_graph", {}).get("relations", []) or [])),
                "missing_shape_count": int(target.get("metadata", {}).get("missing_shape_count", 0)),
            }
        )
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_layout_frame_topology_attach_summary_v1",
        "samples": str(args.samples.as_posix()),
        "checkpoint": str(args.checkpoint.as_posix()),
        "shape_split_root": str(args.shape_split_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(rows)),
        "output_count": int(len(targets)),
        "shape_library_size": int(len(shape_library.targets)),
        "missing_shape_count": int(sum(row["missing_shape_count"] for row in manifest_rows)),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached layout frames topology_samples={len(targets)} "
        f"missing_shapes={summary['missing_shape_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
