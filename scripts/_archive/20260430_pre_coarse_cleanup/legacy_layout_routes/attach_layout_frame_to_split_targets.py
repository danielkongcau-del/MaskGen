from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_layout_frame import attach_predicted_frames_to_split_rows, load_layout_frame_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach predicted layout frames to split targets using true local shapes.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
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
    _checkpoint, model, tokenizer_config = load_layout_frame_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    targets = attach_predicted_frames_to_split_rows(
        args.split_root,
        model=model,
        tokenizer_config=tokenizer_config,
        device=device,
        max_samples=args.max_samples,
    )
    manifest_rows = []
    for index, target in enumerate(targets):
        output_path = args.output_root / "graphs" / f"sample_{index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append(
            {
                "format": "maskgen_layout_frame_manifest_row_v1",
                "sample_index": int(index),
                "output_path": str(output_path.as_posix()),
                "node_count": int(len(target.get("parse_graph", {}).get("nodes", []) or [])),
                "relation_count": int(len(target.get("parse_graph", {}).get("relations", []) or [])),
            }
        )
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_layout_frame_attach_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "checkpoint": str(args.checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "output_count": int(len(targets)),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(f"attached layout frames samples={len(targets)} output={args.output_root}")


if __name__ == "__main__":
    main()
