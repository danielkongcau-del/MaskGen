from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_layout_residual import (  # noqa: E402
    attach_retrieved_residual_layout_to_split_targets,
    build_residual_library_from_split,
    load_layout_residual_checkpoint,
    summarize_retrieved_residual_layout_targets,
)
from partition_gen.manual_layout_retrieval import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach residual-corrected retrieved layouts to split targets using true local shapes."
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--library-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--exclude-same-stem", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint, model, tokenizer_config = load_layout_residual_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    library_entries, library_summary, fallback_frames = build_residual_library_from_split(
        args.library_split_root,
        max_library_samples=args.max_library_samples,
    )
    targets = attach_retrieved_residual_layout_to_split_targets(
        args.split_root,
        library_entries=library_entries,
        fallback_frames=fallback_frames,
        model=model,
        tokenizer_config=tokenizer_config,
        device=device,
        max_samples=args.max_samples,
        exclude_same_stem=bool(args.exclude_same_stem),
    )
    manifest_rows: list[dict] = []
    for index, target in enumerate(targets):
        output_path = args.output_root / "graphs" / f"sample_{index:06d}.json"
        dump_json(output_path, target)
        metadata = target.get("metadata", {}) or {}
        manifest_rows.append(
            {
                "sample_index": int(index),
                "output_path": str(output_path.as_posix()),
                "retrieval_score": metadata.get("retrieval_score"),
                "attached_geometry_count": int(metadata.get("attached_geometry_count", 0)),
                "missing_geometry_count": int(metadata.get("missing_geometry_count", 0)),
                "attach_modes": metadata.get("attach_modes", {}),
            }
        )
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_retrieved_residual_layout_split_attach_summary_v1",
        "split_root": str(args.split_root.as_posix()),
        "library_split_root": str(args.library_split_root.as_posix()),
        "checkpoint": str(args.checkpoint.as_posix()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output_root": str(args.output_root.as_posix()),
        "input_count": int(len(targets)),
        "output_count": int(len(manifest_rows)),
        "max_library_samples": args.max_library_samples,
        "exclude_same_stem": bool(args.exclude_same_stem),
        "library_summary": library_summary,
        **summarize_retrieved_residual_layout_targets(targets),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached retrieved-residual-layout split samples={summary['output_count']} "
        f"attached={summary['attached_geometry_count']} missing={summary['missing_geometry_count']} "
        f"library={library_summary['entry_count']} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
