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
    ManualLayoutResidualDataset,
    build_residual_library_from_split,
    collate_layout_residual_examples,
    evaluate_layout_residual_regressor,
    load_layout_residual_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieved-layout residual frame predictor.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--library-split-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-library-samples", type=int, default=None)
    parser.add_argument("--exclude-same-stem", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_summary_md(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Layout Residual Evaluation",
        "",
        f"- examples: {payload.get('example_count')}",
        f"- loss: {payload.get('loss')}",
        f"- retrieval baseline origin MAE: {payload.get('baseline_origin_mae')}",
        f"- residual-corrected origin MAE: {payload.get('residual_origin_mae')}",
        f"- origin MAE delta: {payload.get('origin_mae_delta')}",
        f"- origin MAE improvement fraction: {payload.get('origin_mae_improvement_fraction')}",
        f"- scale out of range before clamp: {payload.get('scale_out_of_range_count')}",
        f"- tokenizer-scale out of range before clamp: {payload.get('scale_above_tokenizer_max_count')}",
        f"- geometry-aware scale clamps: {payload.get('geometry_scale_clamped_count')}",
        f"- bbox huge after clamp: {payload.get('bbox_huge_count')}",
        f"- raw bbox huge before clamp: {payload.get('raw_bbox_huge_count')}",
        f"- retrieval score stats: `{json.dumps(payload.get('retrieval_score_stats', {}), sort_keys=True)}`",
        f"- mapping modes: `{json.dumps(payload.get('mapping_mode_histogram', {}), sort_keys=True)}`",
        "",
        "| metric | retrieval baseline | residual corrected |",
        "| --- | ---: | ---: |",
        f"| origin_mae | {payload.get('baseline_origin_mae')} | {payload.get('residual_origin_mae')} |",
        f"| scale_mae | {payload.get('baseline_scale_mae')} | {payload.get('residual_scale_mae')} |",
        f"| orientation_mae | {payload.get('baseline_orientation_mae')} | {payload.get('residual_orientation_mae')} |",
        "",
        "| diagnostic | value |",
        "| --- | ---: |",
        f"| scale_below_min_count | {payload.get('scale_below_min_count')} |",
        f"| scale_above_max_count | {payload.get('scale_above_max_count')} |",
        f"| scale_above_tokenizer_max_count | {payload.get('scale_above_tokenizer_max_count')} |",
        f"| geometry_scale_clamped_count | {payload.get('geometry_scale_clamped_count')} |",
        f"| scale_clamped_count | {payload.get('scale_clamped_count')} |",
        f"| raw_bbox_huge_count | {payload.get('raw_bbox_huge_count')} |",
        f"| bbox_huge_count | {payload.get('bbox_huge_count')} |",
        "",
        "| role | count | baseline origin MAE | residual origin MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role, metrics in (payload.get("role_metrics", {}) or {}).items():
        lines.append(
            f"| {role} | {metrics.get('count')} | "
            f"{metrics.get('baseline_origin_mae')} | {metrics.get('residual_origin_mae')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    dataset = ManualLayoutResidualDataset(
        args.split_root,
        library_entries=library_entries,
        fallback_frames=fallback_frames,
        config=tokenizer_config,
        max_samples=args.max_samples,
        max_examples=args.max_examples,
        exclude_same_stem=bool(args.exclude_same_stem),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_layout_residual_examples,
        pin_memory=torch.cuda.is_available(),
    )
    payload = evaluate_layout_residual_regressor(model, loader, device=device, config=tokenizer_config)
    payload.update(
        {
            "format": "maskgen_manual_layout_residual_eval_v1",
            "checkpoint": str(args.checkpoint.as_posix()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split_root": str(args.split_root.as_posix()),
            "library_split_root": str(args.library_split_root.as_posix()),
            "library_summary": library_summary,
            "exclude_same_stem": bool(args.exclude_same_stem),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"evaluated layout residual examples={len(dataset)} "
        f"baseline_origin_mae={payload['baseline_origin_mae']:.4f} "
        f"residual_origin_mae={payload['residual_origin_mae']:.4f} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
