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
    ManualLayoutFrameDataset,
    collate_layout_frame_examples,
    evaluate_layout_frame_model,
    evaluate_role_label_frame_baseline,
    load_layout_frame_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a manual layout/frame predictor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--baseline-train-split-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--baseline-max-samples", type=int, default=None)
    parser.add_argument("--baseline-max-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_summary_md(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Layout Frame Evaluation",
        "",
        f"- loss: {payload.get('loss')}",
        f"- origin MAE: {payload.get('origin_mae')}",
        f"- scale MAE: {payload.get('scale_mae')}",
        f"- orientation MAE: {payload.get('orientation_mae')}",
        "",
        "| head | accuracy |",
        "| --- | ---: |",
    ]
    for key, value in (payload.get("head_accuracy", {}) or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "| head | target unique | prediction unique | top prediction bins |", "| --- | ---: | ---: | --- |"])
    for key, hist in (payload.get("head_histograms", {}) or {}).items():
        top_bins = ", ".join(
            f"{row.get('bin')}:{row.get('fraction'):.3f}" for row in (hist.get("prediction_top_bins", []) or [])[:5]
        )
        lines.append(
            f"| {key} | {hist.get('target_unique_count')} | {hist.get('prediction_unique_count')} | {top_bins} |"
        )
    lines.extend(["", "| role | count | origin MAE | scale MAE | orientation MAE |", "| --- | ---: | ---: | ---: | ---: |"])
    for role, metrics in (payload.get("role_metrics", {}) or {}).items():
        lines.append(
            f"| {role} | {metrics.get('count')} | {metrics.get('origin_mae')} | "
            f"{metrics.get('scale_mae')} | {metrics.get('orientation_mae')} |"
        )
    baseline = payload.get("role_label_mean_baseline")
    if baseline:
        lines.extend(
            [
                "",
                "## Role Label Mean Baseline",
                "",
                f"- train examples: {baseline.get('train_example_count')}",
                f"- origin MAE: {baseline.get('origin_mae')}",
                f"- scale MAE: {baseline.get('scale_mae')}",
                f"- orientation MAE: {baseline.get('orientation_mae')}",
                f"- fallback counts: `{json.dumps(baseline.get('fallback_counts', {}), sort_keys=True)}`",
                "",
                "| head | accuracy | prediction unique | top prediction bins |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for key, value in (baseline.get("head_accuracy", {}) or {}).items():
            hist = (baseline.get("head_histograms", {}) or {}).get(key, {})
            top_bins = ", ".join(
                f"{row.get('bin')}:{row.get('fraction'):.3f}" for row in (hist.get("prediction_top_bins", []) or [])[:5]
            )
            lines.append(f"| {key} | {value} | {hist.get('prediction_unique_count')} | {top_bins} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint, model, tokenizer_config = load_layout_frame_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = model.to(device)
    dataset = ManualLayoutFrameDataset(
        args.split_root,
        config=tokenizer_config,
        max_samples=args.max_samples,
        max_examples=args.max_examples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_layout_frame_examples,
        pin_memory=torch.cuda.is_available(),
    )
    payload = evaluate_layout_frame_model(model, loader, device=device, config=tokenizer_config)
    payload.update(
        {
            "format": "maskgen_manual_layout_frame_eval_v1",
            "checkpoint": str(args.checkpoint.as_posix()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split_root": str(args.split_root.as_posix()),
            "example_count": int(len(dataset)),
        }
    )
    if args.baseline_train_split_root is not None:
        baseline_dataset = ManualLayoutFrameDataset(
            args.baseline_train_split_root,
            config=tokenizer_config,
            max_samples=args.baseline_max_samples,
            max_examples=args.baseline_max_examples,
        )
        payload["role_label_mean_baseline"] = evaluate_role_label_frame_baseline(
            baseline_dataset.rows,
            dataset.rows,
            config=tokenizer_config,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if args.summary_md is not None:
        write_summary_md(args.summary_md, payload)
    print(
        f"evaluated layout frames examples={len(dataset)} origin_mae={payload['origin_mae']:.4f} "
        f"scale_mae={payload['scale_mae']:.4f} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
