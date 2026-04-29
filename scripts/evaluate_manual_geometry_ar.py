from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_coarse_scene_ar import (  # noqa: E402
    CoarseSceneSamplerConfig,
    evaluate_coarse_scene_sample_rows,
    sample_model_coarse_scene_rows,
)
from partition_gen.manual_geometry_conditioned_evaluation import (  # noqa: E402
    sample_model_conditioned_geometry_rows,
    sample_model_oracle_frame_geometry_rows,
)
from partition_gen.manual_geometry_constrained_sampling import GeometryConstrainedSamplerConfig  # noqa: E402
from partition_gen.manual_geometry_evaluation import (  # noqa: E402
    evaluate_geometry_sample_rows,
    sample_model_geometry_rows,
    write_geometry_sample_evaluation_markdown,
    write_geometry_sample_rows,
)
from partition_gen.manual_layout_ar import evaluate_layout_sample_rows, sample_model_conditioned_layout_rows  # noqa: E402
from partition_gen.manual_relative_layout_ar import (  # noqa: E402
    RelativeLayoutSafetyConfig,
    evaluate_relative_layout_sample_rows,
    sample_model_conditioned_relative_layout_rows,
)
from partition_gen.manual_split_token_dataset import ManualSplitTokenSequenceDataset  # noqa: E402
from partition_gen.parse_graph_tokenizer import load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and evaluate a manual geometry AR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--token-root", type=Path, default=None, help="Optional token root used to source forced geometry prefixes.")
    parser.add_argument(
        "--sequence-kind",
        type=str,
        default="auto",
        choices=["auto", "geometry", "conditioned_geometry", "oracle_frame_geometry", "layout", "relative_layout", "coarse_scene"],
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--constrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-role", type=str, default="support_region")
    parser.add_argument("--prefix-label", type=int, default=0)
    parser.add_argument("--prefix-geometry-model", type=str, default="polygon_code")
    parser.add_argument("--max-polygons", type=int, default=8)
    parser.add_argument("--max-points-per-ring", type=int, default=128)
    parser.add_argument("--max-holes-per-polygon", type=int, default=8)
    parser.add_argument("--top-k-invalid", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--safe-relative-layout", action="store_true")
    parser.add_argument("--safe-relative-offset-min", type=float, default=-4.0)
    parser.add_argument("--safe-relative-offset-max", type=float, default=4.0)
    parser.add_argument("--safe-relative-log-scale-min", type=float, default=-3.0)
    parser.add_argument("--safe-relative-log-scale-max", type=float, default=3.0)
    parser.add_argument("--safe-scale-min", type=float, default=1.0)
    parser.add_argument("--safe-scale-max", type=float, default=512.0)
    parser.add_argument("--safe-origin-min", type=float, default=-256.0)
    parser.add_argument("--safe-origin-max", type=float, default=512.0)
    parser.add_argument("--safe-enforce-unit-bbox", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {})
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


def _resolve_token_root(args: argparse.Namespace, checkpoint: dict) -> Path | None:
    if args.token_root is not None:
        return args.token_root
    train_config = checkpoint.get("train_config", {})
    value = train_config.get("val_token_root") or train_config.get("train_token_root")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _relative_layout_safety_config(args: argparse.Namespace) -> RelativeLayoutSafetyConfig:
    return RelativeLayoutSafetyConfig(
        enabled=bool(args.safe_relative_layout),
        relative_offset_min=float(args.safe_relative_offset_min),
        relative_offset_max=float(args.safe_relative_offset_max),
        relative_log_scale_min=float(args.safe_relative_log_scale_min),
        relative_log_scale_max=float(args.safe_relative_log_scale_max),
        scale_min=float(args.safe_scale_min),
        scale_max=float(args.safe_scale_max),
        origin_min=float(args.safe_origin_min),
        origin_max=float(args.safe_origin_max),
        enforce_unit_bbox=bool(args.safe_enforce_unit_bbox),
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    checkpoint, model, _ = load_checkpoint(args.checkpoint, map_location="cpu", load_optimizer=False)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    token_root = _resolve_token_root(args, checkpoint)
    source_rows = None
    sequence_kind = str(args.sequence_kind)
    if sequence_kind == "auto":
        sequence_kind = str(checkpoint.get("train_config", {}).get("sequence_kind", "geometry"))
    if token_root is not None:
        source_rows = ManualSplitTokenSequenceDataset(token_root, sequence_kind=sequence_kind).rows
    constraint_config = None
    if bool(args.constrained):
        constraint_config = GeometryConstrainedSamplerConfig(
            max_polygons=int(args.max_polygons),
            max_points_per_ring=int(args.max_points_per_ring),
            max_holes_per_polygon=int(args.max_holes_per_polygon),
        )
    if sequence_kind == "conditioned_geometry":
        if not bool(args.constrained):
            raise RuntimeError("conditioned_geometry evaluation currently requires constrained sampling")
        if source_rows is None:
            raise RuntimeError("conditioned_geometry evaluation requires --token-root or checkpoint train_config token root")
        rows = sample_model_conditioned_geometry_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            source_rows=source_rows,
            constraint_config=constraint_config,
            progress_every=int(args.progress_every),
            progress_label="conditioned_geometry_eval",
        )
    elif sequence_kind == "oracle_frame_geometry":
        if not bool(args.constrained):
            raise RuntimeError("oracle_frame_geometry evaluation currently requires constrained sampling")
        if source_rows is None:
            raise RuntimeError("oracle_frame_geometry evaluation requires --token-root or checkpoint train_config token root")
        rows = sample_model_oracle_frame_geometry_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            source_rows=source_rows,
            constraint_config=constraint_config,
            progress_every=int(args.progress_every),
            progress_label="oracle_frame_geometry_eval",
        )
    elif sequence_kind == "layout":
        if source_rows is None:
            raise RuntimeError("layout evaluation requires --token-root or checkpoint train_config token root")
        rows = sample_model_conditioned_layout_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            source_rows=source_rows,
            progress_every=int(args.progress_every),
            progress_label="layout_eval",
        )
    elif sequence_kind == "relative_layout":
        if source_rows is None:
            raise RuntimeError("relative_layout evaluation requires --token-root or checkpoint train_config token root")
        safety_config = _relative_layout_safety_config(args)
        rows = sample_model_conditioned_relative_layout_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            source_rows=source_rows,
            safety_config=safety_config,
            progress_every=int(args.progress_every),
            progress_label="relative_layout_eval",
        )
    elif sequence_kind == "coarse_scene":
        rows = sample_model_coarse_scene_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            sampler_config=CoarseSceneSamplerConfig(),
            progress_every=int(args.progress_every),
            progress_label="coarse_scene_eval",
        )
    else:
        rows = sample_model_geometry_rows(
            model,
            vocab,
            num_samples=int(args.num_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            device=device,
            source_rows=source_rows,
            prefix_role=str(args.prefix_role),
            prefix_label=int(args.prefix_label),
            prefix_geometry_model=str(args.prefix_geometry_model),
            constraint_config=constraint_config,
            progress_every=int(args.progress_every),
            progress_label="geometry_eval",
        )
    for row in rows:
        row["checkpoint"] = str(args.checkpoint.as_posix())
        row["sampling_config"] = {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "constrained": bool(args.constrained),
            "constraint_config": None if constraint_config is None else constraint_config.__dict__,
            "token_root": None if token_root is None else str(token_root.as_posix()),
            "relative_layout_safety": _relative_layout_safety_config(args).to_dict()
            if sequence_kind == "relative_layout"
            else None,
        }
    if sequence_kind == "layout":
        summary = evaluate_layout_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    elif sequence_kind == "relative_layout":
        summary = evaluate_relative_layout_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    elif sequence_kind == "coarse_scene":
        summary = evaluate_coarse_scene_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    else:
        summary = evaluate_geometry_sample_rows(rows, top_k_invalid=int(args.top_k_invalid))
    summary.update(
        {
            "checkpoint": str(args.checkpoint.as_posix()),
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "sampling_mode": "constrained" if bool(args.constrained) else "unconstrained",
            "sequence_kind": sequence_kind,
            "sampling_config": rows[0]["sampling_config"] if rows else {},
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if sequence_kind in {"layout", "relative_layout", "coarse_scene"}:
        if sequence_kind == "coarse_scene":
            lines = [
                "# Manual Coarse Scene AR Evaluation",
                "",
                f"- samples: {summary.get('sample_count')}",
                f"- valid: {summary.get('valid_count')} ({summary.get('valid_rate')})",
                f"- semantic valid: {summary.get('semantic_valid_count')} ({summary.get('semantic_valid_rate')})",
                f"- hit_eos: {summary.get('hit_eos_count')}",
                f"- actions: `{json.dumps(summary.get('action_histogram', {}), ensure_ascii=False)}`",
            ]
            args.summary_md.parent.mkdir(parents=True, exist_ok=True)
            args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if args.output_samples is not None:
                write_geometry_sample_rows(args.output_samples, rows)
            print(
                f"evaluated {sequence_kind} samples={summary['sample_count']} valid={summary['valid_count']} "
                f"valid_rate={summary['valid_rate']:.4f} hit_eos={summary['hit_eos_count']} output={args.output_json}"
            )
            return
        lines = [
            "# Manual Relative Layout AR Evaluation" if sequence_kind == "relative_layout" else "# Manual Layout AR Evaluation",
            "",
            f"- samples: {summary.get('sample_count')}",
            f"- valid: {summary.get('valid_count')} ({summary.get('valid_rate')})",
            f"- hit_eos: {summary.get('hit_eos_count')}",
            f"- origin MAE: {summary.get('origin_mae')}",
            f"- scale MAE: {summary.get('scale_mae')}",
            f"- orientation MAE: {summary.get('orientation_mae')}",
        ]
        if sequence_kind == "relative_layout":
            lines.extend(["", f"- anchors: `{json.dumps(summary.get('anchor_mode_histogram', {}), ensure_ascii=False)}`"])
            diagnostics = summary.get("numeric_diagnostics", {}) or {}
            scale_stats = diagnostics.get("scale_stats", {}) or {}
            lines.extend(
                [
                    f"- origin outside ratio: {diagnostics.get('origin_outside_ratio')}",
                    f"- unit bbox visible ratio: {diagnostics.get('unit_bbox_visible_ratio')}",
                    f"- scale p50/p95/max: {scale_stats.get('p50')} / {scale_stats.get('p95')} / {scale_stats.get('max')}",
                ]
            )
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        write_geometry_sample_evaluation_markdown(summary, args.summary_md)
    if args.output_samples is not None:
        write_geometry_sample_rows(args.output_samples, rows)
    print(
        f"evaluated {sequence_kind} samples={summary['sample_count']} valid={summary['valid_count']} "
        f"valid_rate={summary['valid_rate']:.4f} hit_eos={summary['hit_eos_count']} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
