from __future__ import annotations

from collections import Counter
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_ar_training import load_checkpoint  # noqa: E402
from partition_gen.manual_geometry_conditioning import iter_jsonl, load_json, _resolve_path  # noqa: E402
from partition_gen.manual_relative_layout_ar import (  # noqa: E402
    RelativeLayoutSafetyConfig,
    attach_relative_layout_frames_to_topology,
    decode_relative_layout_tokens_to_target,
    sample_relative_layout_constrained,
)
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach relative-layout-AR predicted frames to split targets using true local shapes.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--layout-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {})
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


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
    checkpoint, model, _optimizer = load_checkpoint(args.layout_checkpoint, map_location="cpu", load_optimizer=False)
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    tokenizer_config = ParseGraphTokenizerConfig()
    safety_config = _relative_layout_safety_config(args)
    manifest_path = args.split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    manifest_rows = []
    attach_modes: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    attached_total = 0
    missing_total = 0
    valid_layout_count = 0
    hit_eos_count = 0
    for index, row in enumerate(rows):
        topology_path = _resolve_path(row["topology_path"], split_root=args.split_root, manifest_parent=manifest_path.parent)
        topology_target = load_json(topology_path)
        geometry_targets = [
            load_json(_resolve_path(value, split_root=args.split_root, manifest_parent=manifest_path.parent))
            for value in row.get("geometry_paths", []) or []
        ]
        geometry_by_id = {str(target.get("source_node_id")): target for target in geometry_targets}
        sample = sample_relative_layout_constrained(
            model,
            vocab,
            topology_target=topology_target,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            config=tokenizer_config,
            device=device,
            safety_config=safety_config,
        )
        hit_eos_count += int(bool(sample.get("hit_eos", False)))
        layout_valid = False
        try:
            layout_target = decode_relative_layout_tokens_to_target(sample["layout_tokens"], config=tokenizer_config)
            layout_valid = True
            valid_layout_count += 1
        except Exception:
            layout_target = {"nodes": []}
        target, diagnostics = attach_relative_layout_frames_to_topology(
            topology_target,
            layout_target,
            geometry_by_node_id=geometry_by_id,
            safety_config=safety_config,
        )
        target["metadata"].update(
            {
                "sample_index": int(index),
                "layout_checkpoint": str(args.layout_checkpoint.as_posix()),
                "relative_layout_valid": bool(layout_valid),
                "relative_layout_hit_eos": bool(sample.get("hit_eos", False)),
            }
        )
        attach_modes.update(diagnostics.get("attach_modes", {}))
        safety_counts.update((diagnostics.get("relative_layout_safety", {}) or {}).get("counts", {}))
        attached_total += int(diagnostics.get("attached_geometry_count", 0))
        missing_total += int(diagnostics.get("missing_geometry_count", 0))
        output_path = args.output_root / "graphs" / f"sample_{index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append({"sample_index": int(index), "output_path": str(output_path.as_posix())})
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_relative_layout_ar_split_attach_summary_v1",
        "input_count": int(len(rows)),
        "output_count": int(len(manifest_rows)),
        "layout_valid_count": int(valid_layout_count),
        "layout_hit_eos_count": int(hit_eos_count),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "relative_layout_safety": safety_config.to_dict(),
        "relative_layout_safety_counts": dict(safety_counts),
        "split_root": str(args.split_root.as_posix()),
        "layout_checkpoint": str(args.layout_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached relative-layout-ar split samples={summary['output_count']} layout_valid={valid_layout_count} "
        f"attached={attached_total} missing={missing_total} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
