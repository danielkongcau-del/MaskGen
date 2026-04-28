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
from partition_gen.manual_layout_ar import (  # noqa: E402
    attach_layout_frames_to_topology,
    decode_layout_tokens_to_target,
    sample_layout_constrained,
)
from partition_gen.manual_topology_placeholder_geometry import (  # noqa: E402
    GeometryPlaceholderLibrary,
    decode_topology_tokens_to_target,
    iter_jsonl,
)
from partition_gen.manual_topology_sample_validation import validate_topology_tokens  # noqa: E402
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, load_vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach layout-AR predicted frames and placeholder shapes to topology samples.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--layout-checkpoint", type=Path, required=True)
    parser.add_argument("--shape-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--max-shape-targets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
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


def _resolve_vocab_path(checkpoint: dict) -> Path:
    vocab_path = Path(str(checkpoint["vocab_path"]))
    if vocab_path.exists():
        return vocab_path
    train_config = checkpoint.get("train_config", {})
    candidate = Path(str(train_config.get("train_token_root", ""))) / "vocab.json"
    return candidate if candidate.exists() else vocab_path


def main() -> None:
    args = parse_args()
    checkpoint, model, _optimizer = load_checkpoint(args.layout_checkpoint, map_location="cpu", load_optimizer=False)
    vocab = load_vocabulary(_resolve_vocab_path(checkpoint))
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    tokenizer_config = ParseGraphTokenizerConfig()
    shape_library = GeometryPlaceholderLibrary.from_split_manifest(
        args.shape_split_root,
        seed=int(args.seed),
        max_geometry_targets=args.max_shape_targets,
    )
    sample_rows = list(iter_jsonl(args.samples))
    if args.max_samples is not None:
        sample_rows = sample_rows[: int(args.max_samples)]
    manifest_rows = []
    attach_modes: Counter[str] = Counter()
    skipped_invalid = 0
    valid_layout_count = 0
    attached_total = 0
    missing_total = 0
    for fallback_index, row in enumerate(sample_rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        validation = validate_topology_tokens(tokens)
        if not bool(validation["semantic_valid"]) and not bool(args.include_invalid):
            skipped_invalid += 1
            continue
        topology_target = decode_topology_tokens_to_target(tokens)
        sample_index = int(row.get("sample_index", fallback_index))
        sample = sample_layout_constrained(
            model,
            vocab,
            topology_target=topology_target,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k) if int(args.top_k) > 0 else None,
            config=tokenizer_config,
            device=device,
        )
        layout_valid = False
        try:
            layout_target = decode_layout_tokens_to_target(sample["layout_tokens"], config=tokenizer_config)
            layout_valid = True
            valid_layout_count += 1
        except Exception:
            layout_target = {"nodes": []}
        target, diagnostics = attach_layout_frames_to_topology(
            topology_target,
            layout_target,
            shape_library=shape_library,
        )
        target["metadata"].update(
            {
                "sample_index": int(sample_index),
                "semantic_valid": bool(validation["semantic_valid"]),
                "layout_checkpoint": str(args.layout_checkpoint.as_posix()),
                "layout_valid": bool(layout_valid),
                "layout_hit_eos": bool(sample.get("hit_eos", False)),
            }
        )
        attach_modes.update(diagnostics.get("attach_modes", {}))
        attached_total += int(diagnostics.get("attached_geometry_count", 0))
        missing_total += int(diagnostics.get("missing_geometry_count", 0))
        output_path = args.output_root / "graphs" / f"sample_{sample_index:06d}.json"
        dump_json(output_path, target)
        manifest_rows.append({"sample_index": int(sample_index), "output_path": str(output_path.as_posix())})
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_layout_ar_topology_attach_summary_v1",
        "input_count": int(len(sample_rows)),
        "output_count": int(len(manifest_rows)),
        "skipped_invalid_count": int(skipped_invalid),
        "layout_valid_count": int(valid_layout_count),
        "attached_geometry_count": int(attached_total),
        "missing_geometry_count": int(missing_total),
        "attach_modes": dict(attach_modes),
        "samples": str(args.samples.as_posix()),
        "shape_split_root": str(args.shape_split_root.as_posix()),
        "layout_checkpoint": str(args.layout_checkpoint.as_posix()),
        "output_root": str(args.output_root.as_posix()),
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"attached layout-ar topology samples={summary['output_count']} layout_valid={valid_layout_count} "
        f"attached={attached_total} missing={missing_total} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
