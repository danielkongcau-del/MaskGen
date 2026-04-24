from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    build_bridged_convex_partition_from_geometry_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visualizations for convex splitter benchmark failures.")
    parser.add_argument("--benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/convex_splitter_failures"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/remote_256"))
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cgal", "fallback_hm", "fallback_cdt_greedy"])
    parser.add_argument("--cgal-cli", type=str, default=None)
    parser.add_argument("--max-bridge-sets", type=int, default=256)
    parser.add_argument("--vertex-round-digits", type=int, default=8)
    parser.add_argument("--area-eps", type=float, default=1e-8)
    parser.add_argument("--validity-eps", type=float, default=1e-7)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--iou-threshold", type=float, default=0.999999)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_failure(row: dict, *, iou_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("failure_reason"):
        reasons.append("failure_reason")
    if row.get("validation_iou") is None or float(row.get("validation_iou") or 0.0) < iou_threshold:
        reasons.append("low_iou")
    baseline = row.get("baseline_piece_count")
    bridged = row.get("bridged_piece_count")
    if baseline is not None and bridged is not None and int(bridged) >= int(baseline):
        reasons.append("no_piece_reduction")
    if str(row.get("backend") or "").startswith("fallback"):
        reasons.append("fallback_backend")
    if int(row.get("rejected_bridge_set_count") or 0) > 0:
        reasons.append("rejected_bridge_sets")
    return bool(reasons), reasons


def safe_name(row: dict, reasons: list[str]) -> str:
    split = row.get("split") or "split"
    stem = row.get("stem") or Path(str(row.get("source_file") or "sample")).stem
    face_id = row.get("face_id", "face")
    scale = str(row.get("cut_slit_scale", "scale")).replace(".", "p").replace("-", "m")
    reason = "_".join(reasons[:3])
    return f"{split}_{stem}_face{face_id}_scale{scale}_{reason}"


def build_partition(row: dict, args: argparse.Namespace, output_path: Path) -> dict:
    approx_path = Path(row["source_file"])
    geometry_payload = load_json(approx_path)
    payload = build_bridged_convex_partition_from_geometry_payload(
        geometry_payload,
        config=BridgedPartitionConfig(
            max_bridge_sets=args.max_bridge_sets,
            vertex_round_digits=args.vertex_round_digits,
            area_eps=args.area_eps,
            validity_eps=args.validity_eps,
            backend=args.backend,
            cgal_cli=args.cgal_cli,
            cut_slit_scale=float(row.get("cut_slit_scale") or 1e-6),
        ),
        source_tag=str(approx_path.as_posix()),
    )
    dump_json(output_path, payload)
    return payload


def visualize(partition_path: Path, row: dict, image_path: Path, args: argparse.Namespace) -> bool:
    stem = row.get("stem")
    split = row.get("split")
    if not stem or not split:
        return False
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "visualize_bridged_convex_partition.py"),
        "--partition-json",
        str(partition_path),
        "--mask-root",
        str(args.mask_root),
        "--split",
        str(split),
        "--stem",
        str(stem),
        "--output",
        str(image_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    partitions_dir = args.output_dir / "partitions"
    images_dir = args.output_dir / "images"
    manifest_path = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in iter_rows(args.benchmark_jsonl):
            matched, reasons = is_failure(row, iou_threshold=args.iou_threshold)
            if not matched:
                continue
            if exported >= args.limit:
                break
            name = safe_name(row, reasons)
            partition_path = partitions_dir / f"{name}.json"
            image_path = images_dir / f"{name}.png"
            record = dict(row)
            record["failure_export_reasons"] = reasons
            record["export_partition_json"] = str(partition_path.as_posix())
            record["export_image"] = str(image_path.as_posix())
            try:
                build_partition(row, args, partition_path)
                record["visualized"] = visualize(partition_path, row, image_path, args)
            except Exception as error:  # noqa: BLE001 - export should continue after individual failures.
                record["visualized"] = False
                record["export_failure_reason"] = str(error)
            manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported += 1
            print(f"exported {name}: reasons={','.join(reasons)} visualized={record.get('visualized')}")

    print(f"wrote {exported} failure records to {manifest_path}")


if __name__ == "__main__":
    main()
