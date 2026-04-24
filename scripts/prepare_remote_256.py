import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 256x256 version of the remote dataset for mask generation experiments."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/remote"),
        help="Input dataset root. Expected structure: split/{images,masks}/*.png",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/remote_256"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target square size for output images and masks.",
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Also resize and export RGB images alongside masks.",
    )
    return parser.parse_args()


def collect_split_paths(input_root):
    splits = {}
    for split_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        images = {p.name: p for p in sorted(image_dir.glob("*.png"))}
        masks = {p.name: p for p in sorted(mask_dir.glob("*.png"))}
        common = sorted(set(images) & set(masks))
        splits[split_dir.name] = {
            "images": [images[name] for name in common],
            "masks": [masks[name] for name in common],
        }
    return splits


def infer_mask_values(mask_paths):
    values = set()
    for mask_path in tqdm(mask_paths, desc="Scanning mask values", unit="mask"):
        arr = np.array(Image.open(mask_path), dtype=np.uint8)
        values.update(np.unique(arr).tolist())
    return sorted(values)


def build_lookup(values):
    if len(values) > 256:
        raise ValueError(f"Too many mask values ({len(values)}) for uint8 lookup.")
    lut = np.full(256, 255, dtype=np.uint8)
    for idx, value in enumerate(values):
        lut[value] = idx
    value_to_id = {int(value): int(idx) for idx, value in enumerate(values)}
    id_to_value = {int(idx): int(value) for idx, value in enumerate(values)}
    return lut, value_to_id, id_to_value


def downsample_mask_id(mask_id, target_size, num_classes):
    height, width = mask_id.shape
    if height != width:
        raise ValueError(f"Mask must be square, got {mask_id.shape}.")
    if height % target_size != 0 or width % target_size != 0:
        raise ValueError(f"Mask size {mask_id.shape} is not divisible by target size {target_size}.")

    factor = height // target_size
    blocks = mask_id.reshape(target_size, factor, target_size, factor).transpose(0, 2, 1, 3)
    flat_blocks = blocks.reshape(target_size, target_size, factor * factor)

    counts = np.stack(
        [(flat_blocks == cls_id).sum(axis=-1) for cls_id in range(num_classes)],
        axis=-1,
    )
    out_id = counts.argmax(axis=-1).astype(np.uint8)

    max_counts = counts.max(axis=-1, keepdims=True)
    ties = counts == max_counts
    tie_positions = np.argwhere(ties.sum(axis=-1) > 1)

    if tie_positions.size > 0:
        center_start = max(0, factor // 2 - 1)
        center_end = min(factor, center_start + 2)
        center_block = blocks[:, :, center_start:center_end, center_start:center_end].reshape(
            target_size, target_size, -1
        )
        center_counts = np.stack(
            [(center_block == cls_id).sum(axis=-1) for cls_id in range(num_classes)],
            axis=-1,
        )
        for y, x in tie_positions:
            tied_ids = np.flatnonzero(ties[y, x])
            best_center = tied_ids[np.argmax(center_counts[y, x, tied_ids])]
            out_id[y, x] = np.uint8(best_center)

    return out_id


def save_image_resize(src_path, dst_path, target_size):
    img = Image.open(src_path).convert("RGB")
    img = img.resize((target_size, target_size), resample=RESAMPLE_LANCZOS)
    img.save(dst_path)


def save_mask(mask_id, palette_values, dst_id_path, dst_palette_path):
    mask_palette = palette_values[mask_id]
    Image.fromarray(mask_id, mode="L").save(dst_id_path)
    Image.fromarray(mask_palette, mode="L").save(dst_palette_path)


def counter_to_dict(counter):
    return {str(int(key)): int(value) for key, value in sorted(counter.items())}


def main():
    args = parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    splits = collect_split_paths(args.input_root)
    if not splits:
        raise RuntimeError(f"No valid split directories found under {args.input_root}")

    all_mask_paths = [path for split in splits.values() for path in split["masks"]]
    mask_values = infer_mask_values(all_mask_paths)
    lut, value_to_id, id_to_value = build_lookup(mask_values)
    palette_values = np.array([id_to_value[idx] for idx in range(len(mask_values))], dtype=np.uint8)
    num_classes = len(mask_values)

    meta_dir = args.output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "target_size": args.target_size,
        "mask_values": [int(v) for v in mask_values],
        "value_to_id": {str(k): int(v) for k, v in value_to_id.items()},
        "id_to_value": {str(k): int(v) for k, v in id_to_value.items()},
        "num_classes": num_classes,
        "with_images": bool(args.with_images),
        "splits": {},
    }

    for split_name, split_data in splits.items():
        split_out = args.output_root / split_name
        mask_id_out = split_out / "masks_id"
        mask_palette_out = split_out / "masks"
        image_out = split_out / "images"

        mask_id_out.mkdir(parents=True, exist_ok=True)
        mask_palette_out.mkdir(parents=True, exist_ok=True)
        if args.with_images:
            image_out.mkdir(parents=True, exist_ok=True)

        original_counts = Counter()
        downsampled_counts = Counter()
        unique_hist = Counter()

        paired_paths = list(zip(split_data["images"], split_data["masks"]))
        progress = tqdm(paired_paths, desc=f"Processing {split_name}", unit="pair")
        for image_path, mask_path in progress:
            mask_arr = np.array(Image.open(mask_path), dtype=np.uint8)
            original_counts.update(mask_arr.reshape(-1).tolist())

            unknown_values = np.unique(mask_arr[lut[mask_arr] == 255]).tolist()
            if unknown_values:
                raise ValueError(f"Found unmapped mask values in {mask_path}: {unknown_values}")

            mask_id = lut[mask_arr]
            mask_id_256 = downsample_mask_id(mask_id, args.target_size, num_classes)
            downsampled_counts.update(mask_id_256.reshape(-1).tolist())
            unique_hist.update([int(len(np.unique(mask_id_256)))])

            save_mask(
                mask_id_256,
                palette_values,
                mask_id_out / mask_path.name,
                mask_palette_out / mask_path.name,
            )

            if args.with_images:
                save_image_resize(image_path, image_out / image_path.name, args.target_size)

        summary["splits"][split_name] = {
            "num_samples": len(paired_paths),
            "original_pixel_counts": counter_to_dict(original_counts),
            "downsampled_id_pixel_counts": counter_to_dict(downsampled_counts),
            "downsampled_unique_class_histogram": counter_to_dict(unique_hist),
        }

    with open(meta_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
