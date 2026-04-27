from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_topology_sample_validation import (  # noqa: E402
    validate_topology_sample_file,
    write_topology_sample_validation_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sampled manual topology token sequences.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_topology_sample_file(args.samples, top_k=int(args.top_k))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_topology_sample_validation_markdown(payload, args.summary_md)
    print(
        f"validated samples={payload['sample_count']} valid={payload['valid_count']} "
        f"valid_rate={payload['valid_rate']:.4f} "
        f"semantic_valid_rate={payload['semantic_valid_rate']:.4f} "
        f"hit_eos={payload['hit_eos_count']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
