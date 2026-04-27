from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_split_token_audit import (  # noqa: E402
    audit_manual_split_tokens,
    write_manual_split_token_audit_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit tokenized manual topology/geometry split dataset.")
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_manual_split_tokens(args.token_root, top_k=int(args.top_k))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manual_split_token_audit_markdown(audit, args.summary_md)
    print(
        f"audited token_root={args.token_root} "
        f"topology={audit['topology']['sequence_count']} geometry={audit['geometry']['sequence_count']} "
        f"max_single_sequence_tokens={audit['max_single_sequence_tokens']}"
    )


if __name__ == "__main__":
    main()
