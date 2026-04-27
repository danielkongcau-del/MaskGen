from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partition_gen.manual_parse_graph_target_audit import iter_manual_parse_graph_target_paths  # noqa: E402
from partition_gen.manual_parse_graph_visualization import (  # noqa: E402
    load_json,
    render_manual_parse_graph_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated full manual parse-graph targets to PNG.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target", type=Path, help="One parse_graph JSON file.")
    source.add_argument("--target-root", type=Path, help="A graph directory or output root with manifest.jsonl.")
    parser.add_argument("--output-png", type=Path, default=None, help="Output PNG for --target mode.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory for --target-root mode.")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--annotate", action="store_true")
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.target is not None:
        if args.output_png is None:
            raise SystemExit("--output-png is required with --target")
        diagnostics = render_manual_parse_graph_target(
            load_json(args.target),
            args.output_png,
            dpi=int(args.dpi),
            alpha=float(args.alpha),
            annotate=bool(args.annotate),
        )
        print(
            f"rendered target={args.target} "
            f"nodes={diagnostics['rendered_node_count']} "
            f"polygons={diagnostics['rendered_polygon_count']} "
            f"output={args.output_png}"
        )
        return

    if args.output_root is None:
        raise SystemExit("--output-root is required with --target-root")
    paths = iter_manual_parse_graph_target_paths(args.target_root)
    if args.max_targets is not None:
        paths = paths[: int(args.max_targets)]
    rows = []
    for path in paths:
        output_png = args.output_root / f"{path.stem}.png"
        diagnostics = render_manual_parse_graph_target(
            load_json(path),
            output_png,
            dpi=int(args.dpi),
            alpha=float(args.alpha),
            annotate=bool(args.annotate),
        )
        diagnostics["source"] = str(path.as_posix())
        rows.append(diagnostics)
    summary = {
        "format": "maskgen_manual_parse_graph_visualization_summary_v1",
        "target_root": str(args.target_root.as_posix()),
        "output_root": str(args.output_root.as_posix()),
        "rendered_count": int(len(rows)),
        "rendered_node_count": int(sum(int(row["rendered_node_count"]) for row in rows)),
        "rendered_polygon_count": int(sum(int(row["rendered_polygon_count"]) for row in rows)),
        "rows": rows,
    }
    dump_json(args.output_root / "summary.json", summary)
    print(
        f"rendered targets={summary['rendered_count']} "
        f"nodes={summary['rendered_node_count']} "
        f"polygons={summary['rendered_polygon_count']} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
