from __future__ import annotations

import sys

from train_manual_geometry_ar import main


if "--sequence-kind" not in sys.argv:
    sys.argv.extend(["--sequence-kind", "layout"])
if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", "outputs/manual_layout_ar"])


if __name__ == "__main__":
    main()
