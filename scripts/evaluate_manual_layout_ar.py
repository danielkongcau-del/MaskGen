from __future__ import annotations

import sys

from evaluate_manual_geometry_ar import main


if "--sequence-kind" not in sys.argv:
    sys.argv.extend(["--sequence-kind", "layout"])


if __name__ == "__main__":
    main()
