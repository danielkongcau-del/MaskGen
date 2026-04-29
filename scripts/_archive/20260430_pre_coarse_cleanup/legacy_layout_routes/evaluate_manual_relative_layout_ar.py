from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_manual_geometry_ar import main


if "--sequence-kind" not in sys.argv:
    sys.argv.extend(["--sequence-kind", "relative_layout"])


if __name__ == "__main__":
    main()
