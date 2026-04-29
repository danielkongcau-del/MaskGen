from __future__ import annotations

import sys

from train_manual_geometry_ar import main


if __name__ == "__main__":
    if "--sequence-kind" not in sys.argv:
        sys.argv.extend(["--sequence-kind", "coarse_scene"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "outputs/manual_coarse_scene_ar"])
    main()
