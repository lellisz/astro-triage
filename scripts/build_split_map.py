"""Write the versioned HEALPix pixel split map to manifests/split_pixel_map_v1.csv."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from astro_triage.split import build_pixel_map

OUTPUT = REPO_ROOT / "manifests" / "split_pixel_map_v1.csv"


def main() -> int:
    pixel_map = build_pixel_map()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".csv.tmp")
    try:
        pixel_map.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, OUTPUT)
    except OSError as exc:
        print(
            f"failed to write {OUTPUT}: {exc}; check permissions and free disk space",
            file=sys.stderr,
        )
        tmp.unlink(missing_ok=True)
        return 1
    print(f"wrote {len(pixel_map)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
