"""CLI: ``python -m astro_triage.web_export.figures --release <name> --data <dir> [--out <dir>]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from astro_triage.web_export.figures import FigureError, write_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m astro_triage.web_export.figures",
        description="Desenha as figuras SVG do site a partir dos JSON exportados.",
    )
    parser.add_argument("--release", required=True, help="nome da release")
    parser.add_argument("--data", required=True, type=Path, help="pasta com <release>/*.json")
    parser.add_argument("--out", type=Path, help="pasta de saída (padrão: a mesma de --data)")
    args = parser.parse_args(argv)
    try:
        written = write_figures(args.data, args.out or args.data, args.release)
    except FigureError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"erro: não foi possível gravar ({type(exc).__name__}: {exc.filename})", file=sys.stderr)
        return 1
    for path in written:
        print(f"{path} {path.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
