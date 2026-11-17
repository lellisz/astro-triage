"""Builds the self-contained HTML mockups ('Regua do prior') from the release aggregates.

Reads only aggregate artifacts (no per-object data). Each screen is a module of
``scripts/design_mockups/`` exposing ``SCREEN``, ``TITLE``, ``CSS`` and ``render(rel, ledger)``;
register it in ``SCREENS`` to have it built. Writes ``<screen>.html`` and ``number_sources.json``
(displayed number -> artifact file/key) to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from design_mockups import ceu, comparacao, cortes, kit, loaders, metodo, resultado

SCREENS = {m.SCREEN: m for m in (resultado, ceu, comparacao, metodo)}
DEFAULT_OUTDIR = REPO_ROOT / "data" / "interim" / "design"


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build(release: str, outdir: Path) -> dict:
    rel = loaders.load_release_light(release)
    outdir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for name, module in SCREENS.items():
        ledger = kit.Ledger()
        body, footer, facts = module.render(rel, ledger)
        html = kit.page(name, module.TITLE, body, rel.name, footer, module.CSS)
        kit.audit(html)
        _write(outdir / f"{name}.html", html)
        sources[name] = ledger.rows
        summary[name] = {"kb": round(len(html.encode("utf-8")) / 1024, 1), **facts}
    _write(
        outdir / "number_sources.json",
        json.dumps(
            {"release": rel.name, "screens": sources}, ensure_ascii=False, indent=2
        ),
    )
    _write(outdir / "cortes.md", cortes.render(REPO_ROOT, rel.name))
    summary["contrast"] = kit.contrast_table()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="full_v0")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        summary = build(args.release, args.outdir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"build_design_mockups: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
