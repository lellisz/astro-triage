"""No per-object data (source_id, ra/dec columns, Gaia-sized ids, pixel5) in published results docs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "source_id": re.compile(r"source_id", re.IGNORECASE),
    "ra/dec column": re.compile(
        r"\b(ra|dec)\b[\"']?\s*[|,=:]|\|\s*(ra|dec)\b", re.IGNORECASE
    ),
    "gaia-sized id": re.compile(r"(?<![\w.])\d{17,19}(?![\w.])"),
    "pixel5": re.compile(r"pixel5|pixel_l5|healpix_l5", re.IGNORECASE),
}


def scoped_files(root: Path) -> list[Path]:
    files = [root / "README.md", root / "docs" / "results" / "MANIFEST.md"]
    results = root / "docs" / "results"
    if results.is_dir():
        files += [p for p in results.rglob("*") if p.is_file()]
    return sorted({p for p in files if p.is_file()})


def scan(root: Path) -> list[str]:
    hits = []
    for path in scoped_files(root):
        for n, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            hits += [
                f"{path.relative_to(root).as_posix()}:{n}: {name}"
                for name, rx in FORBIDDEN.items()
                if rx.search(line)
            ]
    return hits


def test_repo_results_docs_have_no_object_level_data():
    assert scan(ROOT) == []


@pytest.mark.parametrize(
    ("rel", "line", "kind"),
    [
        ("docs/results/a.md", "| 4295806720 | x |", None),
        ("docs/results/a.md", "id 1234567890123456789 star", "gaia-sized id"),
        ("README.md", "17 digits: 12345678901234567", "gaia-sized id"),
        ("docs/results/MANIFEST.md", "| source_id | p |", "source_id"),
        ("docs/results/sub/b.csv", "ra,dec,label", "ra/dec column"),
        ("docs/results/b.md", "| ra | dec |", "ra/dec column"),
        ("docs/results/c.md", "pixel5 = 3", "pixel5"),
        ("docs/results/d.json", '{"ra": 12.3}', "ra/dec column"),
        ("docs/results/d.json", '{"dec": -5.1}', "ra/dec column"),
        ("docs/results/d.json", "{'ra': 12.3}", "ra/dec column"),
        ("docs/results/e.csv", "healpix_l5,label,p_star", "pixel5"),
        ("docs/results/e.md", "| healpix_l5 | label |", "pixel5"),
        ("docs/results/f.json", '{"pixel_l5": 7}', "pixel5"),
        ("docs/results/ok.md", "rate 0.1234567890123456789 ok", None),
    ],
)
def test_scan_flags_object_level_data(tmp_path, rel, line, kind):
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(line + "\n", encoding="utf-8")
    hits = scan(tmp_path)
    if kind is None:
        assert hits == []
    else:
        assert any(kind in h and h.startswith(rel + ":1") for h in hits), hits


def test_scan_clean_tree_is_empty(tmp_path):
    (tmp_path / "README.md").write_text(
        "# title\nF1 0.93 on 18 classes\n", encoding="utf-8"
    )
    assert scan(tmp_path) == []
