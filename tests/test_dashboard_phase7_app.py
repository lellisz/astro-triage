"""Phase 7 app pieces: navigation, cache stamp, notes, errors, theme, constants (synthetic)."""

from __future__ import annotations

import ast
import inspect
import os
import re
from pathlib import Path

import pandas as pd

from astro_triage.dashboard import constants, theme
from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import page_errors as PE
from astro_triage.dashboard import sidebar_info as SI

APP = Path(D.__file__).resolve().parent / "app.py"
M = ("lgbm_full", "lgbm_noind")


# --------------------------------------------------------------------------- (a) app.py
def test_app_uses_top_navigation_and_bounded_release_cache() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "tabs" not in attrs
    nav = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "navigation"
    ]
    assert len(nav) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in nav[0].keywords} == {
        "position": "top"
    }
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_release"
    )
    (dec,) = fn.decorator_list
    assert ast.unparse(dec.func) == "st.cache_resource"
    assert {k.arg: ast.literal_eval(k.value) for k in dec.keywords}["max_entries"] == 1


# ----------------------------------------------------------------------- (d) cache
def test_stamp_changes_with_each_artifact(tmp_path: Path) -> None:
    rel = D.Release("r", tmp_path, {"meta": {}}, {}, pd.DataFrame(), pd.DataFrame(), 0)
    assert DC.stamp(rel) == 0
    t = 1_700_000_000
    for i, name in enumerate(DC._ARTIFACTS):
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}", encoding="utf-8")
        os.utime(f, (t, t))
    base = DC.stamp(rel)
    assert base == t * 10**9
    for i, name in enumerate(DC._ARTIFACTS):
        later = t + 100 * (i + 1)
        os.utime(tmp_path / name, (later, later))
        assert DC.stamp(rel) == later * 10**9
        os.utime(tmp_path / name, (t, t))
    assert DC.stamp(rel) == base


def test_cached_wrappers_keep_the_signature_of_data() -> None:
    public = [
        n
        for n, f in vars(DC).items()
        if inspect.isfunction(f) and not n.startswith("_") and n != "stamp"
    ]
    pairs = [n for n in public if hasattr(D, n)]
    assert {"headline", "load_layer2", "error_rate_table", "galaxy_scatter"} <= set(
        pairs
    )
    for n in (
        pairs
    ):  # the wrapper may drop trailing params of data.* (load_layer2: no ``base``)
        a, b = inspect.signature(getattr(DC, n)), inspect.signature(getattr(D, n))
        mine = [(p.name, p.kind, p.default) for p in a.parameters.values()]
        theirs = [(p.name, p.kind, p.default) for p in b.parameters.values()]
        assert mine == theirs[: len(mine)], n
        assert n == "load_layer2" or mine == theirs, n


# ------------------------------------------------------------ (e) notes and errors
def sidebar_rel(n_meta: object, unscored: int, header: str = "") -> D.Release:
    res = {"meta": {"header": header, **({} if n_meta is None else {"n_test": n_meta})}}
    return D.Release(
        "r", Path("."), res, {}, pd.DataFrame(), pd.DataFrame({"a": range(3)}), unscored
    )


def test_sidebar_notes_unscored_and_n_test_divergence() -> None:
    assert SI.sidebar_notes(sidebar_rel(3, 0)) == ["3 objetos de teste"]
    assert SI.sidebar_notes(sidebar_rel(None, 0)) == ["3 objetos de teste"]
    notes = SI.sidebar_notes(sidebar_rel(1234, 2, "PROVISIONAL: x"))
    assert notes[0] == "3 objetos de teste"
    assert notes[1] == "2 objetos sem escore descartados"
    assert notes[2] == "A avaliação registra 1.234 objetos de teste; esta tela usa 3"
    assert len(notes) == 3


def test_page_errors_hide_paths_and_tracebacks() -> None:
    win = r"C:\Users\user\secret\results.json"
    assert (
        PE.describe_error(FileNotFoundError(2, "No such file", win))
        == "faltou o artefato results.json"
    )
    assert PE.describe_error(KeyError("meta")) == "faltou o campo 'meta' em um artefato"
    msg = PE.page_error(ValueError(f"cannot read {win} and /home/u/data/x.parquet"))
    for leak in ("Users", "secret", "home", "Traceback", "C:"):
        assert leak not in msg
    assert msg.startswith(
        "Não foi possível montar esta página: ValueError: cannot read"
    )
    long = PE.describe_error(RuntimeError("x" * 500))
    assert len(long) <= len("RuntimeError: ") + 120
    assert PE.describe_error(RuntimeError()) == "RuntimeError"
    try:
        raise OSError(f"boom {win}")
    except OSError as exc:
        assert "Traceback" not in PE.page_error(exc) and "File " not in PE.page_error(
            exc
        )


# ----------------------------------------------------------------------- (f) theme
def test_theme_width_and_button_target() -> None:
    narrow, wide = theme.css(wide=False), theme.css(wide=True)
    assert (
        f"max-width: {theme.READING_WIDTH_PX}px" in narrow
        and theme.READING_WIDTH_PX == 800
    )
    assert f"max-width: {theme.WIDE_WIDTH_PX}px" in wide
    assert str(theme.WIDE_WIDTH_PX) not in narrow and "800px" not in wide
    for css in (narrow, wide):
        rule = re.search(r'\[data-testid="stButtonGroup"\] button\s*\{([^}]*)\}', css)
        assert rule and re.search(r"min-height:\s*44px", rule.group(1))
    assert theme.MIN_TARGET_PX == 44


# ---------------------------------------------------------------- (h) constants
def test_constants() -> None:
    assert constants.ADR_NUMBER == 6 and constants.NO_BAR == {"displayModeBar": False}
