"""Phase 1a: one read-only data root (``ASTRO_DATA_ROOT``) for every dashboard loader.

All fixtures live in ``tmp_path``; nothing is read from or written to the repository's ``data/``.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from test_dashboard_phase4_sky2 import has_hand_written_digits
from test_dashboard_phase7_data import fake_release

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_dl, data_headline, paths

DASH_DIR = Path(D.__file__).resolve().parent
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND
WRITE_METHODS = (
    "write_text",
    "write_bytes",
    "mkdir",
    "unlink",
    "rmdir",
    "rename",
    "replace",
    "touch",
)
WRITE_OS = ("mkdir", "makedirs", "remove", "unlink", "rmdir", "rename", "replace")
# Links count as writes in both directions: a link inside the root, or one that points into it.
LINK_METHODS = ("symlink_to", "hardlink_to")
LINK_OS = ("symlink", "link")
TWO_PATH_METHODS = frozenset({"rename", "replace", *LINK_METHODS})
TWO_PATH_OS = frozenset({"rename", "replace", *LINK_OS})
FRAME_WRITERS = ("to_parquet", "to_csv", "to_json", "to_feather")


# ------------------------------------------------------------------------- resolver
def test_env_var_sets_root(tmp_path: Path) -> None:
    assert paths.data_root({paths.ENV_VAR: str(tmp_path)}) == tmp_path
    assert paths.interim_dir({paths.ENV_VAR: str(tmp_path)}) == tmp_path / "interim"


def test_default_root_is_repo_data() -> None:
    assert paths.data_root({}) == paths.repo_root() / "data"
    assert paths.repo_root() == D.REPO_ROOT


@pytest.mark.parametrize("value", ["", "   ", "~"])
def test_empty_env_var_is_ignored(value: str) -> None:
    assert paths.data_root({paths.ENV_VAR: value}) == paths.repo_root() / "data"


def test_relative_env_var_resolves_against_cwd() -> None:
    assert paths.data_root({paths.ENV_VAR: "rel"}) == Path.cwd() / "rel"


def test_tilde_prefix_expands_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert paths.data_root({paths.ENV_VAR: "~/algo"}) == tmp_path / "algo"
    assert paths.interim_dir({paths.ENV_VAR: "~/algo"}) == tmp_path / "algo" / "interim"


def test_require_interim_raises_without_root(tmp_path: Path) -> None:
    env = {paths.ENV_VAR: str(tmp_path)}
    with pytest.raises(paths.DataRootMissing) as err:
        paths.require_interim(env)
    assert str(tmp_path) not in str(err.value)
    assert not (tmp_path / "interim").exists()
    (tmp_path / "interim").mkdir()
    assert paths.require_interim(env) == tmp_path / "interim"


def test_is_under_root(tmp_path: Path) -> None:
    assert paths.is_under_root(tmp_path / "a" / "b", tmp_path)
    assert not paths.is_under_root(tmp_path / ".." / "x", tmp_path)


def test_interim_constant_comes_from_resolver() -> None:
    assert D.INTERIM_DIR == paths.interim_dir()


# --------------------------------------------------------------------- write guard
class WriteAttempt(PermissionError):
    pass


def _is_write_mode(mode: object) -> bool:
    return isinstance(mode, str) and any(c in mode for c in "wax+")


@contextlib.contextmanager
def write_guard(monkeypatch: pytest.MonkeyPatch, root: Path) -> Iterator[list[str]]:
    """Record and refuse every write-type call whose target is under ``root``."""
    hits: list[str] = []

    def check(kind: str, target: object, bad: bool = True) -> None:
        if not bad or not isinstance(target, (str, bytes, os.PathLike)):
            return
        if paths.is_under_root(Path(os.fsdecode(target)), root):
            hits.append(f"{kind}: {os.fsdecode(target)}")
            raise WriteAttempt(kind)

    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        check("open", file, _is_write_mode(mode))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)

    real_path_open = Path.open

    def path_open(self, mode="r", *a, **k):
        check("Path.open", self, _is_write_mode(mode))
        return real_path_open(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", path_open)

    for name in WRITE_METHODS + LINK_METHODS:
        original = getattr(Path, name)

        def wrapper(self, *a, _orig=original, _name=name, **k):
            check(f"Path.{_name}", self)
            if _name in TWO_PATH_METHODS and a:
                check(f"Path.{_name} target", a[0])
            return _orig(self, *a, **k)

        monkeypatch.setattr(Path, name, wrapper)

    real_os_open = os.open

    def os_open(path, flags, *a, **k):
        check("os.open", path, bool(flags & WRITE_FLAGS))
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", os_open)

    for name in WRITE_OS + LINK_OS:
        original = getattr(os, name)

        def os_wrapper(path, *a, _orig=original, _name=name, **k):
            check(f"os.{_name}", path)
            if _name in TWO_PATH_OS:
                other = a[0] if a else k.get("dst")
                check(f"os.{_name} target", other)
            return _orig(path, *a, **k)

        monkeypatch.setattr(os, name, os_wrapper)

    for name in FRAME_WRITERS:
        original = getattr(pd.DataFrame, name)

        def frame_wrapper(self, *a, _orig=original, _name=name, **k):
            target = a[0] if a else k.get("path_or_buf", k.get("path"))
            check(f"DataFrame.{_name}", target)
            return _orig(self, *a, **k)

        monkeypatch.setattr(pd.DataFrame, name, frame_wrapper)

    yield hits


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """``<root>/interim/rel`` with every artifact the dashboard loaders read."""
    root = tmp_path / "data"
    rel = fake_release(root / "interim" / "rel")
    for sub, name, text in (
        ("eval", "results.json", json.dumps({"meta": {}})),
        ("", "train_report.json", "{}"),
        ("lgbm_seeds", "delta.json", json.dumps({"k": 1, "k_sha256": "x"})),
        ("dl/eval", "results_dl.json", json.dumps({"k": 1})),
        ("dl/eval", "results_dl.md", "## Limitations\n- um\n"),
        ("dl/eval", "risk_coverage_dl.csv", "system,seed,tau,coverage,accuracy\n"),
        ("dl", "val_freeze.json", "{}"),
    ):
        folder = rel / sub
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")
    return root










# ------------------------------------------------------------------ source scans
def _sources() -> list[Path]:
    return sorted(DASH_DIR.glob("*.py"))


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


ANY_RECEIVER = {*WRITE_METHODS, *LINK_METHODS, *FRAME_WRITERS, "rmtree"} - {
    "rename",
    "replace",
}
# Homonyms of str/list/DataFrame methods: flagged only as ``os.<name>``.
OS_ONLY = {*WRITE_OS, *LINK_OS}


def _call_name(call: ast.Call) -> str:
    fn = call.func
    return fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")


def _write_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        fn = call.func
        called = _call_name(call)
        on_os = (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "os"
        )
        if called in ANY_RECEIVER or (called in OS_ONLY and on_os):
            found.append((call.lineno, called))
        if called == "open":
            modes = [a.value for a in call.args[1:2] if isinstance(a, ast.Constant)]
            modes += [
                k.value.value
                for k in call.keywords
                if k.arg == "mode" and isinstance(k.value, ast.Constant)
            ]
            if any(_is_write_mode(m) for m in modes):
                found.append((call.lineno, "open for write"))
    return found


def test_dashboard_sources_have_no_write_calls() -> None:
    bad = [
        f"{path.name}:{line} {what}"
        for path in _sources()
        for line, what in _write_calls(_tree(path))
    ]
    assert not bad, bad


@pytest.mark.parametrize(
    "source",
    [
        "p.write_text('x')",
        "p.mkdir()",
        "os.remove(p)",
        "os.makedirs(p)",
        "os.symlink(a, b)",
        "os.link(a, b)",
        "os.rename(a, b)",
        "os.replace(a, b)",
        "p.symlink_to(t)",
        "p.hardlink_to(t)",
        "df.to_json(p)",
        "df.to_feather(p)",
        "open(p, 'wb')",
    ],
)
def test_write_scan_catches_planted_calls(source: str) -> None:
    assert _write_calls(ast.parse(source))


@pytest.mark.parametrize("source", ["text.replace('a', 'b')", "df.rename(columns={})"])
def test_write_scan_ignores_homonyms(source: str) -> None:
    assert not _write_calls(ast.parse(source))


ROOT_SEGMENTS = {"data", "interim"}


def _split(text: str) -> list[str]:
    return text.replace(chr(92), "/").split("/")


def _data_segment_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Literals that build the data root: path operands, ``os.path.join`` args, f-string parts."""
    hits: list[tuple[int, str]] = []

    def look(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ROOT_SEGMENTS & set(_split(node.value))
        ):
            hits.append((node.lineno, node.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            look(node.left)
            look(node.right)
        elif isinstance(node, ast.Call) and _call_name(node) in {
            "Path",
            "PurePath",
            "joinpath",
            "join",
        }:
            for arg in node.args:
                look(arg)
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                look(part)
    return hits


@pytest.mark.parametrize(
    "source",
    [
        'root = base / "data" / "interim"',
        'Path("data/interim")',
        "Path('data\\\\interim')",
        'base.joinpath("data", "interim")',
        'os.path.join(base, "data", "interim")',
        'f"{base}/data/interim"',
        'base / "data/interim"',
    ],
    ids=[
        "div",
        "literal-slash",
        "literal-backslash",
        "joinpath",
        "os.path.join",
        "fstring",
        "div-literal",
    ],
)
def test_segment_scan_catches_planted_data_paths(source: str) -> None:
    assert _data_segment_hits(ast.parse(source))


@pytest.mark.parametrize(
    "source", ['{"data": 1}', 'f"data {x}"', 'Path("database")', 'base / "metadata"']
)
def test_segment_scan_ignores_unrelated_literals(source: str) -> None:
    assert not _data_segment_hits(ast.parse(source))


def test_only_paths_builds_data_root() -> None:
    bad: list[str] = []
    for path in _sources():
        if path.name == "paths.py":
            continue
        tree = _tree(path)
        bad += [f"{path.name}:{n} {v!r}" for n, v in _data_segment_hits(tree)]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "ASTRO_DATA_ROOT" in node.value
            ):
                bad.append(f"{path.name}:{node.lineno} env var")
    assert not bad, bad


def test_paths_module_has_no_forbidden_imports() -> None:
    imported: set[str] = set()
    for n in ast.walk(_tree(DASH_DIR / "paths.py")):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add((n.module or "").split(".")[0])
    assert imported <= {"__future__", "os", "collections", "pathlib", "typing"}


def test_paths_text_has_no_hand_written_digits() -> None:
    bad = [
        n.value
        for n in ast.walk(_tree(DASH_DIR / "paths.py"))
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


# ------------------------------------------------------------------------- the app


def exercise(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    extra: Callable[[Path], object] | None = None,
) -> list[str]:
    interim = root / "interim"
    with write_guard(monkeypatch, root) as hits:
        rel = D.load_release("rel", base=interim)
        assert D.load_layer2(rel) is None
        assert data_dl.load_results_dl(rel) == {"k": 1}
        assert data_dl.load_risk_coverage_dl(rel) is not None
        assert data_dl.load_limitations(rel) == ["um"]
        assert data_dl.freeze_exists(rel)
        assert data_headline.load_delta(rel) == {"k": 1}
        if extra is not None:
            with contextlib.suppress(WriteAttempt):
                extra(root)
    return hits


def test_no_loader_opens_for_write_under_root(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert exercise(monkeypatch, data_root) == []


@pytest.mark.parametrize(
    "plant",
    [
        lambda root: open(root / "interim" / "x.txt", "w"),  # noqa: SIM115
        lambda root: (root / "interim" / "x.txt").write_text("x"),
        lambda root: (root / "interim" / "d").mkdir(),
        lambda root: os.open(root / "interim" / "y", os.O_WRONLY | os.O_CREAT),
        lambda root: pd.DataFrame({"a": [1]}).to_csv(root / "interim" / "x.csv"),
        lambda root: pd.DataFrame({"a": [1]}).to_csv(path_or_buf=root / "x.csv"),
        lambda root: pd.DataFrame({"a": [1]}).to_json(root / "interim" / "x.json"),
        lambda root: pd.DataFrame({"a": [1]}).to_feather(root / "interim" / "x.f"),
        lambda root: (root / "interim" / "lnk").symlink_to(root.parent),
        lambda root: (root.parent / "lnk").symlink_to(root),
        lambda root: (root.parent / "lnk").symlink_to(root / "interim", True),
        lambda root: (root / "interim" / "lnk").hardlink_to(root.parent / "f"),
        lambda root: os.symlink(root.parent, root / "interim" / "lnk"),
        lambda root: os.symlink(root, root.parent / "lnk"),
        lambda root: os.link(root / "interim" / "rel", root.parent / "lnk"),
        lambda root: os.link(root.parent, root / "interim" / "lnk"),
        lambda root: (root / "interim" / "rel").rename(root.parent / "moved"),
        lambda root: (root.parent / "src").rename(root / "interim" / "moved"),
        lambda root: (root / "interim" / "rel").replace(root.parent / "moved"),
        lambda root: (root.parent / "src").replace(root / "interim" / "moved"),
        lambda root: os.rename(root / "interim" / "rel", root.parent / "moved"),
        lambda root: os.rename(root.parent / "src", root / "interim" / "moved"),
        lambda root: os.replace(root / "interim" / "rel", root.parent / "moved"),
        lambda root: os.replace(root.parent / "src", root / "interim" / "moved"),
        lambda root: os.remove(root / "interim" / "rel"),
        lambda root: os.makedirs(root / "interim" / "d"),
    ],
    ids=[
        "open",
        "write_text",
        "mkdir",
        "os.open",
        "to_csv",
        "to_csv-named",
        "to_json",
        "to_feather",
        "symlink_to-inside",
        "symlink_to-into-root",
        "symlink_to-dir-target",
        "hardlink_to",
        "os.symlink-inside",
        "os.symlink-into-root",
        "os.link-source",
        "os.link-dest",
        "Path.rename-source",
        "Path.rename-dest",
        "Path.replace-source",
        "Path.replace-dest",
        "os.rename-source",
        "os.rename-dest",
        "os.replace-source",
        "os.replace-dest",
        "os.remove",
        "os.makedirs",
    ],
)
def test_planted_write_is_caught(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, plant: Callable[[Path], object]
) -> None:
    hits = exercise(monkeypatch, data_root, extra=plant)
    assert len(hits) == 1
    assert sorted(p.name for p in (data_root / "interim").iterdir()) == ["rel"]
    assert sorted(p.name for p in data_root.parent.iterdir()) == ["data"]


def test_write_outside_root_is_not_flagged(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "elsewhere.txt"
    assert (
        exercise(monkeypatch, data_root, extra=lambda _: other.write_text("ok")) == []
    )
    assert other.read_text() == "ok"
