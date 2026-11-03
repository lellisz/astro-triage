"""Pilot report (ADR 0005 17.7): numbers from a pre-filled cache and ledger, zero network.

Synthetic release in ``tmp_path`` (real pixel map, fake cache, fake ledger); the opener-free
module is proven by forbidding every socket path and by a clean-interpreter import check."""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_layer2_eval import MODEL, Rel, build_release

from astro_triage.evaluation import layer2_pilot as P
from astro_triage.evaluation.layer2_report import FORBIDDEN_WORDS
from astro_triage.llm.cache import ResultCache
from astro_triage.llm.prompt import PROMPT_VERSION
from astro_triage.llm.routing import (
    DEFAULT_SEED,
    MANIFEST_NAME,
    model_slug,
    request_for,
    select_routed,
)

REPO = Path(__file__).resolve().parents[1]
F_VAL = 0.5
SAMPLE = 100
N_PENDING = 3
LATENCY = 1000  # ms


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", boom)


class World:
    def __init__(self, base: Path) -> None:
        self.r: Rel = build_release(base)
        man = json.loads(self.r.manifest.read_text("utf-8"))
        man["val_routed_fraction"] = F_VAL
        self.r.manifest.write_text(json.dumps(man), encoding="utf-8")
        self.cache_dir = base / "cache"
        self.rows = select_routed(
            self.r.rel, "val", self.r.manifest, SAMPLE, DEFAULT_SEED
        )
        pos = {int(s): i for i, s in enumerate(self.r.ids)}
        self.y = [str(self.r.y[pos[int(s)]]) for s in self.rows["source_id"]]
        self.cache_path = self.cache_dir / model_slug(MODEL) / f"{PROMPT_VERSION}.jsonl"
        self.fill()
        self.write_ledger()

    def fill(self) -> None:
        """90 right OK, 4 wrong OK, 3 UNKNOWN, 3 without any record (pending)."""
        wrong = {"STAR": "QSO", "QSO": "GALAXY", "GALAXY": "STAR"}
        with ResultCache(self.cache_path) as cache:
            for i, (_, row) in enumerate(self.rows.iterrows()):
                if i >= SAMPLE - N_PENDING:
                    continue
                status, label = "OK", self.y[i]
                if 90 <= i < 94:
                    label = wrong[self.y[i]]
                if i >= 94:
                    status, label = "UNKNOWN", "UNKNOWN"
                cache.put(
                    request_for(row, MODEL, "blind", False)[1],
                    {
                        "source_id": int(row["source_id"]),
                        "status": status,
                        "label_l2": label,
                        "confidence_raw": 80,
                        "model_reported": MODEL,
                        "prompt_tokens": 100,
                        "completion_tokens": 7,
                        "latency_ms": LATENCY,
                        "attempts": 1,
                        "finish_reason": "stop",
                    },
                )

    def write_ledger(self) -> None:
        def line(**kw: Any) -> str:
            base = {"model": MODEL, "arm": "blind", "attempt": 1, "http_status": 200}
            return json.dumps({**base, **kw})

        lines = [line(enable_thinking=False) for _ in range(97)]
        lines += [line(enable_thinking=False, http_status=429)]
        lines += [line(enable_thinking=False, attempt=2)]
        lines += [
            line(enable_thinking=True)
        ] * 5  # other reasoning setting: not counted
        lines += [line(), line()]  # written before enable_thinking existed
        lines += [line(model="other/model", enable_thinking=False)]
        (self.r.rel / "layer2" / "calls_20261027T000000Z.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def report(self, rpm_cap: float = 30.0, **kw: Any) -> dict[str, Any]:
        return P.pilot_report(
            self.r.rel,
            [MODEL],
            ["blind"],
            [],
            SAMPLE,
            self.r.rel,
            rpm_cap,
            cache_dir=self.cache_dir,
            n_boot=20,
            **kw,
        )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return World(tmp_path_factory.mktemp("pilot"))


def test_wilson_matches_a_hand_computed_case() -> None:
    lo, hi = P.wilson(90, 100)
    # (p + z^2/2n) / (1 + z^2/n) -+ z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n), z = 1.96
    assert lo == pytest.approx(0.8256, abs=5e-4) and hi == pytest.approx(
        0.9448, abs=5e-4
    )
    assert all(np.isnan(v) for v in P.wilson(0, 0))
    assert P.wilson(0, 10)[0] == 0.0 and P.wilson(10, 10)[1] == 1.0


def test_zero_network_and_cache_read_without_the_writer_lock(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_network(monkeypatch)
    before = world.cache_path.read_bytes()
    with ResultCache(world.cache_path):  # a runner holds the exclusive lock meanwhile
        rep = world.report()
    assert rep["entries"][0]["n"] == SAMPLE
    assert world.cache_path.read_bytes() == before  # read-only
    # the forbidden sockets really are armed
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.invalid", 80))


def test_module_imports_no_http_client_in_a_clean_interpreter() -> None:
    code = (
        "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]];"
        "import astro_triage.evaluation.layer2_pilot as m;"
        "bad = [k for k in sys.modules if k in ('astro_triage.llm.client',"
        " 'astro_triage.dashboard')];"
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(REPO / "src"), str(REPO)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_entry_counts_pending_rates_wilson_and_tokens(world: World) -> None:
    e = world.report()["entries"][0]
    assert (e["n"], e["pending"]) == (SAMPLE, N_PENDING)
    assert e["status_counts"] == {"OK": 94, "UNKNOWN": 3, "pending": 3}
    assert e["status_rates"]["OK"] == pytest.approx(0.94)
    af, asel = e["accuracy_forced"], e["accuracy_selective"]
    assert (af["k"], af["n"]) == (90, 100)  # pending and UNKNOWN count as wrong
    assert (af["lo"], af["hi"]) == pytest.approx((0.8256, 0.9448), abs=5e-4)
    assert (asel["k"], asel["n"]) == (90, 94)
    assert asel["point"] == pytest.approx(90 / 94)
    assert e["unknown_rate"] == pytest.approx(0.03)
    assert e["total_tokens"]["mean"] == 107 and e["total_tokens"]["p95"] == 107
    assert e["prompt_tokens"]["p50"] == 100 and e["completion_tokens"]["p50"] == 7
    assert e["latency_ms"]["n"] == 97 and e["latency_ms"]["p50"] == LATENCY
    assert e["attempts"]["mean"] == 1
    assert e["model_reported"] == {MODEL: 97}
    assert 0.0 <= e["macro_f1_forced"] <= 1.0 and 0.0 <= e["macro_f1_layer1"] <= 1.0


def test_ledger_is_filtered_by_model_arm_and_thinking(world: World) -> None:
    rep = world.report()
    lg = rep["entries"][0]["ledger"]
    assert (
        lg["http_attempts"] == 99
    )  # 97 + the 429 + the retry; thinking-on/legacy/other excluded
    assert (lg["http_429"], lg["retries"]) == (1, 1)
    (legacy,) = rep["ledger_legacy"]
    assert legacy["thinking"] == "unknown" and legacy["http_attempts"] == 2


def test_projection_hours_use_the_rpm_pacing_floor(world: World) -> None:
    rep = world.report(rpm_cap=30.0)
    t = rep["target"]
    n = round(F_VAL * (t["n_val"] + t["n_test"]))
    assert (t["n_val"], t["n_test"], t["objects"], t["f_val_source"]) == (
        900,
        400,
        n,
        "target release",
    )
    p = rep["entries"][0]["projection"]["target_all_routed"]
    assert p["objects"] == n
    assert p["hours_p50"] == pytest.approx(n * 2 / 3600)  # max(60/30, 1 s) = 2 s
    assert p["hours_p95"] == pytest.approx(n * 2 / 3600)
    assert p["calls_p50"] == n * 1 + P.CANARY_N * P.PLANNED_RUNS
    assert p["tokens_p50"] == n * 107
    slow = world.report(rpm_cap=6000.0)["entries"][0]["projection"]["target_all_routed"]
    assert slow["hours_p50"] == pytest.approx(n * 1.0 / 3600)  # latency now dominates
    fr = rep["entries"][0]["projection"]["pixel_fraction"]
    assert [fr[k]["objects"] for k in ("0.05", "0.1", "0.25", "0.5", "1.0")] == [
        round(q * n) for q in (0.05, 0.10, 0.25, 0.5, 1.0)
    ]


def test_projection_band_p50_to_p95_when_latency_varies(tmp_path: Path) -> None:
    w = World(tmp_path)
    w.cache_path.unlink()  # refill with spread latencies
    with ResultCache(w.cache_path) as cache:
        for i, (_, row) in enumerate(w.rows.iterrows()):
            cache.put(
                request_for(row, MODEL, "blind", False)[1],
                {
                    "source_id": int(row["source_id"]),
                    "status": "OK",
                    "label_l2": w.y[i],
                    "confidence_raw": 80,
                    "model_reported": MODEL,
                    "prompt_tokens": 100,
                    "completion_tokens": 7 + i,
                    "latency_ms": 1000 + 100 * i,
                    "attempts": 1 + (i >= 95),
                },
            )
    e = w.report(rpm_cap=6000.0)["entries"][0]
    lat = np.array([1000 + 100 * i for i in range(SAMPLE)], dtype=float)
    assert e["latency_ms"]["p50"] == pytest.approx(np.percentile(lat, 50))
    assert e["latency_ms"]["p95"] == pytest.approx(np.percentile(lat, 95))
    p = e["projection"]["target_all_routed"]
    assert p["hours_p50"] < p["hours_p95"] and p["tokens_p50"] < p["tokens_p95"]
    assert p["calls_p50"] < p["calls_p95"]
    ok = e["projection"]["expected_ok_val_answers"]
    assert (
        ok["point"] == pytest.approx(F_VAL * 900) and ok["calibration_minimum"] == 100
    )
    assert ok["lo"] < ok["point"] <= ok["hi"] + 1e-9  # all 100 OK: Wilson hi is 1


def test_truncated_cache_line_is_tolerated(tmp_path: Path) -> None:
    w = World(tmp_path)
    with w.cache_path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "x", "stat')  # a line being appended
    assert w.report()["entries"][0]["pending"] == N_PENDING


def test_thinking_models_get_a_separate_entry_that_is_all_pending(world: World) -> None:
    rep = P.pilot_report(
        world.r.rel, [MODEL], ["blind"], [MODEL], SAMPLE, world.r.rel, 30.0,
        cache_dir=world.cache_dir, n_boot=10,
    )  # fmt: skip
    off, on = rep["entries"]
    assert (off["thinking"], on["thinking"]) == (False, True)
    assert on["pending"] == SAMPLE and on["status_counts"] == {"pending": SAMPLE}
    assert on["accuracy_forced"]["k"] == 0  # reported, never dropped


def test_markdown_floor_sentence_no_ranking_words_no_label_columns(
    world: World,
) -> None:
    rep = world.report()
    md = P.render_pilot_markdown(rep)
    assert (
        "The reporting floor (>= 100 objects in >= 20 pixels per class) is not reachable "
        "with 100 objects; all pilot numbers are descriptive." in md
    )
    low = md.lower()
    assert not [w for w in FORBIDDEN_WORDS if w in low]
    assert "sdss" not in low and "source_id" not in low
    blob = json.dumps(rep, default=str).lower()
    assert "sdss" not in blob and "source_id" not in blob and "healpix" not in blob
    assert "calibration_minimum" in blob  # K13 projection present
    assert "against a minimum of 100" in md and "p50..p95" in md


def test_ranking_word_in_text_refuses_to_render(world: World) -> None:
    rep = world.report()
    rep["entries"][0]["model"] = f"the {FORBIDDEN_WORDS[0]} model"
    with pytest.raises(ValueError, match="ranking words"):
        P.render_pilot_markdown(rep)


# ------------------------------------------------------------------------------------ CLI
def _cli(w: World, *extra: str) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "l2_pilot_cli", REPO / "scripts" / "layer2_pilot_report.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(
        [
            "--release", "rel", "--models", MODEL, "--arms", "blind",
            "--sample", str(SAMPLE), "--target-release", "rel",
            "--base", str(w.r.base), "--cache-dir", str(w.cache_dir), *extra,
        ]
    )  # fmt: skip


def test_cli_writes_json_and_md_atomically_with_no_network(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_network(monkeypatch)
    assert _cli(world) == 0
    d = world.r.rel / "layer2"
    assert capsys.readouterr().out.split() == ["pilot_report.json", "pilot_report.md"]
    data = json.loads((d / "pilot_report.json").read_text("utf-8"))
    assert len(data["entries"]) == 1 and data["entries"][0]["pending"] == N_PENDING
    assert "reporting floor" in (d / "pilot_report.md").read_text("utf-8")
    assert not list(d.glob("*.tmp"))


def test_cli_failure_exits_1_and_bad_rpm_cap_exits_2(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = World(tmp_path)
    (broken.r.rel / "layer2" / MANIFEST_NAME).unlink()
    assert _cli(broken) == 1
    assert "pilot report failed" in capsys.readouterr().err
    assert not (broken.r.rel / "layer2" / "pilot_report.json").exists()
    with pytest.raises(SystemExit) as exc:
        _cli(world, "--rpm-cap", "0")
    assert exc.value.code == 2


def test_pilot_frame_is_deterministic(world: World) -> None:
    a = (
        json.dumps(P.wilson(1, 2)),
        json.dumps(world.report()["entries"][0], sort_keys=True, default=str),
    )
    b = (
        json.dumps(P.wilson(1, 2)),
        json.dumps(world.report()["entries"][0], sort_keys=True, default=str),
    )
    assert a == b
