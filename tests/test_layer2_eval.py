"""Proofs for the Layer 2 evaluation core and data loading (ADR 0005 D7, T-L2-T3).

Core: a 12-object toy whose metrics are derived by hand (see ``test_toy_*``), the reporting
floor, the shared bootstrap replicates and the pre-registered headline rule. Data: a synthetic
release written to ``tmp_path`` (real pixel map, fake Layer 2 parquets, no network, no key, no
``data/``) that exercises every refusal of ``layer2_data`` and of ``scripts/layer2_evaluate.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astro_triage.abstention.policy import DEFAULT_GRID
from astro_triage.evaluation import layer2 as L2
from astro_triage.evaluation import layer2_data as D
from astro_triage.evaluation.runner import EvalError
from astro_triage.llm.consolidate import SCHEMA, output_path
from astro_triage.llm.prompt import FEATURE_COLUMNS, PROMPT_VERSION, TEMPLATE_SHA256
from astro_triage.split import build_pixel_map
from eval import metrics as M

REPO = Path(__file__).resolve().parents[1]
MODEL = "nvidia/fake-model"
CLS = np.array(["STAR", "GALAXY", "QSO"])
FAKE_KEY = "FAKE-KEY-123"

# ------------------------------------------------------------------------------- toy core
# Hand derivation (tau 0.6, tau2 0.7, tau* 0.55), 4 objects per class so every weight is 1/12.
#   y    S S S S G G G G Q Q Q Q
#   L1   S S Q Q G G G S Q Q S S     conf .9 .9 .5 .5 .9 .9 .55 .9 .9 .9 .5 .5
# S0 hits: idx 0,1 | 4,5,6 | 8,9 = 7/12.  STAR: pred 5, TP 2 -> P .4 R .5 F1 4/9;
#   GALAXY: pred 3, TP 3 -> P 1 R .75 F1 6/7; QSO: pred 4, TP 2 -> P .5 R .5 F1 .5.
# Routed (conf < .6): idx 2 (OK STAR .8), 3 (UNKNOWN), 6 (OK GALAXY .5), 10 (OK QSO .9),
#   11 (TIMEOUT).  With tau2 .7 only idx 2 and 10 are trusted.
# S2 (fallback to L1): final S S S Q G G G S Q Q Q S -> 9/12; F1 STAR 2/3, GALAXY 6/7, QSO .75.
# S3 (abstain): idx 3, 6, 11 abstain -> coverage 9/12, hits 8, accuracy_answered 8/9.
Y = np.array(list("SSSSGGGGQQQQ"))
_FULL = {"S": "STAR", "G": "GALAXY", "Q": "QSO"}
TOY_Y = np.array([_FULL[c] for c in Y])
TOY_L1 = np.array([_FULL[c] for c in "SSQQGGGSQQSS"])
TOY_CONF = np.array([0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.55, 0.9, 0.9, 0.9, 0.5, 0.5])
TOY_STATUS = np.array(
    ["NOT_ROUTED"] * 2
    + ["OK", "UNKNOWN"]
    + ["NOT_ROUTED"] * 2
    + ["OK"]
    + ["NOT_ROUTED"] * 3
    + ["OK", "TIMEOUT"],
    dtype=object,
)
TOY_LABEL = np.array(
    ["", "", "STAR", "UNKNOWN", "", "", "GALAXY", "", "", "", "QSO", ""], dtype=object
)
TOY_S = np.full(12, np.nan)
TOY_S[[2, 6, 10]] = [0.8, 0.5, 0.9]


def toy(
    reps: int = 1,
    tau_star: float = 0.55,
    status: Any = None,
    label: Any = None,
    s_l2: Any = None,
) -> L2.Layer2Inputs:
    k = reps
    return L2.Layer2Inputs(
        y=np.tile(TOY_Y, k),
        pixel=np.arange(12 * k),
        band=np.array(["A"] * (12 * k)),
        l1_label=np.tile(TOY_L1, k),
        l1_conf=np.tile(TOY_CONF, k),
        l2_status=np.tile(TOY_STATUS if status is None else status, k),
        l2_label=np.tile(TOY_LABEL if label is None else label, k),
        s_l2=np.tile(TOY_S if s_l2 is None else s_l2, k),
        tau=0.6,
        tau2=0.7,
        tau_star=tau_star,
    )


@cache
def toy_eval(reps: int) -> dict:
    return L2.evaluate(toy(reps), n_boot=50)


def pt(res: dict, unit: str, key: str) -> float:
    return res["units"][unit]["metrics"][key]["point"]


def test_toy_hand_values_s0_and_s2() -> None:
    res = toy_eval(1)
    assert pt(res, "S0", "accuracy") == pytest.approx(7 / 12)
    assert pt(res, "S0", "f1/STAR") == pytest.approx(4 / 9)
    assert pt(res, "S0", "f1/GALAXY") == pytest.approx(6 / 7)
    assert pt(res, "S0", "f1/QSO") == pytest.approx(0.5)
    assert pt(res, "S0", "macro_f1") == pytest.approx((4 / 9 + 6 / 7 + 0.5) / 3)
    assert pt(res, "S0", "coverage") == 1.0
    assert pt(res, "S2", "accuracy") == pytest.approx(0.75)
    assert pt(res, "S2", "f1/STAR") == pytest.approx(2 / 3)
    assert pt(res, "S2", "f1/QSO") == pytest.approx(0.75)
    assert pt(res, "S2", "macro_f1") == pytest.approx((2 / 3 + 6 / 7 + 0.75) / 3)
    assert pt(res, "S2", "coverage") == 1.0  # fallback to Layer 1 answers everything


def test_toy_reported_values_all_systems() -> None:
    res = toy_eval(1)
    rows = {
        "S0": {"accuracy": 0.5833, "macro_f1": 0.6005, "ece": 0.2292},
        "S1": {
            "accuracy": 0.5,
            "macro_f1": 0.6349,
            "coverage": 0.5833,
            "accuracy_answered": 0.8571,
            "ece": 0.0429,  # one bin: |6/7 - .9|
        },
        "S1*": {"accuracy": 0.5833, "macro_f1": 0.6984, "coverage": 0.6667},
        "S2": {"accuracy": 0.75, "macro_f1": 0.7579},
        "S3": {
            "accuracy": 0.6667,
            "macro_f1": 0.7579,
            "coverage": 0.75,
            "accuracy_answered": 0.8889,
        },
    }
    for unit, vals in rows.items():
        for key, want in vals.items():
            assert pt(res, unit, key) == pytest.approx(want, abs=6e-5), (unit, key)


def test_toy_routed_views_keep_every_routed_object() -> None:
    res = toy_eval(1)
    forced, sel = res["units"]["L2_forced"], res["units"]["L2_selective"]
    assert (forced["n"], forced["n_answered"]) == (5, 3)
    assert (sel["n"], sel["n_answered"]) == (3, 3)
    # routed classes S2 G1 Q2 -> weights 1/6, 1/3, 1/6; hits idx 2, 6, 10 = 2/3, all answered 2/3
    assert pt(res, "L2_forced", "accuracy") == pytest.approx(2 / 3)
    assert pt(res, "L2_forced", "coverage") == pytest.approx(2 / 3)
    assert pt(res, "L2_forced", "accuracy_answered") == pytest.approx(1.0)
    assert forced["raw_coverage"] == pytest.approx(3 / 5)
    assert res["units"]["L1_routed"]["n"] == 5


def test_abstentions_and_failures_stay_in_the_denominator() -> None:
    res = toy_eval(1)
    # S3: 9 of 12 answered, 8 hits: accuracy over ALL 12, answered-only over the 9
    assert pt(res, "S3", "accuracy") == pytest.approx(8 / 12)
    assert pt(res, "S3", "accuracy_answered") == pytest.approx(8 / 9)
    assert pt(res, "S3", "coverage") == pytest.approx(9 / 12)
    # every routed call fails: S3 degenerates to S1 (7 answered), never to a smaller sample
    dead = np.array(
        ["TIMEOUT" if s != "NOT_ROUTED" else s for s in TOY_STATUS], dtype=object
    )
    inp = toy(
        status=dead, label=np.full(12, "", dtype=object), s_l2=np.full(12, np.nan)
    )
    out = L2.evaluate(inp, n_boot=20)
    assert pt(out, "S3", "coverage") == pytest.approx(7 / 12)
    assert pt(out, "S3", "accuracy") == pytest.approx(6 / 12)
    assert pt(out, "S3", "accuracy_answered") == pytest.approx(6 / 7)
    assert out["units"]["S3"]["n"] == 12
    assert pt(out, "L2_forced", "accuracy") == 0.0
    assert pt(out, "L2_forced", "coverage") == 0.0


def test_a_label_outside_the_classes_is_a_non_answer() -> None:
    # status OK but a garbage label: never counted as an answer, never dropped
    label = TOY_LABEL.copy()
    label[2] = "PLANET"
    out = L2.evaluate(toy(label=label), n_boot=20)
    assert out["units"]["S3"]["n_answered"] == 8  # idx 2 joins the abstentions 3, 6, 11
    assert pt(out, "S3", "coverage") == pytest.approx(8 / 12)
    assert pt(out, "S2", "accuracy") == pytest.approx(
        8 / 12
    )  # idx 2 falls back to wrong L1 (QSO)
    assert out["units"]["L2_forced"]["n"] == 5
    assert (
        out["units"]["L2_forced"]["n_answered"] == 2
    )  # PLANET is not an answer there either


# ------------------------------------------------------------------------------- floor
def test_floor_n12_is_na_without_raw_value() -> None:
    res = toy_eval(1)
    for unit in ("S0", "S3"):
        for key in ("macro_f1", "accuracy", "f1/STAR"):
            cell = res["units"][unit]["metrics"][key]
            assert cell["status"] == "n/a (n=4, pixels=4)", (unit, key)
            assert cell["lo"] is None and cell["hi"] is None
    # routed view: GALAXY has 1 object, the weakest class is the one named
    assert (
        res["units"]["L2_forced"]["metrics"]["macro_f1"]["status"]
        == "n/a (n=1, pixels=1)"
    )
    for pair, cells in res["diffs"].items():
        assert cells["macro_f1"]["status"].startswith("n/a"), pair
    assert res["floor"]["all"]["STAR"] == {"n": 4, "pixels": 4, "passes": False}


def test_floor_360_objects_passes_and_small_routed_subset_is_na() -> None:
    res = toy_eval(30)
    assert res["floor"]["all"]["QSO"] == {"n": 120, "pixels": 120, "passes": True}
    assert res["units"]["S0"]["metrics"]["macro_f1"]["status"] == "ok"
    assert res["units"]["S3"]["metrics"]["f1/STAR"]["status"] == "ok"
    # routed: STAR 60, GALAXY 30, QSO 60 -> below 100 -> n/a with the weakest class named
    cell = res["units"]["L2_forced"]["metrics"]["macro_f1"]
    assert cell["status"] == "n/a (n=30, pixels=30)"
    assert (
        res["units"]["L2_forced"]["metrics"]["f1/QSO"]["status"]
        == "n/a (n=60, pixels=60)"
    )
    assert (
        res["units"]["L2_selective"]["metrics"]["macro_f1"]["status"]
        == "n/a (n=30, pixels=30)"
    )
    assert res["diffs"]["L2_forced - L1_routed"]["macro_f1"]["status"].startswith("n/a")


def test_routed_ok_view_with_an_empty_class_does_not_crash() -> None:
    status = TOY_STATUS.copy()
    status[[2, 6]] = "TIMEOUT"  # the OK-routed view now holds QSO only
    out = L2.evaluate(toy(30, status=status), n_boot=10)
    cell = out["units"]["L2_selective"]["metrics"]["macro_f1"]
    assert cell["status"] == "n/a (n=0, pixels=0)"
    assert np.isnan(cell["point"])
    assert out["units"]["L2_forced"]["n"] == 150


# ----------------------------------------------------------------- shared bootstrap draws
def test_replicates_are_shared_between_systems() -> None:
    inp, n_boot = toy(30), 50
    res = toy_eval(30)
    units, _ = L2.build_units(inp)
    rng = np.random.default_rng(M.SEED)
    acc: dict[str, list[float]] = {"S0": [], "S3": [], "S1*": []}
    for _ in range(n_boot):
        idx = M.bootstrap_pixel_draw(inp.pixel, inp.band, rng)
        w = M.regime_weights(inp.y[idx])
        for name, store in acc.items():
            p = units[name].pred[idx]
            hit = np.isin(p, M.CLASSES) & (p == inp.y[idx])
            store.append(float(w[hit].sum() / w.sum()))
    for name, vals in acc.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        cell = res["units"][name]["metrics"]["accuracy"]
        assert cell["lo"] == pytest.approx(lo, abs=1e-12), name
        assert cell["hi"] == pytest.approx(hi, abs=1e-12), name


def test_paired_ci_is_replicate_by_replicate_and_coherent() -> None:
    inp, n_boot = toy(30), 50
    res = toy_eval(30)
    units, _ = L2.build_units(inp)
    rng = np.random.default_rng(M.SEED)
    d = []
    for _ in range(n_boot):
        idx = M.bootstrap_pixel_draw(inp.pixel, inp.band, rng)
        w = M.regime_weights(inp.y[idx])
        f = {
            n: L2.unit_metrics(inp.y[idx], units[n].pred[idx], units[n].conf[idx], w)[
                "macro_f1"
            ]
            for n in ("S2", "S0")
        }
        d.append(f["S2"] - f["S0"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    cell = res["diffs"]["S2 - S0"]["macro_f1"]
    assert cell["lo"] == pytest.approx(lo, abs=1e-12)
    assert cell["hi"] == pytest.approx(hi, abs=1e-12)
    assert cell["point"] == pytest.approx(
        pt(res, "S2", "macro_f1") - pt(res, "S0", "macro_f1"), abs=1e-12
    )
    assert cell["point"] == pytest.approx(
        (2 / 3 + 6 / 7 + 0.75) / 3 - (4 / 9 + 6 / 7 + 0.5) / 3
    )
    assert cell["lo"] <= cell["point"] <= cell["hi"]
    assert res["seed"] == M.SEED and res["n_boot"] == n_boot


def test_same_seed_is_deterministic_and_another_seed_moves_the_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = L2.evaluate(toy(30), n_boot=30)
    b = L2.evaluate(toy(30), n_boot=30)
    assert json.dumps(a, default=str) == json.dumps(b, default=str)  # NaN-safe equality
    monkeypatch.setattr(M, "SEED", M.SEED + 1)
    c = L2.evaluate(toy(30), n_boot=30)
    ca = a["units"]["S0"]["metrics"]["accuracy"]
    cc = c["units"]["S0"]["metrics"]["accuracy"]
    assert ca["point"] == cc["point"]
    assert (ca["lo"], ca["hi"]) != (cc["lo"], cc["hi"])


# ------------------------------------------------------------------- pre-registered rule
def cell(lo: float | None, status: str = "ok") -> dict[str, object]:
    return {"point": 0.1, "lo": lo, "hi": 0.3, "n_invalid": 0, "status": status}


def diffs_with(lo_a: float | None, lo_b: float | None, st_a="ok", st_b="ok") -> dict:
    return {
        "S2 - S0": {L2.HEADLINE_METRIC: cell(lo_a, st_a)},
        "S3 - S1*": {L2.HEADLINE_METRIC: cell(lo_b, st_b)},
    }


def test_headline_rule_is_fixed() -> None:
    assert L2.HEADLINE_METRIC == "macro_f1"
    assert L2.HEADLINE_PAIRS == (("S2", "S0"), ("S3", "S1*"))
    assert L2.PAIRS[:2] == L2.HEADLINE_PAIRS


@pytest.mark.parametrize(
    ("lo_a", "lo_b", "st_a", "st_b", "outcome"),
    [
        (0.01, -0.02, "ok", "ok", "gain"),  # one pair is enough
        (-0.02, 0.01, "ok", "ok", "gain"),
        (0.0, 0.0, "ok", "ok", "no_gain"),  # a limit equal to 0 is not above 0
        (-0.001, -0.5, "ok", "ok", "no_gain"),
        (None, None, "n/a (n=4, pixels=4)", "n/a (floor)", "no_gain"),
        (
            0.2,
            None,
            "n/a (n=4, pixels=4)",
            "ok",
            "no_gain",
        ),  # positive but below the floor
    ],
)
def test_verdict_requires_floor_and_lower_limit_above_zero(
    lo_a, lo_b, st_a, st_b, outcome
) -> None:
    v = L2.headline_verdict(diffs_with(lo_a, lo_b, st_a, st_b))
    assert v["outcome"] == outcome
    assert v["metric"] == "macro_f1"
    if st_a != "ok":
        assert v["pairs"]["S2 - S0"]["gain"] is False
        assert v["pairs"]["S2 - S0"]["evaluable"] is False


def test_verdict_end_to_end_gain_and_no_gain_branches() -> None:
    big = toy_eval(30)
    assert big["verdict"]["outcome"] == "gain"
    assert big["verdict"]["pairs"]["S2 - S0"]["gain"] is True
    assert big["diffs"]["S2 - S0"]["macro_f1"]["lo"] > 0
    # the 12-object toy has a positive S2-S0 point but fails the floor: no gain
    small = toy_eval(1)
    assert small["diffs"]["S2 - S0"]["macro_f1"]["point"] > 0
    assert small["verdict"]["outcome"] == "no_gain"
    assert small["verdict"]["n_evaluable"] == 0
    # a dead Layer 2 on a floor-passing sample: S2 == S0, S3 == S1* -> difference exactly 0
    dead = np.array(
        ["TIMEOUT" if s != "NOT_ROUTED" else s for s in TOY_STATUS], dtype=object
    )
    inp = toy(
        30,
        tau_star=0.6,
        status=dead,
        label=np.full(12, "", dtype=object),
        s_l2=np.full(12, np.nan),
    )
    out = L2.evaluate(inp, n_boot=30)
    for pair in ("S2 - S0", "S3 - S1*"):
        c = out["diffs"][pair]["macro_f1"]
        assert (c["point"], c["lo"], c["hi"]) == (0.0, 0.0, 0.0)
    assert out["verdict"]["outcome"] == "no_gain"
    assert out["verdict"]["n_evaluable"] == 2


def test_risk_coverage_is_monotone_on_the_grid() -> None:
    rows = L2.risk_coverage(toy(1))
    assert [r["tau"] for r in rows] == [float(t) for t in DEFAULT_GRID]
    cov = [r["coverage"] for r in rows]
    assert cov == sorted(cov, reverse=True)
    assert rows[0]["coverage"] == 1.0  # tau .50: every conf >= .5 is kept
    assert np.isnan(rows[-1]["accuracy"])  # tau .99: nothing kept


# ============================================================================ data + refusals
@cache
def _pixels() -> tuple[np.ndarray, np.ndarray]:
    pmap = build_pixel_map()
    return (
        pmap.loc[pmap["fold"] == "test", "healpix_l5"].to_numpy(),
        pmap.loc[pmap["fold"] == "train", "healpix_l5"].to_numpy(),
    )


def sid(pix: int, j: int) -> int:
    return int(pix) * 2**35 * 4**7 + j


@dataclass
class Rel:
    base: Path
    rel: Path
    ids: np.ndarray
    fold: list[str]
    conf: np.ndarray
    y: np.ndarray
    tau: float
    kw: dict = field(default_factory=dict)

    @property
    def manifest(self) -> Path:
        return self.rel / "layer2" / "routing_v1.json"

    @property
    def preds(self) -> Path:
        return self.rel / "predictions_lgbm_full_v0.parquet"

    def out(self, fold: str, arm: str = "blind", thinking: bool = False) -> Path:
        # ADR 0005 17.2: the file name carries think-on / think-off
        return output_path(self.rel, fold, MODEL, arm, enable_thinking=thinking)

    def sha(self) -> str:
        return hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def write_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "predictions_file": self.preds.name,
                    "predictions_sha256": hashlib.sha256(
                        self.preds.read_bytes()
                    ).hexdigest(),
                    "tau": self.tau,
                    "a_keep2": 0.8,
                }
            ),
            encoding="utf-8",
        )

    def routed_ids(self, fold: str) -> np.ndarray:
        sel = [
            i for i, f in enumerate(self.fold) if f == fold and self.conf[i] < self.tau
        ]
        return np.sort(self.ids[sel])

    def unrouted_test_ids(self) -> np.ndarray:
        sel = [
            i
            for i, f in enumerate(self.fold)
            if f == "test" and self.conf[i] >= self.tau
        ]
        return self.ids[sel]

    def l2_table(self, fold: str, acc: float) -> pa.Table:
        rng = np.random.default_rng({"val": 11, "test": 12}[fold])
        pos = {int(s): i for i, s in enumerate(self.ids)}
        sel = [pos[int(s)] for s in self.routed_ids(fold)]
        n = len(sel)
        st = np.array(["OK"] * n, dtype=object)
        lab = np.empty(n, dtype=object)
        cr = np.empty(n, dtype=object)
        for t, i in enumerate(sel):
            u = rng.random()
            if u < 0.1:
                st[t] = "TIMEOUT"
            elif u < 0.2:
                st[t] = "UNKNOWN"
            if st[t] == "TIMEOUT":
                lab[t], cr[t] = None, None
                continue
            right = rng.random() < acc
            wrong = CLS[(int(np.searchsorted(CLS, self.y[i])) + 1) % 3]
            lab[t] = (
                "UNKNOWN" if st[t] == "UNKNOWN" else (self.y[i] if right else wrong)
            )
            cr[t] = int(np.clip(rng.normal(85 if right else 70, 8), 0, 100))
        dic = pa.dictionary(pa.int8(), pa.string())
        return pa.Table.from_arrays(
            [
                pa.array(self.ids[sel], type=pa.int64()),
                pa.array(self.conf[sel], type=pa.float32()),
                pa.array(list(st), type=dic),
                pa.array(list(lab), type=dic),
                pa.array(list(cr), type=pa.int8()),
                pa.array([MODEL] * n),
                pa.array([200 + t for t in range(n)], type=pa.int32()),
                pa.array([10] * n, type=pa.int32()),
                pa.array([900 + t for t in range(n)], type=pa.int32()),
                pa.array([1] * n, type=pa.int8()),
                pa.array(["stop"] * n),
                pa.array(
                    [datetime(2026, 10, 27, tzinfo=UTC)] * n,
                    type=pa.timestamp("ms", tz="UTC"),
                ),
            ],
            schema=SCHEMA,
        )

    def meta(self, fold: str) -> dict[str, str]:
        return {
            "schema_version": "1",
            "release": "rel",
            "fold": fold,
            "model_requested": MODEL,
            "prompt_version": PROMPT_VERSION,
            "template_sha256": TEMPLATE_SHA256,
            "arm": "blind",
            "enable_thinking": "false",
            "max_tokens": "64",
            "temperature": "0",
            "endpoint": "x",
            "routing_manifest_sha256": self.sha(),
            "created_utc": "2026-10-27T12:00:00+00:00",
            "git_sha": "abc",
        }

    def write_l2(self, fold: str, acc: float) -> None:
        tbl = self.l2_table(fold, acc).replace_schema_metadata(self.meta(fold))
        pq.write_table(tbl, self.out(fold))

    def edit(
        self, fold: str, meta: dict[str, str | None] | None = None, table=None
    ) -> None:
        """Rewrite a Layer 2 parquet: patch metadata (None deletes a key) and/or the table."""
        path = self.out(fold)
        tbl = pq.read_table(path)
        md = {k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()}
        for k, v in (meta or {}).items():
            if v is None:
                md.pop(k, None)
            else:
                md[k] = v
        if table is not None:
            tbl = table(tbl)
        pq.write_table(tbl.replace_schema_metadata(md), path)


def build_release(
    base: Path,
    *,
    n_test_pix: int = 40,
    per_pix: int = 10,
    n_val: int = 900,
    val_acc: float = 0.8,
    test_acc: float = 0.8,
) -> Rel:
    """Synthetic release ``base/rel``; Layer 1 right 70% of the time, tau 0.7."""
    test_pix, train_pix = _pixels()
    test_pix, train_pix = test_pix[:n_test_pix], train_pix[:5]
    rng = np.random.default_rng(1)
    rows = [(sid(p, j), "test", int(p)) for p in test_pix for j in range(per_pix)]
    rows += [
        (sid(train_pix[j % 5], 1000 + j), "val", int(train_pix[j % 5]))
        for j in range(n_val)
    ]
    ids = np.array([r[0] for r in rows], dtype="int64")
    fold = [r[1] for r in rows]
    y = CLS[np.arange(len(rows)) % 3]
    rel = base / "rel"
    (rel / "layer2").mkdir(parents=True)
    pd.DataFrame(
        {
            "source_id": ids,
            "sdss_class": y,
            "fold": fold,
            "healpix_l5": np.array([r[2] for r in rows], dtype="int64"),
        }
    ).to_parquet(rel / "labels.parquet")
    feats = pd.DataFrame(
        {c: rng.normal(size=len(ids)).astype("float32") for c in FEATURE_COLUMNS}
    )
    feats["astrometric_params_solved"] = 95.0
    feats["phot_g_mean_mag"] = rng.uniform(15, 20, size=len(ids)).astype("float32")
    feats.insert(0, "source_id", ids)
    feats.to_parquet(rel / "features.parquet")
    conf = rng.uniform(0.4, 1.0, size=len(ids))
    right = rng.random(len(ids)) < 0.7
    top = np.where(right, y, CLS[(np.searchsorted(CLS, y) + 1) % 3])
    probs = np.zeros((len(ids), 3))
    col = {"STAR": 0, "GALAXY": 1, "QSO": 2}
    for i in range(len(ids)):
        probs[i] = (1 - conf[i]) / 2
        probs[i, col[top[i]]] = conf[i]
    pd.DataFrame(
        {
            "source_id": ids,
            "p_cal_star": probs[:, 0],
            "p_cal_qso": probs[:, 2],
            "p_cal_galaxy": probs[:, 1],
        }
    ).to_parquet(rel / "predictions_lgbm_full_v0.parquet")
    r = Rel(base, rel, ids, fold, conf, y, 0.7)
    r.write_manifest()
    r.write_l2("val", val_acc)
    r.write_l2("test", test_acc)
    return r


def load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "l2_eval_cli", REPO / "scripts" / "layer2_evaluate.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cli(r: Rel, *extra: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    argv = [
        "--release",
        "rel",
        "--model",
        MODEL,
        "--base",
        str(r.base),
        "--n-boot",
        "20",
        *extra,
    ]
    with redirect_stdout(out), redirect_stderr(err):
        code = load_cli().main(argv)
    return code, out.getvalue(), err.getvalue()


def nothing_written(r: Rel) -> bool:
    d = r.rel / "layer2"
    return not any(d.glob("eval_l2*"))


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Rel:
    return build_release(tmp_path_factory.mktemp("l2rel"))


@pytest.fixture
def fresh(tmp_path: Path) -> Rel:
    return build_release(tmp_path)


def test_happy_path_header_and_fit_from_val(release: Rel) -> None:
    res = D.run_evaluation(
        release.rel, MODEL, "blind", enable_thinking=False, n_boot=20
    )
    meta, fit = res["meta"], res["fit"]
    assert meta["header"].startswith("PROVISIONAL")
    assert "400 test rows in 40 level-5 pixels" in meta["header"]
    assert meta["n_test"] == 400 and meta["regime"] == "BALANCED"
    assert fit["calibration_status"] == "calibrated" and fit["tau2"] is not None
    assert fit["tau"] == 0.7 and fit["a_keep2"] == 0.8
    assert res["operations"]["n_routed"] == len(release.routed_ids("test"))
    assert res["eval"]["units"]["L2_forced"]["n"] == len(release.routed_ids("test"))
    assert res["eval"]["units"]["S0"]["n"] == 400


def test_tau2_is_the_smallest_grid_value_reaching_a_keep2_on_val(release: Rel) -> None:
    fit = D.run_evaluation(
        release.rel, MODEL, "blind", enable_thinking=False, n_boot=5
    )["fit"]
    tbl = pq.read_table(release.out("val")).to_pandas()
    pos = {int(s): i for i, s in enumerate(release.ids)}
    ok = tbl[tbl["status"].astype(str) == "OK"]
    raw = ok["confidence_raw"].to_numpy(dtype=float) / 100.0
    correct = np.array(
        [
            lab == release.y[pos[int(s)]]
            for s, lab in zip(ok["source_id"], ok["label_l2"])
        ]
    )
    assert fit["n_val_ok"] == len(ok)
    calib = fit["verbal_calibration"]
    s = np.array(
        [
            calib["values"][int(np.searchsorted(calib["edges"], v, side="right"))]
            for v in raw
        ]
    )
    reach = [
        t
        for t in DEFAULT_GRID
        if correct[s >= t].size and correct[s >= t].mean() >= 0.8
    ]
    assert fit["tau2"] == pytest.approx(float(reach[0]))


def test_fit_uses_val_only(tmp_path: Path) -> None:
    a = build_release(tmp_path / "a", test_acc=0.8)
    b = build_release(tmp_path / "b", test_acc=0.2)  # only the TEST answers differ
    ra = D.run_evaluation(a.rel, MODEL, "blind", enable_thinking=False, n_boot=10)
    rb = D.run_evaluation(b.rel, MODEL, "blind", enable_thinking=False, n_boot=10)
    assert ra["fit"] == rb["fit"]
    assert (
        ra["eval"]["units"]["S3"]["metrics"]["accuracy"]["point"]
        != rb["eval"]["units"]["S3"]["metrics"]["accuracy"]["point"]
    )


def test_fewer_than_100_ok_val_answers_is_reported_uncalibrated_not_refused(
    tmp_path: Path,
) -> None:
    # ADR D4.5: s_l2 is NULL and tau2 not applied (the evaluation still runs)
    r = build_release(tmp_path, n_val=150)
    res = D.run_evaluation(r.rel, MODEL, "blind", enable_thinking=False, n_boot=10)
    assert res["fit"]["n_val_ok"] < 100
    assert res["fit"]["tau2"] is None
    assert res["fit"]["calibration_status"].startswith("uncalibrated")


def test_tau2_infeasible_is_not_refused_any_more(tmp_path: Path) -> None:
    # ADR 17.1.2 (R1-001): the old refusal "tau2 cannot be fitted" is gone
    r = build_release(tmp_path, val_acc=0.0)  # no grid value reaches A_keep2
    res = D.run_evaluation(r.rel, MODEL, "blind", enable_thinking=False, n_boot=5)
    assert res["fit"]["tau2_status"].startswith("infeasible")
    code, out, err = cli(r)
    assert code == 0, err
    assert out.split() == ["eval_l2.json", "eval_l2.md"]


# ---------------------------------------------------------------------------- refusals
def assert_refused(r: Rel, match: str | None = None) -> None:
    with pytest.raises((EvalError, ValueError), match=match):
        D.run_evaluation(r.rel, MODEL, "blind", enable_thinking=False, n_boot=5)
    code, out, err = cli(r)
    assert code == 1, err
    assert out == "" and err.startswith("layer2 evaluation")
    assert nothing_written(r)
    assert FAKE_KEY not in err


def test_cli_happy_path_writes_both_files(fresh: Rel) -> None:
    code, out, err = cli(fresh)
    assert code == 0, err
    assert out.split() == ["eval_l2.json", "eval_l2.md"]
    assert (fresh.rel / "layer2" / "eval_l2.json").exists()
    assert not list((fresh.rel / "layer2").glob("*.tmp"))


def test_refuse_manifest_sha_does_not_match_predictions(fresh: Rel) -> None:
    df = pd.read_parquet(fresh.preds)
    df.iloc[::-1].to_parquet(fresh.preds)  # same content, different bytes
    assert_refused(fresh, "sha256")


def test_refuse_routed_object_without_terminal_record(fresh: Rel) -> None:
    fresh.edit("test", table=lambda t: t.slice(1))
    assert_refused(fresh, "no terminal record")


def test_refuse_missing_record_in_val(fresh: Rel) -> None:
    fresh.edit("val", table=lambda t: t.slice(0, t.num_rows - 1))
    assert_refused(fresh, "no terminal record")


def test_refuse_missing_parquet(fresh: Rel) -> None:
    fresh.out("test").unlink()
    assert_refused(fresh, "not found")


def test_refuse_extra_unrouted_row(fresh: Rel) -> None:
    extra = int(fresh.unrouted_test_ids()[0])

    def add(t: pa.Table) -> pa.Table:
        row = t.slice(0, 1)
        row = row.set_column(
            0, SCHEMA.field("source_id"), pa.array([extra], pa.int64())
        )
        return pa.concat_tables([t, row])

    fresh.edit("test", table=add)
    assert_refused(fresh, "exactly the")


def test_refuse_duplicated_row(fresh: Rel) -> None:
    fresh.edit("test", table=lambda t: pa.concat_tables([t, t.slice(0, 1)]))
    assert_refused(fresh, "exactly the")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("routing_manifest_sha256", "0" * 64),
        ("routing_manifest_sha256", None),
        ("model_requested", "nvidia/other-model"),
        ("arm", "with_l1"),
        ("fold", "val"),
        ("prompt_version", "some-other-prompt-version"),
    ],
)
@pytest.mark.parametrize("fold", ["val", "test"])
def test_refuse_divergent_parquet_metadata(
    fresh: Rel, fold: str, key: str, value: str | None
) -> None:
    if key == "fold" and fold == "val":
        value = "test"
    fresh.edit(fold, meta={key: value})
    assert_refused(fresh)


@pytest.mark.parametrize(
    ("key", "value"), [("template_sha256", "f" * 64), ("enable_thinking", "true")]
)
def test_refuse_val_and_test_with_different_template_or_thinking(
    fresh: Rel, key: str, value: str
) -> None:
    fresh.edit("test", meta={key: value})
    assert_refused(fresh, key)


def test_refuse_manifest_regenerated_after_consolidation(fresh: Rel) -> None:
    payload = json.loads(fresh.manifest.read_text(encoding="utf-8"))
    payload["a_keep2"] = 0.5  # any byte change: the parquets recorded the old sha
    fresh.manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert_refused(fresh, "different routing manifest")


def test_refuse_predictions_that_do_not_cover_the_test_fold(fresh: Rel) -> None:
    df = pd.read_parquet(fresh.preds)
    df[df["source_id"] != fresh.unrouted_test_ids()[0]].to_parquet(fresh.preds)
    fresh.write_manifest()  # sha now matches: the coverage check must catch it
    assert_refused(fresh)


# -------------------------------------------------------------------- misc script contract
def test_cli_unknown_arm_exits_2_before_reading_anything(fresh: Rel) -> None:
    with pytest.raises(SystemExit) as exc, redirect_stderr(io.StringIO()):
        load_cli().main(
            [
                "--release",
                "rel",
                "--model",
                MODEL,
                "--arm",
                "bogus",
                "--base",
                str(fresh.base),
            ]
        )
    assert exc.value.code == 2
    assert nothing_written(fresh)
