# Spec: the Gráficos page (`/graficos/`) — ADR 0008, Amendment 1

Status: **Approved by the owner on 2026-11-25** (task G0 done; answers to Q1 to Q6 recorded in ADR
0008, A1.10). G6 (ECE panel) is not authorised yet and must not run. Contract (schema of `graficos.json`): ADR 0008, section A1.8.
Every task runs in its own worktree from current main; none links `data/`; real data is read only
through `ASTRO_DATA_ROOT` pointing at the main checkout, read only.

Verify command (Python): `.venv\Scripts\python.exe -m pytest -q` and `ruff check src tests`.
Verify command (web, from `web/`): `npm run lint`, `npm run typecheck`, `npm run test`,
`npm run build` with `ASTRO_WEB_DATA=test/fixtures ASTRO_RELEASE=fixture_v0`.

---

## G1 — Python aggregations (coder) — parallel with G4

```text
FILE: src/astro_triage/dashboard/data_charts.py        [create]
GOAL: Pure aggregations of Release.test for the Gráficos page (LightGBM, test fold).
EXPORTS:
  G_EDGES: Final[tuple[float, ...]] = (10.0, 11.0, ..., 22.0)      # 13 edges, 1 mag
  QSO_Z_EDGES: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
  CALIBRATION_BINS: Final = 10
  calibration_bins(rel: Release, model: str, n_bins: int = CALIBRATION_BINS) -> dict[str, pd.DataFrame]
      # per class: D.calibration_points columns + hits:int (= round(observed*n)), ci_lo, ci_hi
      # (wilson_interval(hits, n)); raw_mean_pred/raw_observed kept when present
  error_by(rel: Release, model: str, by: Literal["G", "z"]) -> dict[str, pd.DataFrame]
      # "G": {c: error_rate_table(rel, model, c, "G", G_EDGES) for c in CLASSES}
      # "z": {"GALAXY": error_rate_table_z(rel, model),
      #       "QSO": error_rate_table(rel, model, "QSO", "z", QSO_Z_EDGES)}
  confusion_cells(rel: Release, model: str) -> pd.DataFrame
      # 9 rows in CLASSES x CLASSES order: true, pred, n, row_n, share, ci_lo, ci_hi
      # from D.confusion; share = n/row_n (nan if row_n == 0); Wilson via wilson_interval
DEPENDS: math, typing, numpy, pandas; astro_triage.dashboard.data_release (CLASSES, Release);
         data_overview (calibration_points, confusion); data_galaxies (error_rate_table,
         wilson_interval); data_galaxies_z (error_rate_table_z)
INVARIANTS: no streamlit, no data_cached, no views_*, no dl_* import; reads only rel.test; never
            applies the floor (the exporter does); no I/O
ACCEPTANCE: tests/test_dashboard_data_charts.py green; calibration_bins mean_pred/observed equal
            D.calibration_points exactly; confusion_cells n sums to len(rel.test); error_by("G")
            rows equal error_rate_table output for each class; a test (skipped without
            ASTRO_DATA_ROOT) asserts G_EDGES == results["histograms"]["g_mag_edges"] of full_v0
RISK: round(observed*n) must give an int that reproduces observed; assert hits/n == observed
```

```text
FILE: tests/test_dashboard_data_charts.py               [create]
GOAL: Unit tests of data_charts with a small in-memory Release (pattern tests/test_dashboard_data.py::_release).
ACCEPTANCE: green; covers empty class, missing G ("sem medida" row), z <= Z_LOW band for GALAXY
```

## G2 — Exporter (coder, then tester) — after G1, parallel with G3

```text
FILE: src/astro_triage/web_export/graficos.py          [create]
GOAL: Build graficos.json (ADR 0008 A1.8) with ece = null and cmd = null.
EXPORTS: PAGE: Final = "graficos"; MODEL: Final = "lgbm_full"; graficos(rel: D.Release) -> dict
DEPENDS: pandas; astro_triage.dashboard.{data as D, data_charts, fmt}; html_dl.class_label;
         html_galaxies.SMALL_RATE; charts_base.MODEL_PT; data_release.CLASSES;
         astro_triage.web_export.{types (est, num, head), web_text}
INVARIANTS: floor = int(rel.meta["floor"]["min_objects"]); below floor -> errors and value fields
            null + below_floor true (1D charts keep n; confusion cells set n null too); rate digits
            = 2 when rate < SMALL_RATE else 1 (same rule as pages._bands_table, duplicated, not
            imported); lo/hi non-finite -> null; "sem medida" row -> series.missing; no reference
            to dl_eval, dl/ or predictions_mlp/ftt; no HTML; no "ADR"; computes nothing that
            data_charts does not return
ACCEPTANCE: parity tests below green; `python -c "import astro_triage.web_export.graficos, sys;
            assert 'streamlit' not in sys.modules"` exits 0
RISK: pages._bands_table drift; covered by a test that compares the GALAXY by-z rows of
      graficos.json with galaxies.json technical.by_z on the same release (same function, same edges)
```

```text
FILE: src/astro_triage/web_export/web_text.py          [modify]
GOAL: Add the data-dependent sentences of the page.
EXPORTS: charts_scope(n_test: int, model_label: str) -> str
           # names the model, "conjunto de teste", fmt.pt_int(n_test), class = highest calibrated
           # probability, and that rule and prior of the other pages do not apply here and that
           # these shares are not the purity/completeness of Visão geral
         charts_floor_note(floor: int) -> str
         missing_g_note(counts: Mapping[str, int]) -> str    # "" when all zero
         NETWORKS_ABSENT: Final[str]   # why MLP/FT-Transformer are not on the page (no test re-read)
         STAR_Z_NOTE: Final[str]       # why stars have no z chart (Q3 default)
INVARIANTS: no digit literal (tests/test_web_export_headline.py::test_fixed_texts_have_no_digit_literals
            stays green); numbers only via fmt.*; no "ADR"
ACCEPTANCE: digit-detector test green
```

```text
FILE: src/astro_triage/web_export/__main__.py          [modify]
GOAL: Register "graficos": lambda: graficos.graficos(rel) in _documents (line ~81-87).
ACCEPTANCE: export of the synthetic release writes 8 files
```

```text
FILE: tests/test_web_export_parity.py                  [modify]
GOAL: Cover graficos.json.
CHANGES: DOCS gains "graficos"; field parity of every calibration bin, rate row and confusion cell
         against data_charts; equality of graficos.calibration (n >= floor) with
         overview.technical.calibration; test_versioned_fixture_matches_a_fresh_export covers it
ACCEPTANCE: green; the 7 existing fixture files unchanged byte-for-byte except generated_at
```

```text
FILE: tests/test_web_export_invariants.py              [modify]
GOAL: FILES gains "graficos"; new test: below_floor items have null errors and value (and null n
      in confusion/cmd cells); no key in {source_id, ra, dec, l, b, healpix_l5, hp} in graficos.json;
      no array longer than 252; graficos.json < 150 KB on full_v0 (real-release test, skips
      without data); source of graficos.py and data_charts.py contains none of "dl_eval",
      "predictions_mlp", "predictions_ftt", "\"dl\"".
```

```text
FILE: web/test/fixtures/fixture_v0/graficos.json       [create, generated]
GOAL: Regenerated by the exporter on the synthetic release (never hand-edited).
```

## G3 — Web contract and route plumbing (coder) — parallel with G2

```text
FILE: web/lib/schema.ts                                [modify]
GOAL: Add RateRow, CalBin, Graficos (A1.8, with ece and cmd typed as their object | null) and
      PageData.graficos.
ACCEPTANCE: typecheck green; G5 loads the fixture without casts
```

```text
FILE: web/content/labels.ts                            [modify]
GOAL: routes gains { href: "/graficos/", label: "Gráficos" } before "/metodo/"; export
      chartsHref = "/graficos/"; pages.charts {eyebrow, title, lead}; graficosLabels (chart titles,
      axis names, table captions, "Ver os números", legend words). No digits.
ACCEPTANCE: static.test "content/labels.ts nao tem algarismos" green
```

```text
FILE: web/components/shell/StateControls.tsx           [modify]
GOAL: hidesRuleAndPrior also true on chartsHref; the bar and the tech toggle stay on /graficos/.
ACCEPTANCE: render test: on /graficos/ the rule and prior inputs are absent and the tech checkbox
            is present; compare route behaviour unchanged (phase2b tests green)
```

```text
FILE: web/test/data.test.ts                            [modify]
GOAL: PAGES gains "graficos".
```

## G4 — Chart components (coder) — parallel with G1/G2

```text
FILE: web/components/charts/RateByBin.tsx              [create]
EXPORTS: RateByBin({ id, title, series: { key, label, rows: RateRow[] }[], showCi: boolean,
                     yLabel, naText }): JSX.Element
INVARIANTS: x is ordinal (row label text from JSON); series told apart by marker shape and direct
            label, never hue; below_floor rows draw a hollow marker without a value and use NA with
            rate.text; whiskers only when showCi and lo/hi non-null; colours only via CSS vars
ACCEPTANCE: render test: one marker per reportable row; below-floor row has no filled marker and
            shows rate.text; no whisker when showCi false
```

```text
FILE: web/components/charts/ConfusionGrid.tsx          [create]
EXPORTS: ConfusionGrid({ caption, classes, rows, showCi, trueLabel, predLabel })
INVARIANTS: a real <table> with <th scope>; every cell prints share.text (and n.text); shading by
            grey opacity of share.point only as a redundant cue; diagonal marked by weight/border;
            below-floor cell renders NA
ACCEPTANCE: render test: 9 cells; below-floor cell shows "n/a" text from JSON and no shading
```

```text
FILE: web/components/charts/ReliabilityPlot.tsx        [modify]
GOAL: Optional props only: intervals?: boolean (draw freq.lo..hi), raw?: boolean (hollow markers
      at raw_p_mean/raw_freq), accepting points with null values (skipped).
INVARIANTS: default rendering byte-identical for current callers (OverviewTechnical)
ACCEPTANCE: existing overview tests green; new test for intervals and null points
```

```text
FILE: web/components/charts/graficos.module.css        [create]
GOAL: Styles of the new components; CSS vars only (static.test colour sweep).
```

```text
FILE: web/test/graficos-charts.test.tsx                [create]
GOAL: Render tests of G4.
```

## G5 — Page (coder, then tester) — after G2, G3, G4

```text
FILE: web/app/graficos/page.tsx                        [create]
GOAL: Server component: loadPage("meta"), loadPage("graficos"); PageHeader(pages.charts,
      showStateChip false); <GraficosView />.
```

```text
FILE: web/components/pages/GraficosView.tsx            [create]
GOAL: Client view (pattern GalaxiesView: readState(useSearchParams(), meta)). Order: scope_text and
      floor_text; (1) ReliabilityPlot LightGBM 3 classes + networks_note; (2) RateByBin by G, 3
      classes + missing_note; (3) RateByBin by z, GALAXY and QSO + star_note; (4) ConfusionGrid.
      Under each chart a Details "Ver os números" with a DataTable (both modes). Technical mode
      adds: CI whiskers, raw pre-temperature curve, n and IC columns in the tables; ece panel when
      graficos.ece != null (G6); cmd count layer (G7). cmd and ece null -> section absent.
INVARIANTS: no number formatted in TS except fmt.ts axis ticks; no digit literal; NA for nulls
ACCEPTANCE: web/test/graficos.test.tsx: 4 figures in simple mode, each with a table alternative;
            tecnico=1 adds whiskers; every digit in the DOM equals a JSON text or fmt.ts tick
```

```text
FILE: web/components/pages/graficos.module.css         [create]
```

```text
FILE: web/test/static.test.ts                          [modify]
GOAL: PAGES gains ["graficos/index.html", "/graficos/", "graficos"]; test name "6 links".
ACCEPTANCE: after build, each of the 6 pages links the 6 routes and shows the seal
```

## G6 — ECE panel (only if Q1 = option a) — after G5

Files: `web_export/graficos.py` (`_ece(rel)`: `data_dl.load_results_dl(rel)`; families
`lgbm_full`, `mlp_full`, `ftt_full`; `ece/<cls>` mean/min/max; label `html_dl.system_label`;
status "unavailable" when absent), `web_text.py` (`ECE_NOTE`: regime balanced prior-matched,
15 equal-mass bins, min to max of 5 seeds, not a CI), `tests/test_web_export_parity.py`
(`_results_dl()` gains `descriptive.families`; prove compare.json fixture unchanged), fixture
`graficos.json`, `GraficosView.tsx` (reuse `PurityRange` with median = mean, detail = detail_text,
technical mode only). Acceptance: parity of every ECE value with results_dl.json; no file under
`dl/` other than `eval/results_dl.json` opened (test with a patched `open`/`read_parquet`).

## G7 — Colour-magnitude map (coder, tester, reviewer, **security**) — last

```text
FILE: src/astro_triage/dashboard/data_load.py          [modify]
GOAL: features columns (line 83) gain "bp_rp". Nothing else.
ACCEPTANCE: Streamlit tests green; rel.test has bp_rp
```

```text
FILE: src/astro_triage/dashboard/data_charts.py        [modify]
EXPORTS: CMD_G: Final = (12.0, 21.0, 0.5); CMD_C: Final = (-0.5, 3.0, 0.25)
         cmd_edges() -> tuple[np.ndarray, np.ndarray]     # np.linspace, 19 and 15 edges
         cmd_cells(rel, model) -> CmdGrid                 # frozen dataclass: cells DataFrame
           (gi, ci, n, errors, rate, ci_lo, ci_hi) for all 252 cells, half-open [lo, hi);
           n_missing (G or bp_rp NaN), n_outside
INVARIANTS: edges never derived from data; no smoothing
```

Also: `graficos.py` `_cmd` (all 252 cells; below floor n/errors/rate null; ticks text via
`fmt.pt_num`; cap = max reportable rate and n); `web_text.cmd_note(n_missing, n_outside, floor)`;
synthetic features gain a deterministic `bp_rp` (no RNG draw, e.g. from `sid`); fixture regenerated;
`web/components/charts/CmdMap.tsx` (SVG rects, grey opacity with `MIN_OPACITY` from
`components/sky/scale.ts`, below-floor cells hatched with a pattern and no value, `<title>` per
cell with JSON text, table alternative of reportable cells; mode "rate" | "count", count only in
technical mode). Security review scope: A1.2 item 7 of ADR 0008.

## G8 — Cross verification (tester, reviewer)

6 routes, seal, state kept, digit trace, n/a sweep, "ADR" sweep, 1440 and 390 px screenshots for
the owner.
