# Backlog (MINOR findings and deferred items)

Updated 2026-10-25 by the orchestrator; last update 2026-11-03 (T-OPS-STATE2). Items here never trigger a fixer cycle (CONTRACT §3).
"Before T15" means: fix before the first real download, because it affects a long, flaky run.

## DONE 2026-11-03
- H-D9 / G<20 blocker: RESOLVED as a documentation error, no rebuild (T-DOC-GCUT, T-DOC-GCUT-FIX1, reviewer re-check
  T-DOC-GCUT-RECHECK APPROVE_WITH_COMMENTS). Population = no G cut; null-G rows stay (nullable).
- GCUT-R1: fixed (T-DOC-GCUT-FIX1).
- First commit: 10 conventional commits b51ec00..00ae63e, no co-author trailer; pre-commit security CLEAR_TO_COMMIT (106 files).
- README.md and CLAUDE.md updated (provisional-results warning).
- GCUT-RC5 (CLAUDE.md "Status" part): DONE. The coder updated README/CLAUDE.md; reviewer approved. The ADR-text part of RC5
  is handled in the ADR 0006 iteration 2 rewrite / ADR 0004 annotation (see the GCUT section below).
- F2 (stale "100k sample" text of `data/interim/full_v0/eval/results.md` and `results.json`): DONE (T-FIX-RESULTS-TEXT, coder).
  Header, Limitations and line 1653 corrected; line 20 / json line 37 relabelled as measured on the 100k pilot batch; numbers
  untouched; backups `results.md.bak-20261103` and `results.json.bak-20261103` exist. Reviewer has NOT yet re-checked it.
  Remaining MINORs are listed under "Evaluation runner / dataset docs" below.
- Graphify graph rebuilt 2026-11-03: 1964 nodes, 4673 edges, 73 communities, 0 dangling edges. The old 522-dangling-edges item
  is RESOLVED.

## RESOLVED 2026-10-25 (H1 + T13c, verified: 129 passed, ruff clean, 0 xfail)
TAP-1, TAP-2, TAP-3, RV01-3 (code), RV01-5(a)(b), `_sanitize` widened (Bearer with space, quoted keys,
`key=`, `x-goog-api-key`, query text). Still open from those: tests for `pixel_l5`/`assign_folds`
validation (go to T20c); `ruff format --check` flags `metrics.py` and `reweight.py`; `from exc` chains
keep raw server text.

## Ingest client (`src/astro_triage/ingest/tap.py`), from T13b (items below are RESOLVED unless noted)
- **TAP-3 (before T15)**: text markers ("adql", "parsing", ...) are checked before the HTTP status
  (tap.py:163). A real 503 whose body mentions ADQL is classified permanent and not retried. The ESA
  server returns many 503s, so this matters for a long download. Fix: check status first, use markers
  only for 4xx and for 5xx that carry a clear syntax-error text; then flip the pinned test.
- **TAP-1**: cache hit returns NaN for missing strings, a fresh fetch returns None (tap.py:253).
  Two `xfail(strict=True)` tests pin it; remove the markers when fixed.
- **TAP-2**: `_sanitize` redacts secrets but not the query, so a server error that echoes the ADQL
  reaches `logs/ingest.log` (tap.py:141-143).
- Anonymous downloads above 50,000 rows need `login_interactive()`; the client does not enforce it.

## Review (RV-01, 2026-10-25): APPROVE_WITH_COMMENTS, no BLOCKER or MAJOR
The reviewer rebuilt the split from ADR 0003 rule 2 independently and it matches the committed CSV.
- RV01-1 `metrics.py:173`: `low-count` also fires in BALANCED (the function cannot tell regimes apart).
  Pass the regime or a `flag_low_count` parameter; test the BALANCED case. (before T21 runner)
- RV01-2 `metrics.py:311`: `ece` reports a value for a class with zero total weight; `pr_auc` says
  `absent_class`. Use the same absence rule. (before T21 runner)
- RV01-3 `split.py:32-34,73`: negative `source_id` wraps silently (a -1 sentinel gets a fold) and
  `pd.Series` raises `TypeError`. `np.asarray` the input and validate 0..12287.
- RV01-4 `tests/test_split.py:66-70`: S2 is a circular oracle. Add a test that rebuilds folds independently
  from the ADR rule (fresh `default_rng` per band, permutation of sorted pixels, integer cuts).
- RV01-5 `tap.py:161,169`: auth markers are checked before the status (same class as TAP-3); any
  `OSError` is treated as transient. Widen the TAP-3 fix to `_AUTH_MARKERS`; restrict transient to
  network errors. (before T15)
- Design note for T15: retrying `launch_job_async` resubmits a server job; for a logged-in user duplicates
  may count against the quota. Check before the first authenticated download.

## Security (SEC-01, 2026-10-25): no BLOCKER or MAJOR
- RESOLVED: SEC-01-1 personal paths in ADR 0001 (replaced by `$env:TEMP` and repo-relative paths);
  SEC-01-2 `.gitignore` gaps (added csv with `!manifests/*.csv`, fits.*, vot, tmp, log, .aws/, credentials*,
  pem, key; verified with `git check-ignore`); `boto3` removed from requirements (Phase 2 only).
- OPEN (SEC-01-3, low): no hash-pinned lock for transitive dependencies. Generate a lock with hashes
  when the dependency set stabilises; re-add `boto3` when Phase 2 starts.
- OPEN (before Phase 2): `_sanitize` in tap.py does not redact `Authorization: Bearer <token>` (only the
  word), quoted keys, `key=` or `x-goog-api-key`; `tests/test_tap.py:438` uses `Bearer1234` and misses it.
  Must be fixed before Gemini/Bedrock calls. Related: `from exc` chains keep raw server text; review when
  Streamlit shows tracebacks.
- OPEN (Phase 2, not built yet): strict schema validation of LLM/VLM output (never into ADQL, SQL, paths or
  HTML); BudgetTracker with pre-call check, on-disk persistence, fail-closed and a provider-side US$ 20
  cap; Bedrock via SSO only, Gemini key via env only, Ollama on localhost only.
- Note: the Parquet cache has no integrity check (SHA256 identifies the query, not the content). Local
  filesystem is trusted, so this is an integrity risk, not a security one.

## Metrics harness (`src/eval/metrics.py`), from T20
- Prior shift of posteriors and ECE on shifted outputs (ADR 0003 Metrics row): not implemented.
- Macro-F1 inside the bootstrap: not implemented.
- Label case: ADR grid is lowercase (`star`), metrics use `STAR`/`GALAXY`/`QSO`; the Phase 1 runner maps it.
- BALANCED regime with an absent class raises `EmptyClassError`; no renormalisation policy in the ADR.
- No mutation check was run: "tests fail on broken code" rests on hand-computed values only.
- ECE binning and `low-count` rule: being aligned with the ADR in T20b.

## Split (`src/astro_triage/split.py`), from T14
- `assign_folds` rebuilds the pixel map on every call (astropy transform of 12,288 points). Cache it
  or take the map as a parameter before using it on batches.
- Schema 5.1/5.2 asks for `data/splits/split_map_l5_v1.parquet` with metadata keys; only the CSV exists.

## Schema (`docs/specs/schema.md`) open questions still open
- OQ1 raw population: G<20 part RESOLVED 2026-11-03 (no G cut; documentation error, no rebuild). Remaining wording gaps
  are GCUT-RC2/RC3 below.
- OQ6 drop-reason precedence; OQ7 SpecObj/DR14Q column names and types; OQ8 TBV Gaia columns;
  OQ9 5-class combmod; OQ10 `r_match` (E1), `z_star_max`/`z_qso_min` (E3, see GCUT-RC3), DR15 galaxy rule (E4).
- `labels_v1` cannot be built until E1, E3, E4 run.

## Dataset builder (T20 / T20b, 2026-10-26), open items
- `n_specobj_candidates` is counted before the multi-Gaia drop; ADR 0002 section 3 does not say whether SDSS objects
  dropped earlier should still count. Decide and assert in a test.
- RESOLVED 2026-11-03 (owner: no G cut, nullable; documentation error, no rebuild): schema section 4 non-null G vs NULL-G pairs.
  The stale build.py comment is tracked as GCUT-RC4 below.
- `labels.parquet` lacks `label_status`, `in_dr14q`, `galaxy_public_by_dr15`, `in_cut_b`, `pm_propagated`; DR14Q/DR15 cut (b),
  epoch propagation (ADR 0004 D3) and the z-consistency filter (E3, GCUT-RC3) are not implemented yet.
- Folds go through `split.assign_folds`, which rebuilds the pixel map per call (already in the Split section); the test
  proves it matches the committed CSV.
- 100k sample: 0 test pixels reach 100 objects (max 34, median 11): the ADR 0003 reporting floor is only reachable with the
  full ingestion (user postponed it on 2026-10-26).
- Orchestrator error: the example position ra=266.4051, dec=-28.9362 is the Galactic Centre (l~0), not Sgr A*.

## LightGBM training (T21 / T21b, 2026-10-26), open items
- F-T21b-1 `models/lgbm.py:497-498`: `relative_to(REPO_ROOT)` raises ValueError for an output dir outside the repo;
  the CLI would exit 1 with a generic message. Raise a clear TrainError or store the path as given.
- Schema G1 says `schema_version` in metadata equals the version in the file name; prediction files carry
  `schema_version='1'` with a `v0` name (model_version). Ambiguous, needs an architect decision.
- Training weights: ADR 0003 prescribes none; trained unweighted, `pi_src` = train class fractions
  (STAR 0.3830, GALAXY 0.3771, QSO 0.2399). Sensitivity analysis with balanced weights is a candidate later.
- `models/lgbm.py` is about 545 lines (above the ~400-line guideline); split load/fit/calibrate/evaluate/write later.
- `scipy` is imported but not in requirements.txt (transitive via scikit-learn); pin it explicitly.
- Mutation copies left in the shared TEMP by the testers (`mut_*`); the sandbox blocks deleting them. The user can delete them.

## Kaggle pipeline (T23, 2026-10-26), open items
- Nothing in `kaggle/run_kaggle.py` or `scripts/xmatch_full.py` has run for real (only py_compile and a synthetic smoke test);
  first unknowns: pip pins on the Kaggle Python, SAS download speed, XMatch fair-use at 1M-row uploads.
- `models/lgbm.py` `_manifest` hardcodes release `dataset_100k_v0`; full_v0 manifests would carry the wrong label (pass `outdir.name`).
- `lgbm.py` N_THREADS=12 oversubscribes the 4 vCPU Kaggle machine; parametrise.
- Tests for `scripts/xmatch_full.py`, `scripts/make_kaggle_package.py` and the multi-file `build.py` path are missing (tester, T23b).
- Repackage the zip after T22 finishes (it bundles `evaluation/*` and `evaluate.py` as they were).
- Runner calls `evaluate.py --outdir` only; keep in sync with T22.

## Evaluation runner (T22 / T22b, 2026-10-26), open items
- F2 MINOR `scripts/evaluate.py:38`: `relative_to(REPO_ROOT)` raises ValueError for an outdir outside the repo (after writing).
- F1 MAJOR (empty stratum crash) sent to the fixer as T22c.
- `allosmod` has no `pi_src` in ADR 0003, so `prior_matched` equals `as_shipped` for it; needs an architect decision.
- ADR 0004 trigger 8 (macro-F1 loss > 0.10 concentrated in Solved=3) not computed; DSC per-stratum metrics are point estimates only.
- `evaluation/runner.py` is ~1,300 lines; split into modules later.
- Mutation copies in `%TEMP%\mut_T22b\{a,b}` (user may delete).
- Suite is slower (~75 s) because of the evaluate fixtures.

## T17b findings (tester, 2026-10-26), both MINOR, fixed in H2
- F-XM-1 `xmatch.py:192`: `_sanitize` does not remove absolute local paths from exception text.
- F-XM-2 `xmatch.py:176`: row count = newlines - 1 undercounts by one when the body has no trailing newline, so exactly
  2,000,000 data rows without a final newline would pass the truncation check.

## ADR 0004 (no-login data path, Proposed, 2026-10-25)
- Gates G-A / G-B must pass before leaving Proposed (see ADR 0004 section 7). Upload+JOIN on VizieR TAP is
  unverified (two sync attempts timed out); try `/async` first, then source_id range queries.
- No VizieR source found for `ra_error`, `dec_error`, `astrometric_excess_noise_sig`, `astrometric_params_solved`
  (blocker columns for G-B, never filled with placeholders). `l`, `b`, `parallax_over_error`, `pm` derived locally.
- DR13 cross-check (E2) deferred: raw columns 37 to 40 stay NULL.
- Combmod names `PQSO/PGal/PSS` in gaiadr3 are inference only; paramp reuses `PQSO`/`PGal`, so JOINs must alias by table.
- Update OQ8 above and the human-dependent list below to point at the ADR 0004 gates.

## Human-dependent (blocked)
- Gaia account: password page `cas.cosmos.esa.int/cas/passwd` returned HTTP 503 on 2026-10-25 while the
  rest of the ESA services answered 200. The user is to write to gaia-helpdesk@esa.int. Needed for T15
  and for the server-side crossmatch (E1).
- SDSS `specObj-dr17.fits` (6.7 GB) download: the user has not yet answered.
- Literal Gaia DR3 credit/citation text and whether Apsis/DSC must be cited: ESA pages not read.
- SDSS VLM-use question (helpdesk@sdss.org): documentation is silent; user decides whether to ask.

## T-L2-03 (Layer 2 closure, 2026-11-02), MINOR findings from the reviewer (A2)
- R-A2-003 ADR 0005 line ~1100: "no failure status was recorded" holds only for the final status per object in the four report rows; ledgers show 8 SERVER_ERROR (calls_20261027T135902Z) and 1 TIMEOUT (calls_20261027T183246Z), both retried. Limit the sentence.
- R-A2-004 ADR 0006 line 10 cites "ADR 0005 (pre-registered negative-result rule...)"; D7.8 was never executed. Suggested wording: "ADR 0005 (pre-registered negative-result rule D7.8, used as a template; never executed for Layer 2, ADR 0005 section 18; Kaggle/NVIDIA quota discipline)". Architect owns the edit.
- R-A2-005 ADR 0005 header (Date/Task/Supersedes lines) does not mention Amendment 2; add one sentence in section 18 that the CONTRACT section 8 text stands as the human's H8 decision although the ADR was never Accepted.
- SEC-B4 resolved: *.zip and NUL are in .gitignore lines 43-44 (security check, 2026-11-02).
- R-A2-006 (MINOR, re-check after T-L2-03-FIX1): ADR 0005 Amendment 2 should cite the 3 ledger file names it relies on.

## Layer 2 closed 2026-11-02 (T-L2-03 DONE 2026-11-03, branch A)
- Closed on descriptive val-pilot evidence; D7.8 test-fold rule not executed; test fold of full_v0 never exposed to Layer 2.
  Chain: A1 architect DONE; A2 reviewer 2 MAJOR -> fixer T-L2-03-FIX1 -> re-check APPROVE_WITH_COMMENTS; A3 security clear.
- SEC-B1..B4 frozen: no further NVIDIA calls; `scripts/layer2_run.py` is not run. Reopening needs a new ADR (ADR 0005 section 18).
- OPEN check: dashboard (`src/astro_triage/dashboard/*`) Layer 2 page behaviour when `eval_l2.json` is absent was never verified.

## Evaluation runner / dataset docs, from reviewer T-BLK-01 (2026-11-03)
- F3 `evaluation/runner.py:46,87,1267` hardcodes `r_match` 0.75 and the 100k match rates. Read `r_match_arcsec` from the
  labels parquet metadata instead. Also: `runner.make_header` would regenerate the shorter (pre-fix) header text on the next
  evaluate run; when fixing F3, add the "protocol not frozen" wording to the generated header if desired.
- F4 `src/astro_triage/dataset/build.py:1,69,104-105`: comments and the default name still describe the 100k batch. The
  "25 pairs without G" figure is not general (full_v0 has 617 null-G rows). Same fix as GCUT-RC4 below (one coder task).
- F2 results.md/results.json text: DONE 2026-11-03 (T-FIX-RESULTS-TEXT; see DONE section). Reviewer re-check pending.
  Remaining MINOR from that task: `results.md` line ~1650 ("match rate STAR 91% vs GALAXY 31%") should be labelled as measured
  on the 100k pilot batch.

## Pre-commit security audit (2026-11-03): CLEAR_TO_COMMIT, 106 files
- SEC-PC-01 the 3 `docs/*.docx` carry the owner's first name, course and institution in the text and in `docProps/core.xml`.
  Clear the metadata or move them out of git BEFORE the repo goes public.
- SEC-PC-02 `.gitignore` preventive patterns: being added by T-REPO-02.
- SEC-PC-03 decide whether `.claude/state/` stays versioned before the repo goes public.

## Gaia XP, from researcher T-RES-XP (2026-11-03), low priority (XP out of scope for now)
- Read the rest of Section 4 of arXiv:2206.06143: criteria for sources fainter than G=17.65 and origin of the ~100K quasar list
  (a local PDF copy exists in the tool-results folder).
- Re-measure the exact share of G<17.65 per class on full_v0. Reviewer quantiles agree with ADR 0006's 34.3% / 0.1% / 0.9%,
  but the exact share was not recomputed.

## ADR 0006 (deep-learning tabular comparison): section 4 rewrite (architect), iteration 2 under reviewer re-check
- Owner decisions 2026-11-02/03 to write into section 4:
  - Metric: purity of QSO and GALAXY at fixed completeness 0.80 (primary) and 0.90 (secondary), thresholds chosen on val only.
  - Prior P2 primary; P1 and P3 as sensitivity.
  - Gain = pointwise difference > delta AND 95% pixel-block bootstrap CI > 0, in both classes.
  - delta = 2 x SD of LightGBM across 5 seeds (by formula). Tie keeps LightGBM.
  - 5 seeds; same features, split, tuning budget and calibration; missing-value masks on both sides; DSC as context only.
- Owner decisions 2026-11-03 (added):
  - Redshift rule (GCUT-RC3): keep full_v0 as built; record as a limitation plus trigger T7 in ADR 0006.
  - H-D3 = B0 (decided).
  - Low-count rule = total reading (as implemented in metrics.py).
  - Ensemble pairs: descriptive only.
  - Population: no G cut.
- H-D9: resolved: documentation error, no rebuild, 2026-11-03. T6 does not fire.
- Remove the unconfirmed claim "XP focused on red sources" (line ~49).
- Still open (owner): H-D4 (confirm 5 seeds and that delta/CI replaces the 4-of-5 rule); H-D7 (separate requirements-dl.txt);
  H-D8 (re-approve the new constants: SD fold = val, level-3 block bootstrap, MIN_BLOCKS 20, n/a cell = tie, seed-mean reading
  of "pointwise difference"); H-D2; H-D6; H-D1.
- Reviewer T-DL-02-REVIEW findings M3, M4, M5: applied in iteration 2, pending re-check.

## Population docs re-check (T-DOC-GCUT-RECHECK, 2026-11-03): APPROVE_WITH_COMMENTS, MINOR only
- GCUT-RC1 `docs/adr/0001-storage.md:21-23`, `docs/specs/schema.md:123-125`: 17,611 is a count of PAIRS. Also state the
  SpecObj-row count: 8,779 SpecObj rows (STAR 5,494, GALAXY 3,157, QSO 128) dropped by the ADR 0002 section 3 ambiguity rule.
  Architect owns the ADR edit.
- GCUT-RC2 same places + schema OQ1 (~line 483): the population definition omits the second ADR 0002 section 3 rule (one Gaia
  source with several SpecObj: keep the nearest if classes agree, drop if they conflict). In full_v0
  source_class_conflict = 0 and same_class_duplicates_collapsed = 0, which is why 2,061,857 - 17,611 = 2,044,246 holds.
- GCUT-RC3 (more relevant) `docs/specs/schema.md:239` vs 119-127: the ADR 0002 section 2 redshift-consistency rule
  (`z_star_max` / `z_qso_min`) is NOT implemented anywhere in `src/` (E3/OQ10 open), yet schema.md:239 lists it as part of the
  label set; full_v0 labels did not pass that cut. Needs an explicit statement in the population text.
  OWNER DECIDED 2026-11-03: keep full_v0 as built (rule not implemented); limitation + trigger T7 in ADR 0006.
- GCUT-RC4 `src/astro_triage/dataset/build.py:104-105`: stale comment "25 pairs have no G photometry" (full_v0 has 617 null-G
  rows) and it cites schema section 4 non-null G. Coder task, code outside docs; merge with F4 above.
- GCUT-RC5 cleanup: text that still describes the G<20 anomaly as open, to handle in the ADR 0006 section 4 rewrite and the
  ADR 0004 annotation decision (architect): `docs/adr/0004-no-login-data-path.md:258`;
  `docs/adr/0006-deep-learning-tabular-comparison.md:55-58, 296, 309, 327`. The `CLAUDE.md` "Status" part is DONE 2026-11-03
  (coder updated README/CLAUDE.md, reviewer approved).
