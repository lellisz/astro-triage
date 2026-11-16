# ADR 0002: Label source (SDSS) and DSC circularity

> Amended by ADR 0004 regarding the crossmatch engine and column names.
>
> Amended 2026-10-26 (task A-04b, facts from `docs/research/R-09-no-login-path.md` "Update" and "XMatch
> smoke test", and user decisions of 2026-10-25/26):
> - §1 and §2: `specObj-dr17.fits` has no `RA`, `DEC` or `SCIENCEPRIMARY`. Positions are `PLUG_RA`/
>   `PLUG_DEC` (fibre/plug positions), and the primary flag is `SPECPRIMARY == 1`, checked explicitly.
> - §4 cut (c), and every DSC metric: DSC coverage rule (DSC output exists for only part of the matched
>   rows).
> - Limitation 1 and E1: measured class-dependent match rate, and plug-position scatter.
> Amended text is marked "(A-04b)". The original text is kept where it is not contradicted.

- Status: Accepted (2026-10-25, human approval). Experiments E1 to E6 stay open and must be run
  before `labels_v1` is built.
- Date: 2026-10-25
- Task: T11
- Approved upstream: D1 (own crossmatch against a recent SDSS release; DR13 official crossmatch only as a check), D2 (DSC reported in three cuts), D8 (LightGBM does not use BP/RP spectra).
- Out of scope: train/test split and class priors (ADR 0003), ingest mechanics, quotas, layer 2 (VLM).

## Context

- Labels STAR/GALAXY/QSO exist only in SDSS SpecObj. The only official Gaia x SDSS crossmatch is `gaiadr3.sdssdr13_best_neighbour` (+ `sdssdr13_neighbourhood`, `sdssdr13_join`), against DR13, with no class column. [1]
- `specObj-dr17.fits` (6.7 GB) is confirmed on the SAS. (Correction 2026-10-26: it is NOT primary-only. Applying `SPECPRIMARY == 1` dropped 688,476 of 5,801,200 rows, and `ZWARNING == 0` dropped 829,217 more, in that order.) Latest public release is DR20 (2026-07-30); DR17 is 2021-12-06. DR19/DR20 file names and schema are not confirmed. [3][4]
- (A-04b) The file is downloaded and verified: 6,741,023,040 bytes, 5,801,200 rows, 133 columns. Present: `PLATEID, MJD, FIBERID, SPECOBJID, CLASS, SUBCLASS, Z, Z_ERR, ZWARNING, PLUG_RA, PLUG_DEC, SPECPRIMARY, PRIMTARGET, SURVEY, PROGRAMNAME`. Absent: `RA`, `DEC`, `SCIENCEPRIMARY`. The file carries a `SPECPRIMARY` flag, so "primary spectra only" is not assumed; it is filtered. [5]
- Gaia DSC was trained on ~300k SDSS-DR14Q quasars (Paris+2018, ZWARNING=0), ~50k SDSS-DR15 galaxies (Aguado+2019), ~720k random Gaia sources as stars (not SDSS-labelled), ~40k white dwarfs (MWDD). No training-membership flag was found in the parts of the archive read. Source confidence: medium-low. [2]
- `gaia_source` carries `classprob_dsc_combmod_{quasar,galaxy,star}`; `astrophysical_parameters` adds `specmod` (BP/RP only) and `allosmod` (photometry + astrometry only). [1]
- (A-04b) Measured on 467 matched rows: only 262 (56%) have non-null allosmod probabilities. The DSC was not run for every source. [5]
- Consequence: scoring DSC on SDSS objects partly scores it on its own training data (circularity), and the sample is doubly selected (SDSS chooses what to observe; Gaia detects only compact/bright galaxies).

## Options

| # | Option | Real cost |
|---|---|---|
| L1 | Use official DR13 `best_neighbour` + join DR13 SpecObj for class | Cheapest (server-side, no own matching). Locks labels to DR13; ignores DR14-DR17 spectra (eBOSS QSOs, etc.). Pipeline class/zWarning of DR13 is older. |
| L2 | Own positional crossmatch Gaia DR3 x SpecObj DR17 | Must download/upload 6.7 GB file subset and run a cross-match; must choose radius and ambiguity policy ourselves. Schema and file confirmed. |
| L3 | Own crossmatch Gaia DR3 x SpecObj DR20 | Most recent labels. DR20 file name, schema and SpecObj semantics unverified; zero digest evidence. Risk of building on an unconfirmed table. |

| # | Option (circularity) | Real cost |
|---|---|---|
| C1 | Report DSC on all SDSS objects only | Trivial; result biased in DSC's favour, cannot be told apart from true skill. |
| C2 | Three cuts (a/b/c), D2 | More bookkeeping (training-set reconstruction, two DSC variants). Cut (b) rests on a medium-low-confidence reconstruction. |

## Decision

### 1. Release and crossmatch method (L2)
- Labels come from **SpecObj DR17** (`specObj-dr17.fits`). Reason: only release whose label file is confirmed; DR20 adds evidence cost with no confirmed gain. Reversible in < 1 day once a DR20 SpecObj with the same columns is confirmed (see reversal triggers).
- DR13 `sdssdr13_best_neighbour` is used **only** as a cross-check (agreement rate on the overlap), never as a label source.
- Method: positional match of SpecObj (ra, dec) to Gaia DR3 sources, with Gaia positions propagated to the SDSS epoch using pmra/pmdec where available (Gaia epoch 2016.0). Execution engine is **deferred to a future task**, to be chosen among: (i) upload a SpecObj subset as a Gaia Archive user table and join with ADQL `DISTANCE`/`CONTAINS` or `ESDC_CROSSMATCH_POSITIONAL` (documented [1-round2]); (ii) CDS XMatch (limits unconfirmed); (iii) local match with astropy on downloaded data. Upload limits are unconfirmed for (i) and (ii).
- (A-04b) The SpecObj position is **`PLUG_RA`, `PLUG_DEC`** (the file has no `RA`/`DEC`). These are fibre/plug positions, not a final astrometric position. The engine is CDS XMatch (ADR 0004 D3).
- Match radius `r_match` is a parameter fixed by experiment E1 (Open questions), not by this ADR.

### 2. Label quality filter
Keep a SpecObj row only if all hold:
- `zWarning == 0`;
- `class` in {STAR, GALAXY, QSO};
- primary spectrum (already true for `specObj-dr17.fits` [4]); *(A-04b: checked explicitly as `SPECPRIMARY == 1`; `SCIENCEPRIMARY` does not exist in the file)*
- (A-04b) finite `PLUG_RA` and `PLUG_DEC`;
- STAR/QSO consistency by redshift: STAR requires |z| below `z_star_max`; QSO requires z above `z_qso_min`. Thresholds set by E3; rows failing are dropped, counted, and logged.
- `subclass` is **kept as metadata, not a filter** (e.g. to slice white dwarfs, AGN-flagged galaxies, broad-line QSOs in error analysis).

(A-04b) Before the z rule, the file has **4,283,507** rows that pass the other conditions: STAR 890,744, GALAXY 2,639,535, QSO 753,228.

### 3. Duplicates and multiple neighbours
- Unit of the dataset = Gaia `source_id` (one row per Gaia source).
- One SpecObj with > 1 Gaia source within `r_match`: **drop** (ambiguous); count recorded per class.
- One Gaia source matched by > 1 SpecObj rows: keep if all classes agree (retain the nearest); **drop** if classes disagree; both counts recorded.
- Gaia sources with no SpecObj match are unlabelled and outside every cut below.

### 4. The three cuts (D2)
Let `S` = labelled set after sections 2-3 restricted to the test split defined by ADR 0003.

- **Cut (a) — full SDSS, optimistic.** Population: all of `S`. Report DSC `combmod` (3 classes from `gaia_source`) and `allosmod`. Every table/plot carrying this number is labelled "DSC: optimistic (training overlap)".
- **Cut (b) — outside DSC training.** Population: `S` minus
  1. any object matched (same `r_match` policy) to the SDSS-DR14Q catalogue (all rows, not only ZWARNING=0, to be conservative);
  2. any GALAXY whose spectrum belongs to the Aguado+2019 DR15 galaxy selection; until that selection is reproducible, exclude **every GALAXY spectrum public by DR15** (plate/mjd-based approximation — hypothesis, E4);
  3. STARs: Gaia random-star training members cannot be identified; no removal, residual overlap declared.
  Cut (b) is reported with the per-class counts removed, and the class mix shown next to cut (a).
- **Cut (c) — same information.** Population: `S` (and repeated on cut (b)). Compare LightGBM vs DSC `allosmod` only. LightGBM features: `gaia_source` photometry and astrometry only; **no BP/RP spectra (D8), no `classprob_dsc_*` column as a feature** (would import DSC's training labels).
  - (A-04b) **DSC coverage rule.** Population for the DSC `allosmod` baseline: the rows of `S` (or of `S ∩ (b)`) where all three allosmod probabilities are non-null, called `S_dsc`. LightGBM is reported twice: on `S_dsc` (the fair comparison, same rows as DSC) and on the full `S`. The coverage `|S_dsc| / |S|` is reported per class, next to the metrics.
- (A-04b) **The coverage rule applies to every DSC number** in cuts (a), (b) and (c), for `combmod` and `allosmod` separately, each on the rows where that variant has non-null output. A missing DSC output is **never** mapped to STAR (or any class) and is **never** counted as an implicit abstention ("don't know"). The row is outside that DSC's evaluation set, and it is counted in the coverage.

## Consequences
- Good: labels use 4 more data releases of spectra than L1; circularity is measured, not ignored; cut (c) isolates model skill from input information.
- Bad: we own the crossmatch quality (radius, proper motion, blends). Cut (b) shrinks QSO and GALAXY counts and shifts class mix, so (a) vs (b) differences mix circularity with population change. LightGBM is trained on the same SDSS-selected population, so cut (c) does not remove selection bias; it only equalises inputs.
- (A-04b) Bad: DSC numbers describe only the DSC-covered subset (56% of matches in the smoke sample), which may differ in class mix and brightness from the full set. The LightGBM-on-`S_dsc` versus LightGBM-on-`S` difference shows the size of that effect.
- Maintenance cost: one crossmatch job + one DR14Q match + one DR15-membership rule to keep reproducible; re-run on any release change. (A-04b) Plus one per-class DSC coverage table per report.

## Limitations
1. **Double selection bias.** SDSS targets what it chose to observe; Gaia detects only compact/bright galaxies. Metrics in all three cuts describe that intersection, not the Gaia sky. Not corrected here (priors: ADR 0003).
   (A-04b) Measured on 1,000 random eligible SpecObj rows (5", best match): the Gaia match rate is STAR 84.2%, QSO 59.9%, GALAXY 31.1%, with a sampling error of a few points. The class mix of the matched sample differs strongly from the SDSS mix. This is the central selection effect of the label set; it is reported per class from the full run, and it is why the realistic-prior regime of ADR 0003 reweights.
2. **Residual circularity.** Training stars (~720k random Gaia sources) and MWDD white dwarfs cannot be removed from cut (b); galaxy removal is approximate.
3. **Low-confidence reconstruction.** Training-set facts come from a search summary of ESA docs (medium-low), ESA pages returned 503, and the absence of a training flag is "not found", not "confirmed absent". Spectrum first-release date from plate/mjd is unverified.

## Open questions (each with the experiment that resolves it)
- E1: `r_match`. Histogram of separations (epoch-propagated) for SpecObj DR17 x Gaia DR3; choose radius at the knee; compare with DR13 `best_neighbour` choices.
  (A-04b) E1 must account for the scatter of the plug positions (`PLUG_RA`/`PLUG_DEC` are not final astrometry). Baseline from the smoke test (467 matches, 5", unpropagated): median 0.08", p95 0.44", max 4.62". A 5" radius likely admits false matches. E1 runs on the batch of ADR 0004 gate G-A1.
- E2: DR13 check. On objects in both DR13 `best_neighbour` and our match: fraction with the same `source_id`. Disagreements inspected by hand on a sample.
- E3: `z_star_max`, `z_qso_min`. Distribution of z per class at `zWarning == 0`; set cuts where classes separate.
- E4: First-release date. Verify whether plate/mjd ranges map to DR15 publication (SDSS docs or DR15 SpecObj diff vs DR17).
- E5: Training flag. Re-read Delchambre+2023 [2] and ESA DSC docs when gea.esac.esa.int is up.
- E6: DR20 SpecObj. Confirm file name, schema, and whether `class`/`zWarning` semantics match DR17.

## What would change my mind
1. A DR20 SpecObj file is confirmed with the same `class`/`zWarning`/primary semantics → switch L2 to L3 (reversal trigger).
2. ESA publishes a DSC training-membership flag or list → replace the reconstruction in cut (b) with it.
3. E2 shows our match disagrees with DR13 `best_neighbour` on a non-trivial share of the overlap → stop and revisit radius/epoch policy before any metric is reported.
4. E4 shows plate/mjd cannot date spectra → drop rule (b).2 approximation and report cut (b) for QSO only, stating galaxies as unresolvable.
5. Cut (b) leaves too few GALAXY rows for a stable metric (threshold set in ADR 0003) → report (b) per class only where counts allow.
6. (A-04b) `S_dsc` falls below the ADR 0003 reporting floor for a class → that class's DSC cell is `n/a`; LightGBM on the full `S` is still reported.

## Sources
[1] Gaia DR3 metadata (AIP mirror): https://gaia.aip.de/metadata/gaiadr3/ ; Apsis I: https://arxiv.org/abs/2206.05864
[2] DSC, Apsis III (Delchambre+2023): https://arxiv.org/abs/2206.06710
[3] SDSS citing / DR17 (Abdurro'uf+2022, ApJS 259, 35): https://www.sdss4.org/collaboration/citing-sdss/
[4] `specObj-dr17.fits` on SDSS SAS (6.7 GB), per research round 2 (2026-10-25). The "primary spectra only" claim in that round was wrong; see the correction in section 1.
[5] `docs/research/R-09-no-login-path.md`, sections "Update" (2026-10-25) and "XMatch smoke test" (2026-10-26).
[1-round2] Gaia Archive ADQL crossmatch functions (ESDC_CROSSMATCH_POSITIONAL, DISTANCE/CONTAINS), per research round 2.

## Note 2026-11-16 (state of experiments E1 to E6)

Status unchanged: Accepted. This note records which experiments ran, with their sources. It changes no
decision above, and no text above is removed.

- **E1 (`r_match`): run.** `r_match` = 0.75 arcsec, chosen by the ADR 0004 G-A2 criterion on the
  100k batch plus one offset run (`scripts/xmatch_offset.py`; offset manifest
  `data/interim/xmatch/xmatch_gaiadr3_offset180_n100000_r5_all.manifest.json`, `obtained_utc`
  2026-10-26T12:34:48+00:00; `data/interim/xmatch/e1_report.json`: `r_c` STAR 0.75, GALAXY 0.25,
  QSO 0.5). Recorded in `docs/research/R-09-no-login-path.md:130-151`. Used by full_v0:
  `data/interim/full_v0/dataset_report.json` `r_match_arcsec` = 0.75, and ADR 0001 `:19-21`. Caveats
  (R-09 `:135-146`): the plateau criterion is weak on this data, 0.75" is driven by STAR, and the radius
  biases the sample against high-proper-motion stars. Not produced by that run (R-09 `:147`): the
  multi-pair rate, the 0.1" grid, and local epoch propagation. The E1 text above ("epoch-propagated",
  "compare with DR13 `best_neighbour`") is therefore met only in part.
- **E2 (DR13 check): not run.** It is deferred (as text, not a status) by ADR 0004 D1 (`:177-180`) and
  ADR 0004 Consequences (`:408-409`): no DR13 source exists on the XMatch path. Trigger 3 of "What would
  change my mind" cannot be evaluated.
- **E3 (`z_star_max`, `z_qso_min`): not run, and the §2 redshift-consistency rule is not implemented.**
  Owner decision of 2026-11-03 (GCUT-RC3): keep full_v0 as built. Recorded in ADR 0006 `:82-85`
  (known limitation), risk R13 (`:522`) and reversal trigger T7 (`:542-544`: implementing the rule
  changes the labels and redoes the LightGBM seeds, `delta.json`, every DL run and `val_freeze.json`).
  No occurrence of `z_star_max` or `z_qso_min` in `src/`, `scripts/`, `kaggle/` or `tests/` (search
  of 2026-11-16). The columns an optional measurement would need exist in
  `data/interim/full_v0/labels.parquet` (`sdss_z`, `sdss_zwarning`, `sdss_class`; schema read, values
  not read).
- **E4 (DR15 first-release date) and E5 (DSC training flag): not run.** Both serve only cut (b), and
  cut (b) was not produced on full_v0: `data/interim/full_v0/eval/results.md:12` and `:21`
  ("ADR 0002 cut (b): not available: DR14Q/DR15 not ingested"; the same text is emitted by
  `src/astro_triage/evaluation/runner.py:79` and `:912`). No execution evidence found in the
  repository for either.
- **E6 (DR20 SpecObj): not run.** No DR20 reference in `src/`, `scripts/`, `kaggle/` or `tests/`.
  Labels stay DR17 (§1, L2).
- **`labels_v1`: not built.** The Status line above makes E1 to E6 a precondition, and only E1 ran.
  The labels in use are the full_v0 release
  (`data/interim/full_v0/full_v0_labels.manifest.json`: `release` full_v0, 2,044,246 rows).
- **Documented deviation from §2.** full_v0 applies the §2 conditions `ZWARNING == 0`, class in
  {STAR, GALAXY, QSO}, `SPECPRIMARY == 1` and finite plug positions (ADR 0001 `:17-19`;
  `src/astro_triage/labels/specobj.py:130`), but not the redshift-consistency condition (E3 above).
  Every full_v0 metric is computed on labels that did not pass the full §2 filter.
