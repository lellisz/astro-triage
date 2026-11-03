# ADR 0004: Data path without a Gaia Archive account (VizieR, CDS XMatch, SDSS SAS)

- Status: **Proposed (gates G-A partially passed; G-B pending).** It becomes Accepted only after gates
  G-A and G-B (section 7) pass and a human approves. Until then, nothing in `data/` is built from this
  route, except the SDSS file of D2, which the user approved.
  *(A-04c: G-A1 and G-A3 passed on the 100,000-position batch of 2026-10-26. Remaining: G-A2, G-B and
  human approval; see section 7.)*
- Date: 2026-10-25. Amended 2026-10-26 (task A-04b) with the measurements of R-09 "Update" and
  "XMatch smoke test" and with the user decisions of 2026-10-25/26.
  Amended again 2026-10-26 (task A-04c) with the measurements of R-09 "100k batch": the four columns
  that A-04b planned to remove exist in the XMatch output and are mapped; the removal is reverted.
  Amended text is marked "(A-04c)". Original text is kept; superseded text is struck through.
- Task: A-04, A-04b, A-04c
- Amends: ADR 0002 §1 (the crossmatch engine is now chosen; ADR 0002 deferred that choice), ADR 0002 §2
  and §4 cut (c) (A-04b: SDSS position and primary columns, DSC coverage rule), and the `Origin` column
  of `docs/specs/schema.md` (see schema §9). ADR 0001 and ADR 0003 are unchanged.
  *(A-04c: also refines the DSC coverage rule of ADR 0002 §4 cut (c) into two reported layers (D5), and
  proposes, without applying, an addition to the evaluation report of ADR 0003 (D7.4).)*
- Main source: `docs/research/R-09-no-login-path.md` (researcher, confidence low to medium, plus the
  orchestrator's anonymous smoke tests, including the "Update" and "XMatch smoke test" sections).
  Background: `docs/research/R-00-digest.md` round 3.
  *(A-04c: and the R-09 section "100k batch", which is measured.)*

## Context

- The project cannot log in to the Gaia Archive. The ESA password page
  (`cas.cosmos.esa.int/cas/passwd`) returned HTTP 503 on 2026-10-25 while other ESA services answered
  200 (`.claude/state/backlog.md`, "Human-dependent"). Anonymous Gaia Archive use is capped at 50,000
  rows per query with a 1-minute async timeout (R-00 round 3). User tables, which ADR 0002 §1 option (i)
  needs, require an account.
- R-09 found a route that needs no account:
  - VizieR TAP (`https://tapvizier.cds.unistra.fr/TAPVizieR/tap`) serves `"I/355/gaiadr3"`
    (225 columns, join key `Source`) and `"I/355/paramp"` (215 columns, join key `Source`). The column
    names differ from the Gaia Archive names. The advertised upload limit is 100,000 rows per job.
  - CDS XMatch (`http://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync`) has a 100 MB upload limit, a
    maximum radius of 180 arcsec, output truncated at 2 M rows, no epoch propagation in the documented
    API, and no login.
  - The SDSS DR17 `specObj-dr17.fits` file (6.7 GB) is on the SAS with no login. SkyServer SQL works
    without login, but its DR17 row limit is not verified.
- Measured by the orchestrator on 2026-10-25 (anonymous, `/sync`):
  - A two-table JOIN with TOP 5 worked. `RUWE`, `epsi`, `PQSO-A`, `PGal-A` and `Pstar-A` exist and
    return values, and `RUWE` came back NULL for some sources.
  - A full-table COUNT timed out at 120 s.
  - TAP_UPLOAD accepts only VOTable; a CSV upload was rejected. Two VOTable uploads of 3 ids timed out
    after 90 to 120 s, so upload plus JOIN is **unverified**.
- Measured later (R-09 "Update", 2026-10-25, and "XMatch smoke test", 2026-10-26 UTC). Added by A-04b:
  - The earlier VizieR tests had a quoting bug: PowerShell 5.1 strips embedded double quotes from
    arguments passed to `curl.exe`, so `"I/355/gaiadr3"` reached the server unquoted. With correct
    quoting, a plain `SELECT TOP 200 "Source" ... WHERE "Gmag" < 9` returns in about 3 s on `/sync`.
  - Upload (VOTable, 200 ids) joined to `"I/355/gaiadr3"`: `/sync` timed out at 100 s, and `/async`
    stayed `EXECUTING` for 180 s and was aborted. Upload plus JOIN is **unusable at the moment**.
  - CDS XMatch works over **https** (`https://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync`); the
    `http://` endpoint reset the connection. 1,000 SDSS positions against `vizier:I/355/gaiadr3`,
    `distMaxArcsec=5`, `selection=best`: HTTP 200 in 6 s, **467 matched (46.7%)**, no duplicate input id.
  - XMatch **returns the VizieR columns directly**: 142 columns for `vizier:I/355/gaiadr3` (not 225;
    `RA_ICRS`, `DE_ICRS` and `NSS` absent, `RAdeg`, `DEdeg` and `angDist` present) and 220 columns for
    `vizier:I/355/paramp`, both with `Source`. The paramp pass returned the same 467 rows.
  - Only **262 of 467** matched rows (56%) have non-null allosmod probabilities (`PQSO-A`, `PGal-A`,
    `Pstar-A`). The DSC was not run for every source.
  - Match rate by SDSS class in the sample: STAR 84.2%, QSO 59.9%, GALAXY 31.1%. The sampling error is
    a few points.
  - `angDist` on the 467 matches: median 0.080", p95 0.438", max 4.619".
  - Output size is about 780 bytes per matched row in CSV.
- (A-04c) Measured on the **100,000-position batch** (R-09 "100k batch", 2026-10-26; gate G-A1):
  - Run: 100,000 random eligible SDSS positions (seed 20261025), `distMaxArcsec=5`, `selection=all`,
    both catalogues, HTTP 200, **65 s**. Output: gaiadr3 pass **52,524 rows** (142 columns), paramp pass
    **51,850 rows** (220 columns); both far below the 2,000,000-row truncation.
  - Matched SDSS objects: **49,724 of 100,000 (49.7%)**. By class: STAR 19,088/20,964 = **91.1%**,
    QSO 11,473/17,471 = **65.7%**, GALAXY 19,163/61,565 = **31.1%**.
  - Pairs per matched object (all pairs within 5"): median 1, p95 2, max 15. Nearest-pair `angDist`:
    median 0.082", p95 0.476", max 5.0" (the radius cap, so the tail is truncated by `r_search`).
  - 6 Gaia sources are matched by two different SDSS objects (one source, two spectra at different
    plug positions). ADR 0002 §3 (one row per source) applies.
  - **Upload id bug (ours, fixed).** XMatch parses an all-digit upload column as double, so the 19-digit
    `SPECOBJID` came back rounded. Fix: the upload id is sent as text, `"S" + specobjid`. After the fix,
    all 52,524 returned ids exist in the upload, the class agrees, and the echoed ra/dec equal the upload
    (max difference 2e-6 arcsec).
  - **Columns.** `ra_error` = `e_RAdeg`, `dec_error` = `e_DEdeg`, `astrometric_excess_noise_sig` =
    `sepsi`, `astrometric_params_solved` = `Solved` exist in the gaiadr3 XMatch output. There is no
    `GLON`/`GLAT`. `QSO`, `Gal` (in_qso_candidates, in_galaxy_candidates) and `NSS` (non_single_star)
    are **not** in the 142 XMatch columns; they exist only in the 225-column TAP table.
  - **paramp coverage.** The paramp pass lacks **674 of 52,518** distinct Gaia sources (1.3%; fainter,
    median G 20.02). Of the paramp rows, only **59.7%** have non-null `PQSO-A`.
  - **Missing astrometry.** `Solved` = 3 (2-parameter solution: no parallax, no proper motion, no RUWE)
    on **40.4%** of matched rows. RUWE NULL share by class: GALAXY 87.9%, QSO 18.5%, STAR 6.5%.
- R-09 could not read column descriptions, because TAP_SCHEMA came back empty and the web pages were
  blocked by an anti-bot page. Every name mapping below comes from column headers and naming convention.
  None comes from a description.

## Options

| # | Option | Real cost |
|---|---|---|
| O1 | **VizieR TAP for features and DSC, CDS XMatch for the match, SDSS file for labels** | New name mapping to maintain. Upload is unverified, and jobs are 100k rows each (10 to 50 jobs). The DSC name semantics are unverified. XMatch has no epoch propagation. Depends on CDS fair-use limits that we have not found. *(A-04b: the measured variant is O1 with XMatch returning the VizieR columns, see D4; the upload cost no longer applies.)* |
| O2 | Wait for the Gaia account (ADR 0002 §1 option (i), server-side ADQL crossmatch) | Zero new code. Unbounded calendar delay that the ESA helpdesk controls, and the project stays idle. |
| O3 | Gaia CDN bulk files (`GaiaSource_*.csv.gz` and `AstrophysicalParameters_*.csv.gz`), with stream-filter-delete and a local match | About 757 GB of transfer against 83 GB of free disk. Whether the files contain the `classprob_dsc_*` columns is not verified. Days of download. A local crossmatch has to be written. |
| O4 | Anonymous Gaia Archive, in many queries of at most 50,000 rows | Canonical names, with no mapping. No user-table upload, a 1-minute async timeout, and 100 active slots shared by all anonymous users. Batching by id lists inside ADQL text is fragile and was not tested. |

## Decision

O1 is adopted provisionally, with O4 kept as a **verification oracle** for small samples (G-B). O2 stays
open as the reversal path (section 8).

### D1. Source of features and DSC

- `"I/355/gaiadr3"` supplies the `gaia_source` columns and `"I/355/paramp"` supplies the
  `astrophysical_parameters` columns. Both are read through the anonymous VizieR TAP and joined on `Source`.
  *(Amended A-04b: both are read through CDS XMatch, one pass per table, and joined locally on `Source`;
  see D4.)*
- **The canonical name is the Gaia Archive name.** VizieR names exist only inside the ingest adapter and
  in the versioned map `manifests/column_map_vizier_v1.csv`. That map is created by the ingest task, not
  by this ADR. Every table in `docs/specs/schema.md`, and all code downstream of ingest, uses Gaia Archive
  names only. When a Gaia account becomes available, only the adapter changes.
- Mapping (from the R-09 column headers; "XMatch" = seen in the 2026-10-26 XMatch output):

| Canonical (Gaia Archive) | VizieR table | VizieR column | Status |
|---|---|---|---|
| source_id | gaiadr3 / paramp | `Source` | header seen; XMatch output of both passes |
| ra, dec | gaiadr3 | `RA_ICRS`, `DE_ICRS` | header seen; epoch 2016.0 to be checked in G-B. **XMatch output has `RAdeg`, `DEdeg` instead**; their equivalence to `ra`, `dec` (epoch 2016.0) is checked in G-B. *(A-04c: on the XMatch path the source is `RAdeg`, `DEdeg`. Training knowledge, not verified: in VizieR I/355, `RA_ICRS`/`DE_ICRS` may be positions propagated by CDS to epoch 2000, and `RAdeg`/`DEdeg` the epoch-2016.0 Gaia positions. G-B decides which pair equals `ra`/`dec`, and E1 records which pair XMatch matches on.)* |
| (A-04c) ra_error, dec_error | gaiadr3 | `e_RAdeg`, `e_DEdeg` | (A-04c) XMatch, 100k batch. Unit (Gaia: mas) checked in G-B |
| parallax, parallax_error | gaiadr3 | `Plx`, `e_Plx` | header seen; XMatch |
| pmra, pmra_error, pmdec, pmdec_error | gaiadr3 | `pmRA`, `e_pmRA`, `pmDE`, `e_pmDE` | header seen; `pmRA`, `pmDE` in XMatch, errors not yet confirmed there (G-A3). *(A-04c: not among the absent columns of the G-A3 comparison; present)* |
| phot_g/bp/rp_mean_mag | gaiadr3 | `Gmag`, `BPmag`, `RPmag` | header seen; XMatch |
| phot_g_mean_flux_over_error | gaiadr3 | `RFG` | header seen (R-09 pairs `RFG/RFBP/RFRP` with flux_over_error); XMatch |
| bp_rp | gaiadr3 | `BP-RP` | header seen; needs quoting in ADQL; XMatch presence not yet confirmed (G-A3). *(A-04c: present)* |
| ruwe | gaiadr3 | `RUWE` | header seen, values returned; NULL in some rows; XMatch |
| astrometric_excess_noise | gaiadr3 | `epsi` | header seen, values returned; XMatch |
| (A-04c) astrometric_excess_noise_sig | gaiadr3 | `sepsi` | (A-04c) XMatch, 100k batch; semantics checked in G-B |
| (A-04c) astrometric_params_solved | gaiadr3 | `Solved` | (A-04c) XMatch, 100k batch; value 3 on 40.4% of matched rows. Encoding (3/31/95) checked in G-B and by schema R2 |
| ipd_gof_harmonic_amplitude | gaiadr3 | `IPDgofha` | header seen; XMatch |
| ipd_frac_multi_peak, ipd_frac_odd_win | gaiadr3 | `IPDfmp`, `IPDfow` | header seen; XMatch presence not yet confirmed (G-A3). *(A-04c: present)* |
| phot_bp_rp_excess_factor | gaiadr3 | `E(BP/RP)` | header seen; description unread; XMatch |
| in_qso_candidates, in_galaxy_candidates | gaiadr3 | `QSO`, `Gal` | header seen; type (bool or 0/1) unverified; XMatch presence not yet confirmed (G-A3). *(A-04c: **absent** from the XMatch output; TAP table only. NULL on the XMatch path, and excluded from the LightGBM features on every path (D7.3).)* |
| (A-04c) non_single_star | gaiadr3 | `NSS` | (A-04c) absent from the XMatch output; TAP table only. Not a schema v1 column (D1, last bullet) |
| classprob_dsc_combmod_{quasar,galaxy,star} (3 classes, `gaia_source`) | gaiadr3 | `PQSO`, `PGal`, `PSS` | **inference only.** These names come from the task brief and are not in the R-09 header list. That they equal the 3 `gaia_source` combmod columns is assumed, not verified. XMatch presence not yet confirmed (G-A3). *(A-04c: not among the absent columns of the G-A3 comparison; semantics still inference only until G-B)* |
| classprob_dsc_allosmod_{quasar,galaxy,star} | paramp | `PQSO-A`, `PGal-A`, `Pstar-A` | **unverified.** Values returned, but that the `-A` suffix means allosmod is convention only. This is the fair baseline of ADR 0002 cut (c), so it may not be used until G-B passes. XMatch: present; non-null on 262 of 467 matched rows. *(A-04c: 100k batch: non-null on 59.7% of paramp rows; 674 of 52,518 Gaia sources have no paramp row)* |
| (5-class combmod, specmod) | paramp | `PQSO`, `PGal`, `Pstar`, `PWD`, `Pbin`; `PQSO-S`, ... | not ingested in v1 (schema OQ9). `PQSO`/`PGal` share names with gaiadr3, so any JOIN must alias by table |

- Canonical columns with **no VizieR column identified** in R-09: `l`, `b`, `ra_error`, `dec_error`,
  `parallax_over_error`, `pm`, `astrometric_excess_noise_sig`, `astrometric_params_solved`. Rule:
  *(A-04c: corrected by the 100k batch. `ra_error`, `dec_error`, `astrometric_excess_noise_sig` and
  `astrometric_params_solved` are mapped (table above). Only `l`, `b`, `parallax_over_error` and `pm`
  have no VizieR column in the XMatch output, and all four are derived locally under the rule below.)*
  - A column is **derived locally** only when its Gaia definition is a closed formula over columns
    already mapped: `l`, `b` (ICRS to Galactic with astropy), `parallax_over_error`
    (`parallax / parallax_error`, NULL if either is NULL) and `pm` (`sqrt(pmra² + pmdec²)`, NULL if
    either is NULL). Its schema `Origin` becomes `derived (ADR 0004)`.
    *(A-04c: `l`, `b` are computed from the canonical `ra`, `dec` (ICRS, the epoch-2016.0 pair chosen
    by G-B), never NULL because `ra`, `dec` are non-null. `parallax_over_error` is
    `float32(Plx / e_Plx)` and is NULL when `Plx` or `e_Plx` is NULL, which on real data happens on
    every `Solved` = 3 row; a zero `e_Plx` would also give NULL, never ±inf, and is counted in the
    ingest log. G-B compares the derived values with the Gaia Archive values.)*
  - Any other column without a VizieR source is a **G-B blocker**. Either G-B finds its VizieR column,
    or the schema loses the column in a new version through a separate decision. It is never filled
    with a sentinel.
  - ~~**Decision (user, 2026-10-25/26; recorded A-04b):** the four remaining columns `ra_error`,
    `dec_error`, `astrometric_excess_noise_sig` and `astrometric_params_solved` are **removed in a new
    schema version** and documented as a limitation of the no-login route. The project does not wait
    for the Gaia account to fill them. If the full 142-column comparison of G-A3 finds a VizieR source
    for any of them, that column is mapped instead of removed. They are no longer G-B blockers. The
    schema revision itself is a separate task (schema §9.2).~~
  - **(A-04c) User decision, restated.** On 2026-10-25/26 the user authorised removing the four columns
    `ra_error`, `dec_error`, `astrometric_excess_noise_sig` and `astrometric_params_solved` **only if**
    the no-login route had no source for them. The G-A3 comparison on the 100k batch found all four
    (`e_RAdeg`, `e_DEdeg`, `sepsi`, `Solved`). The condition is false, so **no column is removed**, no
    new schema version is needed for them, and schema §2 keeps them with their v1 type and nullability.
    The schema revision task that A-04b announced is cancelled.
  - **(A-04c) Columns of the 225-column table that the XMatch output lacks.** `QSO`, `Gal` and `NSS`.
    - `in_qso_candidates`, `in_galaxy_candidates` (`QSO`, `Gal`): kept in `raw` as **nullable**, NULL on
      every row of the XMatch path (schema §2 rows 29-30), so that the canonical schema stays the same
      when the account route returns. They are never fed to LightGBM (D7.3), so their absence costs
      nothing on the model side.
    - `non_single_star` (`NSS`): not a schema v1 column, and it stays out of v1. Status: optional;
      absent on the XMatch path; if a later version adds it, it is nullable and NULL on every
      XMatch-path row. It is never filled with a sentinel.
    - None of the three is a G-B blocker.
- The DR13 best-neighbour columns (raw 37 to 40) have no identified VizieR source. R-09 notes a built-in
  `SDSS13` crossmatch in VizieR, but its content was not examined. Interim rule: columns 37 to 40 are
  NULL on every row, which satisfies schema R4, and E2 of ADR 0002 is **deferred** until a DR13 source is
  confirmed or the account works.

### D2. Labels: SDSS DR17 SpecObj

| Option | Pros | Cons |
|---|---|---|
| L-F: `specObj-dr17.fits` from the SAS | Full schema, one immutable artefact, offline reruns, matches ADR 0002 §1 | 6.7 GB download and disk space |
| L-S: SkyServer SQL, paged | Small transfer, only the needed columns | DR17 row limit unverified (DR9 had 500,000 rows, 600 s and 60 queries/min), paging is fragile, and server-side state could drift between pages |

- **Recommendation: L-F.** The decision is **blocked on user approval**. The user has not approved the
  6.7 GB download, and this ADR does not authorise it.
- **Update (A-04b): L-F is decided.** The user approved the download, and it is complete:
  6,741,023,040 bytes (equal to Content-Length), 5,801,200 rows, 133 columns, stored in
  `data/external/sdss/` and outside git.
- Facts about the file that bind ADR 0002 (amended there):
  - The file has **no `RA`, `DEC` or `SCIENCEPRIMARY`** columns. Positions come from `PLUG_RA`,
    `PLUG_DEC`. The primary flag is `SPECPRIMARY`.
  - **Eligible rows**: `SPECPRIMARY == 1`, `ZWARNING == 0`, `CLASS` in {STAR, GALAXY, QSO}, and finite
    `PLUG_RA`/`PLUG_DEC`. There are **4,283,507** of them: STAR 890,744, GALAXY 2,639,535, QSO 753,228.
    The z-consistency rule of ADR 0002 §2 (E3) is applied after this count.
  - `PLUG_RA`/`PLUG_DEC` are the **fibre/plug positions**, not a final astrometric position. Their
    positional uncertainty and its effect on `r_match` are part of **E1** (ADR 0002). The baseline for E1
    is the measured `angDist` distribution on 467 matches (5" radius, unpropagated): median 0.08",
    p95 0.44", max 4.62".
    *(A-04c: superseded as baseline by the 100k batch, nearest pair per object, 5", unpropagated:
    median 0.082", p95 0.476", max 5.0" at the radius cap.)*

### D3. Crossmatch: CDS XMatch

- Engine: CDS XMatch, uploading SpecObj positions (`specobjid`, ra, dec) against
  `vizier:I/355/gaiadr3`, in batches below the 100 MB upload limit and the 2 M output-row truncation.
  The XMatch output must return **every pair** inside the search radius, not only the best match,
  because ADR 0002 §3 needs the counts of SpecObj rows with several Gaia sources.
  *(A-04b: the endpoint is `https://`; the uploaded positions are `PLUG_RA`/`PLUG_DEC` (D2). The smoke
  test used `selection=best`; production uses all pairs. A batch's expected output, about 780 bytes per
  row, must stay below both the 2 M row truncation and the upload limit.)*
  *(A-04c: measured at 100,000 positions per batch: 52,524 and 51,850 output rows, 65 s. A batch of
  100,000 positions is therefore about 40 times below the 2 M row cap; the full run of 4,283,507
  eligible positions is about 43 batches of two passes each.)*
- **(A-04c) Upload id rule (permanent).** Every id column sent to XMatch is **text with a non-digit
  prefix**: the SpecObj id is uploaded as `"S" + specobjid` and the prefix is stripped on read. An
  all-digit id column is parsed by the service as double and loses precision at 19 digits. The ingest
  task asserts after every job that each returned id exists in the upload and that the echoed positions
  equal the uploaded ones within 1e-5 arcsec. The rule is recorded in `raw.xmatch_request` (D6).
- Search radius `r_search ≤ 180 arcsec` (the service maximum). `r_match` remains experiment **E1** of
  ADR 0002. This ADR adds the upper bound `r_match ≤ r_search ≤ 180 arcsec`.
- Epoch: XMatch compares Gaia epoch-2016.0 positions with SDSS positions without proper-motion
  propagation. Mitigation, which keeps the propagation rule of ADR 0002 §1:
  1. XMatch with an `r_search` comfortably larger than the expected `r_match` returns the candidate pairs.
  2. The pipeline fetches `pmra`/`pmdec` for the candidates (D4). *(A-04b: these come in the XMatch
     output itself; no separate fetch.)*
  3. It propagates the Gaia position to the SDSS epoch locally.
  4. It recomputes `angular_distance_arcsec` and applies `r_match`.
  `pm_propagated` keeps the schema meaning.
  *(A-04c: on `Solved` = 3 rows (40.4% of matched rows) `pmra`/`pmdec` are NULL, so no propagation is
  possible and `pm_propagated` is false. Which position pair XMatch itself matches on, epoch 2016.0 or
  a CDS-propagated epoch, is not known (D1 table, ra/dec row) and is recorded by E1.)*
- Open sub-issue: which SDSS epoch to propagate to. The SpecObj position comes from targeting
  astrometry, which may not have the epoch of the spectroscopic `MJD`. This is resolved within E1.
  Interim rule: use `MJD` and record the rule in the `labels` metadata.
- **New risk R-PM: high proper-motion stars may fail to match.** A star whose Gaia-to-SDSS offset is
  larger than `r_search`, or whose unpropagated offset puts it outside the candidate set, never enters
  the labelled set. This biases the sample against nearby stars.
  - Measurement, using only our own tables. Among kept STAR labels, report the match rate and the
    distribution of the unpropagated XMatch separation per bin of total proper motion
    (`pm`: <10, 10–50, 50–200, 200–1000, >1000 mas/yr) and per bin of parallax (<1, 1–5, 5–20, >20 mas).
    Also report the count of pairs whose propagated separation is ≤ `r_match` but whose unpropagated
    separation is > `r_match`, which are the objects the propagation rescued.
  - Detection rule: if the fraction of candidates with unpropagated separation > 0.8·`r_search` grows
    with `pm`, the search radius is truncating real matches.
  - The result goes into the **selection-bias section** of the evaluation report (ADR 0002 Limitation 1,
    ADR 0003 "SDSS selection bias") as a named item.
- **Class-dependent match rate (measured, A-04b).** On a random sample of 1,000 eligible SpecObj rows
  (seed 20261025, 5", best match), the match rate is STAR 84.2%, QSO 59.9%, GALAXY 31.1%, with a sampling
  error of a few points. The class mix of the matched sample therefore differs strongly from the SDSS
  mix. This is a **central selection effect**: it is reported as a named item in the selection-bias
  section, with the per-class match rate of the full run, and it is the reason the realistic-prior regime
  of ADR 0003 reweights instead of using the matched-sample mix. The rough projection of about 2.0 M
  labelled sources with a Gaia counterpart (before the G < 20 cut) is not a measurement and is not used
  in any design (D5).
  *(A-04c: confirmed on 100,000 positions, all pairs within 5": STAR 91.1%, QSO 65.7%, GALAXY 31.1%,
  overall 49.7%.)*

### D4. Retrieving features by id

- Planned primary path: VizieR TAP_UPLOAD of matched `Source` ids, sent as a **VOTable** (CSV is
  rejected), at most 100,000 rows per job, LEFT JOIN `"I/355/gaiadr3"` to `"I/355/paramp"`.
  Status: **unverified**. Two `/sync` attempts with 3 ids timed out after 90 to 120 s.
- **Superseded (A-04b). New primary path: XMatch returns the columns.** Two XMatch passes over the same
  uploaded SDSS positions, one with `cat2=vizier:I/355/gaiadr3` and one with `cat2=vizier:I/355/paramp`,
  each returning that table's VizieR columns. The two outputs are **joined locally on `Source`**, as a
  LEFT JOIN from the gaiadr3 pass to the paramp pass. A gaiadr3 pair with no paramp row gets NULL in
  every paramp column. No id fetch is needed, so the upload plus JOIN becomes unnecessary. Evidence:
  1,000 positions, 467 matches, 142 and 220 columns, 6 s (R-09, 2026-10-26).
  - Both passes use the same `r_search` and the same pair selection. Only pairs whose `Source` appears
    in the gaiadr3 pass enter `raw`.
  - (A-04c) Evidence at scale: 100,000 positions, 52,524 and 51,850 rows, 65 s. The LEFT JOIN leaves
    674 of 52,518 Gaia sources (1.3%) without a paramp row. The adapter records, per source, whether
    the paramp row was found (`raw.ap_row_present`, schema §2 row 41), because a NULL allosmod value
    alone cannot tell "no paramp row" from "paramp row without allosmod output" (D5).
- Contingency, in order. The ingest task tries each one only after the previous one fails G-A:
  - (b) **No upload.** Query by `Source` ranges: sort the matched ids, merge them into contiguous
    intervals (`"Source" BETWEEN lo AND hi`), keep the matched ids locally and discard the extra rows.
    Because `source_id` encodes HEALPix level 12, each interval is a small sky patch.
  - (d) The **ARI Heidelberg or AIP** TAP mirrors. Neither was tested anonymously, and whether they hold
    DR3 and DSC data is not verified.
  - (a) The same upload on **`/async`**, with UWS polling. **Status: unusable at the moment.** With
    correct quoting, 200 ids timed out at 100 s on `/sync` and stayed `EXECUTING` for 180 s on `/async`.
    Retried only if (b) and (d) fail.
  - (c) *Promoted to the primary path above.*
  - (e) **Wait for the Gaia account** (O2).
- Every path must produce identical canonical columns. The path used is recorded in the file metadata
  (D6).
- Query transport (A-04b): any ADQL is sent from a file or through a library, never as a shell
  argument (see Consequences).

### D5. Sample size

- Sample size is **unknown** until the match runs. No number is assumed in any design.
- Target: in the test fold of ADR 0003, every class meets the reporting floor (`n_c ≥ 100` over
  `≥ 20` level-5 pixels) in cut (a) and in cut (b). Per latitude band, the floor is a bonus.
- If the sample is small:
  - Cells below the floor show `n/a (n=..., pixels=...)`, as ADR 0003 requires.
  - Intervals from the pixel bootstrap are reported however wide they are, and conclusions are
    restricted to the cells that pass the floor.
  - Do **not** widen `r_match` beyond the E1 knee, relax the quality filter of ADR 0002 §2, or reseed
    beyond ADR 0003 rule 5 to make the numbers bigger.
  - Do not extend a conclusion from one cut, class or band to another.
- DSC coverage (A-04b): cut (c) and every DSC metric run on the subset with non-null DSC output
  (ADR 0002 §4, amended). In the smoke sample that subset is 56% of the matches, so the floor check above
  is applied to that subset too, per class.
- (A-04c) The coverage rule of ADR 0002 §4 cut (c) is unchanged and binding. The 100k batch shows two
  distinct causes of a missing allosmod value, so the coverage is **reported in two layers**, per class
  and per cut, next to every DSC metric:
  1. **No paramp row**: `ap_row_present = false` (1.3% of Gaia sources in the 100k batch; fainter,
     median G 20.02).
  2. **Paramp row without allosmod output**: `ap_row_present = true` and the allosmod columns NULL
     (40.3% of paramp rows in the 100k batch).
  `S_dsc` (ADR 0002) is the remainder. The two layers are computed from `raw` alone (schema R6). A
  missing value in either layer is never mapped to a class and never counted as an abstention. The
  combmod coverage (`raw` 31-33) is reported with the same two-layer table only if G-B shows that its
  NULL pattern has more than one cause; otherwise as one layer.

### D6. Consequences for the schema (applied in `docs/specs/schema.md` §9)

- **Explicit NULL.** VizieR returns empty values, for example `RUWE` for some sources. The adapter
  converts empty, masked and NaN values to Parquet NULL before writing (schema rule 3). It never writes
  0, an empty string, or a sentinel.
- **Canonical names** as in D1. The adapter renames before the explicit `pa.schema` is applied, so G1
  catches a missing or misnamed column.
- **Manifest per source.** The `raw` metadata records:
  - `source_service` = `"VizieR/CDS TAP"` *(A-04b: `"CDS XMatch"` on the primary path)*
  - `vizier_tables` = `["I/355/gaiadr3","I/355/paramp"]`
  - `column_map` (path and SHA-256 of `manifests/column_map_vizier_v1.csv`)
  - `retrieval_path` (one of `upload_sync`, `upload_async`, `source_range`, `xmatch_columns`, `mirror:<name>`)
  - `adql_query` (the template) *(A-04b: ADQL paths only)*
  - `xmatch_request` *(A-04b: the XMatch request parameters of each pass, without the uploaded file;
    `xmatch_columns` path only)* *(A-04c: includes the upload id column name and its text prefix `"S"`)*
  - `n_jobs`

  The `labels` metadata records:
  - `match_engine` = `"CDS XMatch"`
  - `xmatch_catalogue` = `"vizier:I/355/gaiadr3"` *(A-04b: the catalogue of the match pass; the
    paramp pass is recorded in `raw.xmatch_request`)*
  - `r_search_arcsec`
  - `sdss_epoch_rule`

  No v1 file exists yet, so these keys are part of v1 and do not bump the version.
- (A-04c) **Schema changes made by A-04c** (no v1 file exists, so all stay in v1; exact list in schema
  header and §9.2): `raw` 29-30 become nullable; new `raw` column 41 `ap_row_present`; `features` rows
  23-24 removed; new invariant R6; F3 widened; the A-04b write lock on `raw` is lifted.
- **Citations to verify before any public output:** VizieR, CDS XMatch, the I/355 catalogue entry, and
  the Gaia DR3 reference (Vallenari et al. 2023). None of their texts was read in R-09.

### D7. (A-04c) Missing astrometry and Gaia-classifier columns

1. **Missing astrometry is NULL.** On `Solved` = 3 rows (40.4% of matched rows) `parallax`,
   `parallax_error`, `parallax_over_error`, `pm`, `pmra`, `pmra_error`, `pmdec`, `pmdec_error` and `ruwe`
   have no value. They are Parquet NULL in `raw` and `features`, never imputed, never a sentinel, and
   never NaN on disk (schema rule 3, G3). In memory, pandas may show them as NaN; that is the form
   LightGBM reads.
2. **`astrometric_params_solved` is a legitimate feature.** It is a property of the Gaia solution, is
   published for every Gaia source, and is available at inference time on any Gaia query. It stays in
   `features` (row 13). LightGBM handles the NULLs of item 1 natively (default missing-value handling,
   no `zero_as_missing`); no imputation step exists anywhere in the pipeline.
3. **`in_qso_candidates`, `in_galaxy_candidates` are excluded from the LightGBM features, on every
   path.** Reason: membership in the DR3 QSO and galaxy candidate tables is decided by Gaia's own
   classifiers, the DSC among them, so the flags carry DSC output and, through it, DSC's SDSS training
   labels. Feeding them to LightGBM would break cut (c) of ADR 0002 ("same information": photometry and
   astrometry only, no DSC output) in the same way a `classprob_dsc_*` feature would. They are excluded
   for the same reason, independently of their absence from the XMatch output. Cost: none on the
   XMatch path (they are NULL there anyway); on the account path, a possible loss of accuracy that is
   by design not attributable to LightGBM.
4. **Evaluation requirement: missingness is strongly tied to class** (RUWE NULL: GALAXY 87.9%,
   QSO 18.5%, STAR 6.5%). The model can learn "no astrometry, so not a star", which is real information
   at inference but also mirrors the SDSS selection. Two items are therefore required in every
   evaluation report, on the test fold, for cut (a) and cut (c) (both `S` and `S_dsc`), in the BALANCED
   regime, with the ADR 0003 reporting floor and pixel bootstrap applied per cell:
   - **Ablation "with and without missingness indicators".** Model M-full: the `features` table as
     specified. Model M-noind: the same training run without `astrometric_params_solved` and without
     the nine columns of item 1, so the solution type cannot be read from either a flag or a NULL
     pattern. Report per-class purity, completeness and macro-F1 of both, and the paired difference
     with its bootstrap CI (same replicates, ADR 0003 "Uncertainty"). NULLs in photometric columns are
     left as they are in both models.
   - **Stratified performance by `astrometric_params_solved`**: strata 3 (2 parameters), 31
     (5 parameters), 95 (6 parameters). Per stratum: class counts, and per-class purity and
     completeness of M-full and of the DSC baselines on their covered rows.
   - Where this is placed, and why: ADR 0003 owns the metrics and is Accepted, so this ADR does not
     change it. The requirement is an additional report item, in the same way R-PM is (D3): it changes
     no split, prior, metric definition or reporting floor. The proposed note for ADR 0003, to be
     applied only with human approval when this ADR is accepted, is: *"Amended by ADR 0004 D7.4: the
     evaluation report adds a missingness ablation (M-full vs M-noind) and per-`astrometric_params_solved`
     strata (3, 31, 95), in BALANCED, on cuts (a) and (c); split, priors, metrics and floor unchanged."*

## Consequences

- Good: the project can proceed without ESA's password service. The canonical names keep the schema,
  the tests and the downstream code unchanged. O4 gives an independent check of the mapping on small
  samples. *(A-04b: XMatch alone delivers match, features and DSC, so no VizieR TAP job is on the
  primary path.)*
  *(A-04c: no `raw` column is lost; the full v1 schema is sourced on the XMatch path, except
  `in_qso_candidates`/`in_galaxy_candidates` (NULL, and not features) and DR13 columns 37-40.)*
- Bad:
  - The ingest now depends on a third-party mirror whose column semantics we inferred.
  - 10 to 50 jobs replace one job, and anonymous limits may be stricter than advertised.
    *(A-04b: on the primary path the count is two XMatch passes per batch of positions; batch size is
    bounded by the 100 MB upload and the 2 M output rows.)*
  - XMatch adds a candidate stage (D3) and a new bias (R-PM) that must be measured and reported.
  - E2 (the DR13 cross-check) is deferred, so the DR13 safety net of ADR 0002 trigger 3 is not
    available until a DR13 source is found.
  - The SDSS file download is still blocked. *(Resolved A-04b: approved and done.)*
  - (A-04b) The matched sample is strongly class-selected (STAR 84%, QSO 60%, GALAXY 31% match rate in
    the smoke sample). Metrics on the matched mix do not describe the SDSS mix.
    *(A-04c: 100k batch: STAR 91.1%, QSO 65.7%, GALAXY 31.1%.)*
  - (A-04b) The DSC has output for only about 56% of matched rows, so DSC metrics describe a subset, and
    LightGBM must be reported on that subset as well as on the full set.
    *(A-04c: 100k batch: 1.3% of sources lack a paramp row, and 59.7% of paramp rows have allosmod
    output; reported in two layers, D5.)*
  - ~~(A-04b) Four `raw` columns are lost (`ra_error`, `dec_error`, `astrometric_excess_noise_sig`,
    `astrometric_params_solved`) unless G-A3 finds them. This is a documented limitation.~~
    *(A-04c: G-A3 found them; nothing is lost.)*
  - (A-04b) SDSS positions are fibre/plug positions, which adds positional scatter to E1.
  - (A-04b) Operational: on PowerShell 5.1, double quotes inside arguments to `curl.exe` are stripped, so
    ADQL must be sent from a file or through a library, never as a shell argument.
  - (A-04c) 40.4% of matched rows have no parallax, proper motion or RUWE, and the share depends
    strongly on class. Part of the model's skill may come from the missingness pattern; D7.4 measures it.
  - (A-04c) Operational: XMatch reads all-digit id columns as double; ids are always uploaded as
    prefixed text (D3).
- Maintenance cost:
  - one column-map CSV and one adapter with a rename step
  - one candidate-propagation step
  - the R-PM table in each evaluation report
  - re-running G-B whenever VizieR updates I/355
  - (A-04b) one local join of the two XMatch passes on `Source`, and the per-class DSC coverage table in
    each evaluation report
  - (A-04c) one extra LightGBM training run per report (M-noind), the `astrometric_params_solved` strata
    table, the second coverage layer, and the post-job id/position echo assertion of D3
- Reversibility: high. When the account works, the adapter is replaced and nothing downstream changes
  (less than one day of work). The one irreversible element is a model trained on VizieR-sourced
  features. Gate G-B makes those values provably equal to the Gaia Archive values.

## 7. Gates to leave Proposed

**G-A Retrieval works (test of D4 and D3).** All of the following must hold:
1. A CDS XMatch job of at least 1,000 positions against `vizier:I/355/gaiadr3` with `r_search` = 30 arcsec
   returns all pairs (not best-only) in under 120 s.
2. A retrieval job (the first path of D4 that works) for **1,000** distinct `Source` ids returns
   **exactly 1,000 rows** whose id set equals the input. The time from submit to result is
   **under 120 s**.
3. Every mapped column of D1 is present under its expected VizieR name.
4. Empty values arrive as NULL. At least one NULL `RUWE` in a sample that contains one is preserved as
   NULL through the canonical rename into Parquet, and G3 passes.
5. A source absent from `paramp` yields NULL allosmod columns through the LEFT JOIN.
6. The run is repeated once on a different day with the same result.

**G-A as revised by A-04b (supersedes the list above, which assumed an id-fetch path).**

Passed on 2026-10-26 (R-09 "XMatch smoke test"):
- P1. XMatch over https, 1,000 eligible SDSS positions, 5", `selection=best`: HTTP 200 in 6 s,
  467 matches (46.7%), no duplicate input id.
- P2. The same positions against `vizier:I/355/paramp`: 220 columns with `Source`, the same 467 rows,
  allosmod columns present (non-null on 262).
- P3. The gaiadr3 pass returns `Source` and the VizieR feature columns directly (142 columns), so the
  id-fetch step of old items 2 and 5 is no longer needed.

Remaining, all of which must hold:
- G-A1. **A batch of at least 100,000 eligible positions** runs through both passes with all pairs
  inside `r_search` (`r_match ≤ r_search ≤ 180"`). Each job returns HTTP 200, and its output row count is
  below 2,000,000 (no truncation). Row counts, bytes and wall time per job are recorded. Because this run
  is on a later day than P1 and P2, it also covers old item 6.
  **(A-04c) PASSED 2026-10-26.** 100,000 positions sent, `r_search` = 5", `selection=all`, both passes
  HTTP 200 in 65 s; 52,524 and 51,850 rows (2.6% of the cap); 49,724 objects matched (49.7%). After the
  upload id fix (D3), every returned id is in the upload. Byte counts are in the run log (R-09).
- G-A2. **`r_match` is decided** by E1 (ADR 0002) on the G-A1 output, after local epoch propagation,
  taking the `PLUG_RA`/`PLUG_DEC` scatter into account. The value and its histogram are recorded.
  **(A-04c) Objective criterion.**
  - Inputs: the G-A1 gaiadr3 output (real run), and one **offset run**: the same 100,000 positions
    with `PLUG_DEC` shifted by +180" (−180" where `PLUG_DEC` > 89.9°), the same XMatch parameters and
    the same upload id rule. Cost: one request of 100,000 positions, gaiadr3 pass only (about 1 min).
    Every pair of the offset run is a chance coincidence, so it estimates the false-match count at the
    local Gaia density.
  - Both runs go through the same local epoch propagation and separation recomputation (D3). For each
    SDSS class c and each radius r on the grid 0.1" to 5.0" in steps of 0.1":
    `N_real_c(r)` = objects with ≥ 1 pair ≤ r in the real run; `N_false_c(r)` = the same in the
    offset run; `N_true_c(r) = N_real_c(r) − N_false_c(r)`; false-match fraction
    `f_c(r) = N_false_c(r) / N_real_c(r)`; multi-pair rate `m_c(r)` = share of matched objects with
    ≥ 2 Gaia sources ≤ r (real run).
  - Rule: `r_c` = the smallest r with `N_true_c(r) ≥ 0.995 · N_true_c(5.0")` (the true-match count has
    reached its plateau). `r_match = max_c r_c`. It is accepted if `f_c(r_match) ≤ 0.01` for every
    class. The false-match and multi-pair rates rise with r (about as r²) and do not stabilise on their
    own; the plateau of the true-match count is the quantity that stabilises, and the 1% cap bounds
    their cost.
  - Fail paths: if `N_true_c` still grows by more than 0.5% between 4.5" and 5.0" for any class, the
    5" `r_search` truncates real matches (see R-PM): rerun G-A1 and the offset run at 10" before
    deciding. If no r meets both thresholds, the orchestrator escalates to the human with the curves.
  - Recorded: `r_match`, the curves `N_true_c`, `f_c`, `m_c` per class, and the histogram of the
    propagated nearest-pair separation. The thresholds 0.995 and 0.01 are reversible parameters of E1.
- G-A3. **Full column comparison.** The complete list of the 142 gaiadr3 columns and the 220 paramp
  columns is compared with every canonical column of schema §2. Each canonical column is marked as
  mapped (VizieR name), derived (D1 rule), or removed (D1 user decision), with no other state. The
  result is `manifests/column_map_vizier_v1.csv`.
  **(A-04c) PASSED 2026-10-26 (R-09 "100k batch").** Every schema §2 column is mapped or derived,
  except `in_qso_candidates`/`in_galaxy_candidates` (absent from the XMatch output: NULL, D1) and
  `dr13_*` (no source: NULL, D1). No column is removed, so the state "removed" is unused. Writing
  `manifests/column_map_vizier_v1.csv` from this result is an acceptance item of the ingest task, with
  the states `mapped`, `derived`, `absent_null`.

Moved out of the gate: old items 4 (NULL preserved through rename) and 5 (absent paramp row gives NULL
allosmod, now through the local LEFT JOIN) become acceptance tests of the ingest task, checked by G3.

**G-B Mapping is correct (test of D1).** All of the following must hold:
1. For 1,000 ids, the anonymous Gaia Archive (O4, under its 50,000-row limit) returns the canonical
   columns, including `classprob_dsc_combmod_*` from `gaia_source` and `classprob_dsc_allosmod_*` from
   `astrophysical_parameters`.
2. Every mapped VizieR column equals its canonical column on every row, with a relative tolerance of
   1e-5 for floats or exactly for ints and bools. NULL patterns must be identical.
3. This confirms or refutes: `PQSO/PGal/PSS` = combmod, `-A` = allosmod, the units and epoch of
   `RA_ICRS`/`DE_ICRS`, the `E(BP/RP)` semantics, and the derived columns of D1 (l, b, pm,
   parallax_over_error). *(A-04b: also `RAdeg`/`DEdeg` of the XMatch output against `ra`/`dec`, and the
   NULL pattern of allosmod, which must match the Gaia Archive and not be an XMatch artefact.)*
   *(A-04c: also `e_RAdeg`/`e_DEdeg` against `ra_error`/`dec_error` (unit mas), `sepsi` against
   `astrometric_excess_noise_sig`, and `Solved` against `astrometric_params_solved` (values 3/31/95).)*
4. Canonical columns without a VizieR source are listed. Each one is either matched to a VizieR column
   here or sent to a schema decision. *(A-04b: the schema decision for the four known columns is made;
   see D1.)* *(A-04c: the four are mapped; the only unsourced columns are `in_qso_candidates`,
   `in_galaxy_candidates` and `dr13_*`, all decided as NULL in D1.)*
5. If anonymous Gaia Archive access is down, G-B falls back to reading the column descriptions from the
   VizieR ReadMe or the VOTable `FIELD` descriptions. Without the numeric comparison, G-B can pass only
   with confidence "medium", which the human must accept explicitly. *(A-04b: the user accepted this on
   2026-10-25/26. G-B at "medium" is acceptable if the anonymous Gaia Archive is down.)*
6. (A-04c) **Sample.** The 1,000 rows are drawn from the G-A1 output (seed 20261025, distinct `Source`
   values), with at least 100 rows having `Solved` = 3, at least 100 with a paramp row but NULL
   allosmod, and at least 10 with no paramp row, so that each NULL pattern of D5 and D7 is compared.
   The VizieR side is the XMatch output itself, not a separate TAP query.

**Remaining to leave Proposed (objective list, A-04b):** G-A1, G-A2, G-A3; G-B1 to G-B4 (or G-B5 at
confidence "medium" if the anonymous Gaia Archive is down); then human approval.
**(A-04c) Remaining, updated:** G-A2 (criterion above; needs the one offset run); G-B1 to G-B4 and
G-B6 (or G-B5 at "medium" if the anonymous Gaia Archive is down, already accepted by the user); then
human approval, which includes the ADR 0003 note of D7.4. G-A1 and G-A3 are passed.

## 8. Reversal triggers

1. The Gaia account works and G-A has not passed yet → switch to ADR 0002 §1 option (i). This ADR is
   superseded and the schema §9 provenance is removed.
2. The Gaia account works after G-A passed → keep O1 for the data already built. New ingests may use
   either source, since the canonical names are equal.
3. G-B finds any mapped column that differs from the Gaia Archive beyond tolerance → stop. That column
   is unmapped until its correct source is found, and if it is the allosmod column, cut (c) is blocked.
4. All contingencies (a) to (d) fail G-A → status becomes Rejected and the project waits on (e).
   *(A-04b: the primary XMatch path and contingencies (b), (d) and (a) all fail G-A.)*
5. R-PM shows that `r_search` truncates matches at high `pm` → raise `r_search`, up to 180 arcsec. If
   that is still not enough, declare the affected `pm` range out of scope in the report.
6. CDS publishes, or applies, a fair-use limit that the plan of 10 to 50 jobs would break → re-plan the
   batching before running. *(A-04b: the plan is now two XMatch passes per batch.)*
7. (A-04b) The paramp pass returns a different set of `Source` values or a different allosmod NULL
   pattern than the Gaia Archive (G-B) → the paramp pass is dropped and allosmod is fetched by
   contingency (b).
8. (A-04c) The D7.4 ablation shows that M-noind loses more than 0.10 macro-F1 against M-full on cut (a)
   with a CI that excludes 0, **and** the loss is concentrated in the `Solved` = 3 stratum → the
   headline LightGBM result is reported with the M-noind number next to it, and the human decides
   whether `features_v2` drops the missingness signal. The 0.10 threshold is reversible.

## Non-goals

- Downloading the SDSS file (needs user approval). *(A-04b: approved and done; no longer a non-goal.)*
- Running any test or query (this ADR defines the gates and does not execute them).
- Choosing `r_match`, `z_star_max`, `z_qso_min` or the DR15 rule (E1, E3, E4).
  *(A-04c: this ADR fixes the criterion of G-A2 for `r_match`, not its value.)*
- DR19/DR20 labels (E6).
- Ingesting the 5-class combmod or specmod (schema OQ9).
- Changing the split, the priors or the metrics (ADR 0003).
  *(A-04c: D7.4 adds report items and proposes a note; it changes none of these.)*
- ~~(A-04b) Writing the new schema version that removes the four unsourced columns (separate task).~~
  *(A-04c: cancelled; no column is removed.)*
- (A-04c) Adding `non_single_star` or any other column of the 225-column table to the schema.
- (A-04c) Any imputation of missing astrometry, or any missingness-indicator column beyond
  `astrometric_params_solved`.
