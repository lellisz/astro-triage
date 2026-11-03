# Spec: Phase 1 data schema

- Task: T10b (second half of T10). Date: 2026-10-25.
- Implements: ADR 0001 (storage), ADR 0002 (labels and circularity), ADR 0003 (split and priors).
  This spec decides no policy. Where an ADR leaves a gap, the gap is listed under "Open questions"
  together with the interim rule this spec uses, which can be reversed.
- Column provenance when there is no Gaia account: ADR 0004 (Proposed), applied in §9. §9 changes
  no column, type, nullability or invariant.
  *(A-04c: this sentence no longer holds; see the A-04c line below for what changed.)*
- Column names are checked against `docs/research/R-00-digest.md`. A name the digest does not confirm
  is marked **TBV** (to be verified against the TAP schema or the source file before ingest).
- Amended 2026-10-26 (task A-04b): SDSS position columns of §3 resolved from the downloaded file
  (OQ7), and §9 updated to the revised ADR 0004. No column, type or nullability changed.
- Amended 2026-10-26 (task A-04c), from ADR 0004 (A-04c) and R-09 "100k batch". No v1 file exists, so
  every change stays in v1 (rule 10). Exact changes:
  1. §0 rule 3: note on structural missing astrometry.
  2. §2 rows 29-30 (`in_qso_candidates`, `in_galaxy_candidates`): Nullable no → **yes**; description
     extended.
  3. §2 new row 41 `ap_row_present` (bool, non-null).
  4. §2 Notes: `non_single_star` is not in v1. §2.1: null rule extended to rows 29-30 and 41.
  5. §4: rows 23-24 (`in_qso_candidates`, `in_galaxy_candidates`) **removed**; the exclusion list and
     the null rule extended. `features` now has 22 columns.
  6. §7: new invariant **R6**; **F3** widened. R2 and R3 are kept and now testable on the ADR 0004
     path (`Solved` is mapped).
  7. §8 OQ8 note. §9.2 rows 6-7, 18, 19, 29-30, 34-36 and new row 41; the A-04b write lock is lifted.
     §9.3 `xmatch_request` note. §9.4 updated.
- (Amended 2026-11-03, T-DOC-GCUT) Owner decision: the Phase 1 population has **no magnitude cut**.
  No code path ever applied G < 20 (`src/astro_triage/dataset/build.py` lines 104-105). **This is a
  documentation correction to v1, not a schema change: `schema_version` stays 1 and no file is
  renamed.** Rule 10 asks for a version bump on a nullability change so that valid files of two
  versions are never confused. That cannot happen here: no data file could ever be valid under the
  old text (the built data has 617 rows with NULL `phot_g_mean_mag` and 55.08% of rows with G ≥ 20,
  so G2 and the old R1 would fail on it), so there is no valid "old v1" file to keep apart. Exact changes:
  1. §2 Population: the G < 20 statement is struck and replaced by the real population.
  2. §2 rows 23-24 (`phot_g_mean_mag`, `phot_g_mean_flux_over_error`): Nullable no → **yes**.
  3. §4 rows 17-18 (same two columns): Nullable no → **yes**.
  4. §7: **R1 retired**; new report check **R7** states the real facts (no cut; NULLs counted, not
     forbidden).
  5. §8 OQ1 resolved. §9.4 note on R1.

## 0. Rules for every table

1. **Format.** Each table is a single Parquet file with zstd compression (ADR 0001 §1). There is no
   hive partitioning and no `.duckdb` file.
2. **Order.** Rows are sorted by `source_id` ascending, and `source_id` is strictly increasing,
   which also means it is unique. The one exception is the pixel map (§5.1), which is sorted by
   `healpix_l5`.
3. **Missing values.** A missing value is a Parquet NULL. NaN and ±inf never appear on disk
   (ADR 0001 §3). A writer that receives NaN, for example from pandas, converts it to NULL before
   writing: `pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)` does this, or
   `pc.if_else(pc.is_nan(c), None, c)` does it per column. The writer then runs check G3. Values
   are never imputed.
   (A-04c) **Structural missing astrometry.** A 2-parameter Gaia solution
   (`astrometric_params_solved = 3`; 40.4% of matched rows in the 100k batch) has no parallax, proper
   motion or RUWE. On those rows `parallax`, `parallax_error`, `parallax_over_error`, `pm`, `pmra`,
   `pmra_error`, `pmdec`, `pmdec_error` and `ruwe` are NULL in `raw` and `features`. They are never
   imputed and never NaN on disk. The share is strongly class-dependent (RUWE NULL: GALAXY 87.9%,
   QSO 18.5%, STAR 6.5%); its use by the model is measured as ADR 0004 D7.4 requires.
4. **Explicit schema.** Each file is written with an explicit `pa.schema(...)` whose fields match
   the tables below exactly: same names, same order, same types, and `nullable=False` where the
   Nullable column says "no". The writer never relies on type inference.
5. **Atomic write.** The writer calls
   `pq.write_table(t.sort_by("source_id"), tmp, compression="zstd")` with
   `tmp = target + ".tmp"` in the **same directory** as the target, and then calls
   `os.replace(tmp, target)`. Because of this a reader never sees a half-written file. If the
   process crashes, the `*.tmp` file is left behind and the next run deletes it. On Windows,
   `os.replace` fails with `PermissionError` while another process holds the target open. In that
   case the writer fails loudly and does not retry in a loop.
6. **Single writer.** No two processes write the same file at the same time (ADR 0001, Consequences).
7. **DuckDB access.** DuckDB runs in-process with `duckdb.connect()` and no file, and reads files
   with `read_parquet(?)` and a bound parameter. Results are taken with `.fetchall()`, `.df()` or
   `.arrow()`. **`fetchone()` is banned** because DuckDB 2.0 removed it (ADR 0001 §4).
8. **Large integer IDs in pandas.** A nullable int64 column that contains NULLs comes back as
   float64 from a default `to_pandas()`/`.df()`, and that loses precision on SDSS object IDs
   (about 1.2e18). Read ID columns with `to_pandas(types_mapper=pd.ArrowDtype)` or keep them in Arrow.
9. **File metadata.** Every file has Parquet key-value metadata with these keys:
   `schema_name`, `schema_version` (int), `created_utc` (ISO 8601), `git_sha`,
   `inputs` (a JSON list of the input file names it was built from), and the table-specific keys
   listed per table. Parameters go here, not into columns that would hold the same value on every row.
10. **Versioning.** File names are `<table>_v<N>.parquet`. Bump N on any change to columns, types,
    nullability or the meaning of a column, or on a change of a parameter recorded in the metadata.
    A file of version N may be overwritten only by a re-run that produces the identical schema and
    parameters. A new column means a new version and never an in-place `ALTER` (ADR 0001).

## 1. Directory layout

```
data/                                   (gitignored, never committed: CONTRACT §8)
  raw/
    gaia_raw_v1.parquet                 table raw (§2)
    specObj-dr17.fits                   SDSS DR17 SpecObj, unmodified download (ADR 0002)
    dr14q.fits                          SDSS DR14Q catalogue, unmodified download (exact file name TBV)
  labels/
    labels_v1.parquet                   table labels (§3)
    labels_v1.counts.json               discard counts, including per-SpecObj counts (§3.4)
  features/
    features_v1.parquet                 table features (§4)
  splits/
    split_map_l5_v1.parquet             pixel -> band -> fold, 12,288 rows (§5.1)
    split_v1.parquet                    table split, one row per source (§5.2)
  predictions/
    predictions_<model_id>_<model_version>.parquet   table predictions (§6), e.g. predictions_lgbm_v1.parquet
```

The `*.counts.json` sidecar is written the same way as the Parquet files (temporary file, then rename).

(A-04b) The downloaded `specObj-dr17.fits` currently sits in `data/external/sdss/`, not `data/raw/`.
Both are gitignored. The ingest task either moves it or records its actual path in `inputs`; the
layout above is not changed by this note.

## 2. Table `raw`: `data/raw/gaia_raw_v1.parquet`

- **Content.** One row per ingested Gaia DR3 source. The columns come from `gaiadr3.gaia_source`,
  plus the DSC allosmod columns from `gaiadr3.astrophysical_parameters` (LEFT JOIN on `source_id`),
  plus the DR13 official crossmatch keys from `gaiadr3.sdssdr13_best_neighbour` (LEFT JOIN). Per
  ADR 0002 §1, the DR13 keys are used **only for the E2 cross-check** and are never a label source.
- **Population.** ~~All sources with `phot_g_mean_mag < 20`, restricted by an ingest region. The
  region is an open question (OQ1).~~
  (Amended 2026-11-03, T-DOC-GCUT) **No magnitude cut.** The population is the Gaia DR3 sources
  matched within `r_match` = 0.75 arcsec, through CDS XMatch, to SDSS SpecObj DR17 rows with
  `SPECPRIMARY == 1`, `ZWARNING == 0`, `CLASS` in {STAR, GALAXY, QSO}, finite
  `PLUG_RA`/`PLUG_DEC` and `|PLUG_DEC| <= 90` (`src/astro_triage/labels/specobj.py:130`)
  (`r_match` taken from `dataset_report.json`). Per ADR 0002 §3, SpecObj rows that match more than
  one Gaia source within `r_match` are dropped (17,611 pairs in full_v0: 2,061,857 pairs after
  `r_match`, 2,044,246 final rows). The full_v0 figures are measured on the kept rows of the
  features table, not on a raw table. No code path ever applied
  G < 20. Measured on `data/interim/full_v0` (2,044,246 rows): 55.08% of rows have G ≥ 20
  (STAR 20.35%, GALAXY 87.10%, QSO 59.45%), the maximum G is 22.64, and 617 rows have a NULL
  `phot_g_mean_mag` (STAR 367, GALAXY 165, QSO 85). OQ1 is resolved by this statement.
- **Types.** The ADR 0001 benchmark types are kept: float64 for positions and for
  parallax/pmra/pmdec, float32 for errors, photometry and probabilities, and small ints for flags.
- **Table metadata keys.** `adql_query` (the exact query text) and `gaia_release` = `"DR3"`.

| # | Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|---|
| 1 | `source_id` | int64 | no | Gaia DR3 source id (primary key) | gaia_source |
| 2 | `ra` | float64 | no | Right ascension, ICRS, epoch 2016.0 (deg) | gaia_source **TBV** |
| 3 | `dec` | float64 | no | Declination, ICRS, epoch 2016.0 (deg) | gaia_source **TBV** |
| 4 | `l` | float64 | no | Galactic longitude (deg) | gaia_source **TBV** |
| 5 | `b` | float64 | no | Galactic latitude (deg) | gaia_source **TBV** |
| 6 | `ra_error` | float32 | no | Uncertainty of ra (mas) | gaia_source **TBV** |
| 7 | `dec_error` | float32 | no | Uncertainty of dec (mas) | gaia_source **TBV** |
| 8 | `parallax` | float64 | yes | Parallax (mas); NULL for 2-parameter solutions | gaia_source |
| 9 | `parallax_error` | float32 | yes | Parallax uncertainty (mas) | gaia_source |
| 10 | `parallax_over_error` | float32 | yes | parallax / parallax_error | gaia_source **TBV** |
| 11 | `pm` | float32 | yes | Total proper motion (mas/yr) | gaia_source |
| 12 | `pmra` | float64 | yes | Proper motion in RA·cos(dec) (mas/yr) | gaia_source |
| 13 | `pmra_error` | float32 | yes | Uncertainty of pmra | gaia_source **TBV** |
| 14 | `pmdec` | float64 | yes | Proper motion in Dec (mas/yr) | gaia_source |
| 15 | `pmdec_error` | float32 | yes | Uncertainty of pmdec | gaia_source **TBV** |
| 16 | `ruwe` | float32 | yes | Renormalised unit weight error | gaia_source |
| 17 | `astrometric_excess_noise` | float32 | no | Excess noise (mas) | gaia_source |
| 18 | `astrometric_excess_noise_sig` | float32 | no | Significance of the excess noise | gaia_source **TBV** |
| 19 | `astrometric_params_solved` | int8 | no | 3, 31 or 95 (2-, 5- or 6-parameter solution) | gaia_source |
| 20 | `ipd_gof_harmonic_amplitude` | float32 | yes | Image-parameter goodness-of-fit amplitude | gaia_source |
| 21 | `ipd_frac_multi_peak` | int8 | yes | % of windows with more than one peak (0-100) | gaia_source |
| 22 | `ipd_frac_odd_win` | int8 | yes | % of windows with odd window flags (0-100) | gaia_source |
| 23 | `phot_g_mean_mag` | float32 | yes (T-DOC-GCUT; was no) | G magnitude ~~(non-null because of the G < 20 selection)~~. (Amended 2026-11-03, T-DOC-GCUT) NULL when Gaia publishes no G photometry; there is no magnitude cut. 617 NULL rows in full_v0 | gaia_source |
| 24 | `phot_g_mean_flux_over_error` | float32 | yes (T-DOC-GCUT; was no) | G flux / error. (Amended 2026-11-03, T-DOC-GCUT) NULL when Gaia publishes no G photometry; 617 NULL rows in full_v0, the same rows as the `phot_g_mean_mag` NULLs (both NULL together: 617) | gaia_source **TBV** |
| 25 | `phot_bp_mean_mag` | float32 | yes | BP magnitude | gaia_source |
| 26 | `phot_rp_mean_mag` | float32 | yes | RP magnitude | gaia_source |
| 27 | `bp_rp` | float32 | yes | BP − RP colour | gaia_source |
| 28 | `phot_bp_rp_excess_factor` | float32 | yes | BP/RP flux excess factor | gaia_source |
| 29 | `in_qso_candidates` | bool | yes (A-04c; was no) | Source is in the DR3 QSO candidate table. (A-04c) NULL on every row of the ADR 0004 XMatch path, where the column is absent. Never a feature (§4) | gaia_source |
| 30 | `in_galaxy_candidates` | bool | yes (A-04c; was no) | Source is in the DR3 galaxy candidate table. (A-04c) NULL on every row of the ADR 0004 XMatch path. Never a feature (§4) | gaia_source |
| 31 | `classprob_dsc_combmod_quasar` | float32 | yes | DSC combmod P(quasar); includes the global prior | gaia_source |
| 32 | `classprob_dsc_combmod_galaxy` | float32 | yes | DSC combmod P(galaxy) | gaia_source |
| 33 | `classprob_dsc_combmod_star` | float32 | yes | DSC combmod P(star) (3 of the 5 combmod classes) | gaia_source |
| 34 | `classprob_dsc_allosmod_quasar` | float32 | yes | DSC allosmod P(quasar), from photometry and astrometry only | astrophysical_parameters (name **TBV**) |
| 35 | `classprob_dsc_allosmod_galaxy` | float32 | yes | DSC allosmod P(galaxy) | astrophysical_parameters (name **TBV**) |
| 36 | `classprob_dsc_allosmod_star` | float32 | yes | DSC allosmod P(star) | astrophysical_parameters (name **TBV**) |
| 37 | `dr13_objid` | int64 | yes | SDSS DR13 best-neighbour objID; NULL if there is no DR13 neighbour | sdssdr13_best_neighbour (column name and type **TBV**; called `sdss_bestobjid` in the benchmark) |
| 38 | `dr13_angular_distance_arcsec` | float32 | yes | Gaia-to-SDSS DR13 separation (arcsec) | sdssdr13_best_neighbour (`angular_distance`, unit **TBV**) |
| 39 | `dr13_number_of_neighbours` | int16 | yes | Number of DR13 neighbours | sdssdr13_best_neighbour **TBV** |
| 40 | `dr13_number_of_mates` | int16 | yes | Number of other Gaia sources sharing that DR13 object | sdssdr13_best_neighbour **TBV** |
| 41 | `ap_row_present` | bool | no | (A-04c) True if the LEFT JOIN to `astrophysical_parameters` (paramp on the ADR 0004 path) found a row for this source. Separates "no row" from "row without allosmod output" for the two-layer DSC coverage (ADR 0004 D5) | derived from the LEFT JOIN (A-04c) |

The benchmark had 41 columns because it also stored `healpix_l5`. Here `healpix_l5` lives in `split`
and is not repeated in `raw` (§5).
(A-04c) With row 41, `raw` v1 has 41 columns again; `healpix_l5` is still not among them.

Notes:
- The digest confirms `source_id` only as the join key of `astrophysical_parameters`. It confirms
  the 3 allosmod classes but not their literal column names.
- The digest does not confirm `ipd_frac_n_win` and the benchmark does not use it, so it is
  **not in v1**. Adding it means v2.
- The "Nullable = no" entries for columns marked TBV are assumptions about the Gaia data model
  (OQ8). The writer fails, and never drops rows, if a non-nullable column contains a NULL, so a
  wrong assumption shows up at ingest.
- (A-04c) `non_single_star` is **not in v1**. It is absent from the ADR 0004 XMatch output. If a later
  version adds it, it is nullable and NULL on every XMatch-path row (ADR 0004 D1).

### 2.1 Null rule for `raw`
A column is NULL exactly when Gaia publishes no value (TAP NULL or masked), or when the LEFT JOIN
finds no row: columns 34-36 for a missing `astrophysical_parameters` row, columns 37-40 for a
missing DR13 neighbour. Columns 37-40 are NULL together or non-NULL together, and so are
columns 34-36.
(A-04c) Columns 34-36 are also NULL when the `astrophysical_parameters` row exists but has no allosmod
output; `ap_row_present` (column 41) tells the two cases apart and is never NULL. Columns 29-30 are
NULL together on every row when the source table lacks them (ADR 0004 XMatch path), and non-NULL
together otherwise. On `astrometric_params_solved = 3` rows, the astrometric columns listed in §0
rule 3 are NULL.

## 3. Table `labels`: `data/labels/labels_v1.parquet`

- **Content.** One row per Gaia `source_id` that has at least one SpecObj DR17 row within `r_match`
  after epoch propagation (ADR 0002 §1). Both kept and discarded sources are stored. The label set
  is the subset with `label_status = 'kept'`. Gaia sources with no SpecObj candidate do not appear here.
- **Table metadata keys.** `r_match_arcsec` (from E1), `z_star_max` and `z_qso_min` (from E3),
  `specobj_file` = `"specObj-dr17.fits"`, `dr14q_file`, `dr15_galaxy_rule` (a text description of
  the plate/mjd approximation, from E4), and `match_engine` (ADR 0002 defers this choice).

| # | Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|---|
| 1 | `source_id` | int64 | no | Gaia DR3 source id (primary key) | raw |
| 2 | `label_status` | dictionary<int8, string> | no | `kept`, or the drop reason (§3.2) | derived (ADR 0002 §2-3) |
| 3 | `n_specobj_candidates` | int16 | no | Number of SpecObj rows within `r_match` of this source (≥ 1) | own crossmatch |
| 4 | `sdss_plate` | int32 | no | Plate of the reference SpecObj (§3.1) | specObj-dr17 (`PLATE`) |
| 5 | `sdss_mjd` | int32 | no | MJD of the reference SpecObj; also used for rule (b).2 | specObj-dr17 (`MJD`) |
| 6 | `sdss_fiberid` | int32 | no | Fiber of the reference SpecObj | specObj-dr17 (`FIBERID`) |
| 7 | `sdss_specobjid` | string | no | SDSS specObjID, stored as a decimal string because it can exceed int64 (type **TBV** in the FITS file) | specObj-dr17 (`SPECOBJID` **TBV**) |
| 8 | `sdss_ra` | float64 | no | SpecObj RA (deg); the fibre/plug position, not a final astrometric position (A-04b) | specObj-dr17 (`PLUG_RA`; the file has no `RA`, A-04b) |
| 9 | `sdss_dec` | float64 | no | SpecObj Dec (deg); the fibre/plug position (A-04b) | specObj-dr17 (`PLUG_DEC`; the file has no `DEC`, A-04b) |
| 10 | `angular_distance_arcsec` | float32 | no | Separation between the Gaia position propagated to the SDSS epoch and the SpecObj position (arcsec) | own crossmatch |
| 11 | `pm_propagated` | bool | no | True if pmra/pmdec were used for the epoch propagation (false if they are NULL in raw) | own crossmatch |
| 12 | `sdss_class` | dictionary<int8, string> | no | SpecObj `CLASS`, trimmed, upper case, verbatim (`STAR`/`GALAXY`/`QSO`, or another value on dropped rows) | specObj-dr17 (`CLASS`) |
| 13 | `sdss_subclass` | string | yes | SpecObj `SUBCLASS`, trimmed; empty string becomes NULL. Metadata only, never a filter | specObj-dr17 (`SUBCLASS`) |
| 14 | `sdss_z` | float32 | no | Redshift | specObj-dr17 (`Z`) |
| 15 | `sdss_zwarning` | int32 | no | `ZWARNING` bitmask | specObj-dr17 (`ZWARNING`) |
| 16 | `in_dr14q` | bool | no | The source matches any DR14Q row within `r_match` (all rows, any ZWARNING); used for rule (b).1 | own match against DR14Q |
| 17 | `galaxy_public_by_dr15` | bool | yes | For `sdss_class = GALAXY`: the spectrum was public by DR15 under the plate/mjd rule. NULL when the class is not GALAXY. Used for rule (b).2 | derived from plate/mjd (hypothesis, E4) |
| 18 | `in_cut_b` | bool | yes | Membership in cut (b) (ADR 0002 §4). NULL unless `label_status = 'kept'` | derived (§3.3) |

### 3.1 Reference SpecObj row and drop reasons

These are the mechanics of ADR 0002 §2-3. The precedence order is this spec's interim rule (OQ6).

- A SpecObj candidate of a source is **eligible** if it passes the quality filter (`ZWARNING == 0`,
  class in {STAR, GALAXY, QSO}, and the z-consistency rule with `z_star_max`/`z_qso_min`) **and**
  it has exactly one Gaia source within `r_match`.
- If the source has ≥ 1 eligible candidate and all of them have the same class, the row gets
  `label_status = 'kept'` and the reference row is the **nearest eligible** candidate.
- If the eligible candidates disagree on class, the row gets `label_status = 'class_conflict'` and
  the reference row is the nearest eligible candidate.
- If there are no eligible candidates, the reference row is the **nearest** candidate, and
  `label_status` is the first check that candidate fails, in this order: `zwarning_nonzero`,
  `class_not_allowed`, `z_inconsistent`, `specobj_multi_gaia`.
- The allowed values of `label_status` are exactly: `kept`, `class_conflict`, `zwarning_nonzero`,
  `class_not_allowed`, `z_inconsistent`, `specobj_multi_gaia`.
- (A-04b) Only SpecObj rows with `SPECPRIMARY == 1` and finite `PLUG_RA`/`PLUG_DEC` are uploaded to the
  crossmatch (ADR 0002 §2, amended), so they are the only possible candidates. Rows failing these two
  conditions are counted in `per_specobj` (§3.4) and never become a `label_status` value.

Columns 4-15 always describe the reference row.

### 3.2 Null rule for `labels`
Only `sdss_subclass`, `galaxy_public_by_dr15` and `in_cut_b` are nullable, and only under the
conditions stated in their rows. Every other column is non-null on every row, kept or dropped.

### 3.3 Cut definitions (expressions; S = kept rows with `fold = 'test'` from `split`)
- `in_cut_b = NOT in_dr14q AND NOT (sdss_class = 'GALAXY' AND galaxy_public_by_dr15)` when
  `label_status = 'kept'`, and NULL otherwise. STAR rows are never removed (ADR 0002 §4 (b).3).
- Cut (a): S. Cut (b): S where `in_cut_b`. Cut (c): S and S ∩ (b). Cut (c) uses the `features`
  table and the allosmod columns of `raw`; it needs no extra column.
- (A-04b) DSC coverage (ADR 0002 §4, amended): each DSC metric uses the rows of the cut where that DSC
  variant's columns are non-null (`raw` 31-33 for combmod, 34-36 for allosmod). This is an evaluation
  filter on existing columns and needs no extra column.
- (A-04c) The allosmod coverage is reported in two layers (ADR 0004 D5): `raw.ap_row_present = false`
  (no paramp row), and `raw.ap_row_present = true` with columns 34-36 NULL (row without allosmod
  output). Both are computed from `raw`; `labels` needs no extra column.

### 3.4 Discard counts: `labels_v1.counts.json`
- The file holds a JSON object with these keys: `per_source` (count per `label_status` × `sdss_class`,
  which must equal a GROUP BY on the table), `per_specobj` (counts of SpecObj rows dropped by each
  quality check and by `specobj_multi_gaia`, per class, including SpecObj rows with no Gaia
  candidate at all, under `no_gaia_match`), and `cut_b_removed` (count per class removed by
  (b).1 and by (b).2).
- ADR 0002 requires these counts. Some of them are per SpecObj, so they cannot be recomputed from
  the table.

## 4. Table `features`: `data/features/features_v1.parquet`

- **Content.** One row per `source_id` in `raw`. Every feature column is **float32**, so pandas
  reads it as float32 with NULL becoming NaN, which LightGBM accepts (ADR 0001, Consequences).
- **Excluded by ADR.** BP/RP spectra (ADR 0002 D8) and every `classprob_dsc_*` column
  (ADR 0002 §4 (c)).
  (A-04c) Also `in_qso_candidates` and `in_galaxy_candidates` (ADR 0004 D7.3): membership in the Gaia
  QSO/galaxy candidate tables is decided by Gaia's own classifiers, including the DSC, so it carries
  DSC output into LightGBM, which cut (c) of ADR 0002 forbids. Excluded on every ingest path.
  Likewise `ap_row_present` (`raw` 41) is an ingest bookkeeping column, not a feature.
- **Excluded by the interim rule (OQ3).** The position columns `ra`, `dec`, `l`, `b`, `ra_error`
  and `dec_error`, and the DR13 crossmatch columns.
- **Table metadata keys.** `raw_file` (for example `gaia_raw_v1.parquet`).

| # | Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|---|
| 1 | `source_id` | int64 | no | Key (the only non-float32 column) | raw |
| 2 | `parallax` | float32 | yes | cast | raw.parallax |
| 3 | `parallax_error` | float32 | yes | copy | raw |
| 4 | `parallax_over_error` | float32 | yes | copy | raw |
| 5 | `pm` | float32 | yes | copy | raw |
| 6 | `pmra` | float32 | yes | cast | raw |
| 7 | `pmra_error` | float32 | yes | copy | raw |
| 8 | `pmdec` | float32 | yes | cast | raw |
| 9 | `pmdec_error` | float32 | yes | copy | raw |
| 10 | `ruwe` | float32 | yes | copy | raw |
| 11 | `astrometric_excess_noise` | float32 | no | copy | raw |
| 12 | `astrometric_excess_noise_sig` | float32 | no | copy | raw |
| 13 | `astrometric_params_solved` | float32 | no | cast from int8 (3/31/95) | raw |
| 14 | `ipd_gof_harmonic_amplitude` | float32 | yes | copy | raw |
| 15 | `ipd_frac_multi_peak` | float32 | yes | cast from int8 | raw |
| 16 | `ipd_frac_odd_win` | float32 | yes | cast from int8 | raw |
| 17 | `phot_g_mean_mag` | float32 | yes (T-DOC-GCUT; was no) | copy | raw |
| 18 | `phot_g_mean_flux_over_error` | float32 | yes (T-DOC-GCUT; was no) | copy | raw |
| 19 | `phot_bp_mean_mag` | float32 | yes | copy | raw |
| 20 | `phot_rp_mean_mag` | float32 | yes | copy | raw |
| 21 | `bp_rp` | float32 | yes | copy | raw |
| 22 | `phot_bp_rp_excess_factor` | float32 | yes | copy | raw |

(A-04c) Rows removed from this table: 23 `in_qso_candidates` (float32, no, "1.0 / 0.0 cast from bool",
raw) and 24 `in_galaxy_candidates` (float32, no, "1.0 / 0.0 cast from bool", raw). Reason above.

**Null rule:** a feature is NULL exactly when its raw source column is NULL. No imputation and no
sentinel values (such as −999). Derived features, such as absolute magnitudes or extra colours,
are not in v1 (OQ3).
(A-04c) On `astrometric_params_solved = 3` rows, features 2-10 are NULL (§0 rule 3). Feature 13
(`astrometric_params_solved`) is a legitimate feature, published for every Gaia source and available
at inference. LightGBM consumes the NULLs natively; no missingness-indicator column is added in v1.
The ablation without features 2-10 and 13, and the strata by feature 13, are evaluation items of
ADR 0004 D7.4; they need no extra column.

## 5. Split (ADR 0003 D3)

### 5.1 Pixel map: `data/splits/split_map_l5_v1.parquet` (sorted by `healpix_l5`)

The map has 12,288 rows, one per HEALPix level-5 nested pixel, whether or not the pixel contains
data (ADR 0003 rule 1).

**Table metadata keys:** `seed` = `20261025`, `band_edges_deg` = `[11.54, 30]`,
`fractions` = `[0.70, 0.15, 0.15]`, `pixel_centre_frame` = `"ICRS->Galactic"`.

| Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|
| `healpix_l5` | int16 | no | Pixel index 0..12287 (nside 32, nested, equatorial ICRS, the Gaia `source_id` convention) | enumerated |
| `lat_band` | dictionary<int8, string> | no | `A` if \|b_centre\| < 11.54, `B` if 11.54 ≤ \|b_centre\| < 30, `C` if \|b_centre\| ≥ 30. b_centre is the Galactic latitude of the pixel centre | astropy-healpix, then a Galactic transform |
| `fold` | dictionary<int8, string> | no | `train` / `val` / `test`: within each band, pixels sorted by index, permuted with `default_rng(20261025)`, then cut 70/15/15 | ADR 0003 rule 2 |

### 5.2 Table `split`: `data/splits/split_v1.parquet`

One row per `source_id` in `raw`. Table metadata keys: `map_file`, `raw_file`.

| Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|
| `source_id` | int64 | no | Key | raw |
| `healpix_l5` | int16 | no | `source_id // (2**35 * 4**7)` | derived (ADR 0003, verified) |
| `lat_band` | dictionary<int8, string> | no | Band of the **pixel centre**, copied from the map (not the source's own b) | map |
| `fold` | dictionary<int8, string> | no | Copied from the map | map |

**Null rule:** no column is nullable.

## 6. Table `predictions`: `data/predictions/predictions_<model_id>_<model_version>.parquet`

- **Content.** One file per trained model run. It contains one row per `source_id` the run scored,
  and the model is fit on train only. The class order is fixed as star, qso, galaxy (ADR 0003).
- **DSC baselines are not stored here.** Evaluation reads them from `raw` columns 31-36, where the
  combmod star score is `1 − p_qso − p_galaxy` (ADR 0003 D4 point 3), so no second copy is kept.
- **Table metadata keys.** `model_id`, `model_version`, `features_file`, `labels_file`,
  `split_file`, `train_class_fractions` (JSON; this is `pi_src` for the prior shift in ADR 0003),
  `calibration` (method name, or `"none"`), and `calibration_fit_fold` = `"val"` (or `"none"`).

| Column | Arrow type | Nullable | Description | Origin |
|---|---|---|---|---|
| `source_id` | int64 | no | Key | features |
| `model_version` | dictionary<int8, string> | no | Constant within the file; equal to the metadata value | training run |
| `p_raw_star` | float32 | no | Uncalibrated model probability | model output |
| `p_raw_qso` | float32 | no | as above | model output |
| `p_raw_galaxy` | float32 | no | as above | model output |
| `p_cal_star` | float32 | yes | Calibrated probability; NULL if and only if `calibration = "none"` | calibrator (fit on val) |
| `p_cal_qso` | float32 | yes | as above | calibrator |
| `p_cal_galaxy` | float32 | yes | as above | calibrator |

**Null rule:** the `p_cal_*` columns are either all NULL in every row (no calibrator) or never NULL.
No other column is nullable. Argmax and prior-shifted outputs are computed at evaluation time and
are not stored.

## 7. Invariants (checks the tester implements)

Use pyarrow for the order and schema checks, and DuckDB with `read_parquet(?)` and `.fetchall()`
for the aggregate checks. Every check below must pass on every file. Each one is a separate test.

### General (every table)
- **G1 Schema.** `pq.read_schema(path)` has the field names, order, types and `nullable` flags of
  this spec, and `metadata[b"schema_version"]` equals the version in the file name.
- **G2 Non-nullable.** For every non-nullable column, `null_count == 0` over all row groups.
- **G3 No NaN or inf.** For every float column, `SELECT count(*) FROM read_parquet(?) WHERE isnan(c) OR isinf(c)` returns 0.
- **G4 Key.** `source_id` is strictly increasing:
  `pc.all(pc.greater(sid[1:], sid[:-1]))` is true. This implies that `source_id` is unique and sorted.
- **G5 Compression.** Every column chunk reports compression `ZSTD` in the Parquet metadata.
- **G6 No leftover temp.** After a successful run, no `*.tmp` file exists in the target directory.
- **G7 fetchone.** A search for `fetchone(` in `src/` and `tests/` finds nothing.

### raw
- ~~**R1.** `phot_g_mean_mag < 20` on every row.~~
  (Amended 2026-11-03, T-DOC-GCUT) **R1 is retired** and its id is not reused. There is no magnitude
  cut, so no magnitude bound is checked; it is replaced by R7.
- **R2.** `astrometric_params_solved IN (3, 31, 95)`.
- **R3.** `parallax IS NULL` if and only if `astrometric_params_solved = 3` (TBV: if real data
  contradicts this, the check is reported and the spec is updated, not silenced).
- **R4.** Columns 34-36 are NULL together, and columns 37-40 are NULL together.
- **R5.** All `classprob_dsc_*` values lie in [0, 1].
- **R6.** (A-04c) `ap_row_present = false` implies columns 34-36 are NULL. Columns 29-30 are NULL
  together.
- **R7.** (Amended 2026-11-03, T-DOC-GCUT) **No magnitude cut; G NULLs are reported, not forbidden.**
  The check never fails on a magnitude value or on a NULL in `phot_g_mean_mag` or
  `phot_g_mean_flux_over_error` (both nullable, §2 rows 23-24); NaN and ±inf stay forbidden by G3.
  It reports, per `sdss_class` (join to `labels` on `source_id`) and in total: the NULL count of
  each of the two columns, the count with both NULL, `max(phot_g_mean_mag)`, and the share with
  `phot_g_mean_mag >= 20`. Reference values measured on `data/interim/full_v0` (2,044,246 rows):
  617 NULL `phot_g_mean_mag` (STAR 367, GALAXY 165, QSO 85), 617 NULL `phot_g_mean_flux_over_error`
  (the same rows; both NULL together: 617), max G 22.64, G ≥ 20 on 55.08%
  (STAR 20.35%, GALAXY 87.10%, QSO 59.45%).
- (A-04c) R2 and R3 are unchanged and apply on the ADR 0004 path, where `astrometric_params_solved`
  comes from `Solved`. The 100k batch is consistent with R3 in the direction "`Solved` = 3 implies no
  parallax"; the reverse direction is checked at first ingest.

### labels
- **L1.** `labels.source_id ⊆ raw.source_id`: an anti-join count of 0.
- **L2.** `label_status` takes only the 6 allowed values. On `kept` rows, `sdss_zwarning = 0` and
  `sdss_class IN ('STAR','GALAXY','QSO')`.
- **L3.** `angular_distance_arcsec <= r_match_arcsec` (from the file metadata), and
  `n_specobj_candidates >= 1`.
- **L4.** `in_cut_b IS NULL` if and only if `label_status <> 'kept'`. On kept rows, `in_cut_b`
  equals the §3.3 expression.
- **L5.** `galaxy_public_by_dr15 IS NULL` if and only if `sdss_class <> 'GALAXY'`.
- **L6.** On kept rows, `sdss_class = 'STAR'` implies `in_cut_b` is true unless `in_dr14q`.
- **L7.** `counts.json.per_source` equals `GROUP BY label_status, sdss_class` on the table.
- **L8.** `pm_propagated` is false wherever `raw.pmra` is NULL (join on `source_id`).

### features
- **F1.** The set of `features.source_id` equals the set of `raw.source_id`: counts are equal and
  the anti-join is 0 in both directions.
- **F2.** Every column except `source_id` is float32.
- **F3.** No column named `classprob_dsc_*` and no BP/RP spectrum column (names `*xp*`,
  `*sampled*`, `*coefficients*`) is present.
  (A-04c) Also no column named `in_qso_candidates`, `in_galaxy_candidates` or `ap_row_present`.
- **F4.** For each column, `features.c IS NULL` if and only if `raw.c IS NULL`, joined on
  `source_id`.

### split
- **S1.** The map has exactly 12,288 rows, `healpix_l5` runs from 0 to 12287 with no gaps, and
  `fold` and `lat_band` are non-null.
- **S2.** Rebuilding the map with the seed from the metadata produces an identical table, so the
  map is deterministic.
- **S3.** Within each band, the pixel counts per fold follow the integer cuts of ADR 0003:
  `train = n*70//100`, `val = n*15//100`, `test = n - train - val`. Consequently train and val are
  within 1 below their share (0.70·n and 0.15·n), and test is between 0 and 2 above 0.15·n (it
  receives both rounding remainders). Amended 2026-10-25 after the T14c finding (band B: 554 test
  pixels against 552.9).
- **S4.** For split: `healpix_l5 = source_id // 34359738368 // 16384` (2**35 = 34359738368;
  4**7 = 16384) on every row.
- **S5.** `split` joined to the map on `healpix_l5` gives identical `lat_band` and `fold`.
- **S6.** The set of `split.source_id` equals the set of `raw.source_id`.
- **S7.** No `healpix_l5` value appears in two folds, which follows from S5. Checked with
  `count(DISTINCT fold) = 1` per pixel.

### predictions
- **P1.** `predictions.source_id ⊆ features.source_id`. Every `labels` row with
  `label_status = 'kept'` and `fold IN ('val','test')` has a prediction.
- **P2.** All probabilities lie in [0, 1]. `|p_raw_star + p_raw_qso + p_raw_galaxy − 1| <= 1e-5`,
  and the same holds for `p_cal_*` when they are non-null.
- **P3.** The `p_cal_*` columns are NULL on every row or on no row. They are all NULL if and only
  if the metadata has `calibration = "none"`.
- **P4.** `model_version` has exactly one distinct value, equal to the metadata value and to the
  file name.

## 8. Open questions (not decided here; each has an interim rule)

1. **OQ1 Raw population.** Which Gaia sources are ingested: the whole G < 20 sky, the SDSS
   footprint only, or only sources that have SpecObj candidates. This is for the ingest task, and
   it sets the row count, which is the ADR 0001 reversal trigger at 50 M. Interim rule: the schema
   does not depend on the answer.
   **RESOLVED 2026-11-03 (owner decision, T-DOC-GCUT):** only Gaia DR3 sources matched to an eligible
   SpecObj DR17 row within `r_match` = 0.75 arcsec, after the ADR 0002 §3 ambiguity drop and the
   `|PLUG_DEC| <= 90` check (the population stated in §2), with **no magnitude cut**. The "whole G < 20 sky" option was never built. Measured row count 2,044,246, far below the
   50 M trigger of ADR 0001.
2. **OQ2 ADR 0001 vs ADR 0002 wording. RESOLVED 2026-10-25.** The orchestrator amended the
   local-crossmatch trigger in ADR 0001 to follow ADR 0002 (own DR17 match, DR13 as a check only).
3. **OQ3 Feature set beyond D8. DECIDED 2026-10-25 (human): position is excluded.** `ra`, `dec`,
   `l`, `b` and the position errors do not enter LightGBM, because the split is spatial and position
   mostly encodes the SDSS footprint (selection bias, requirement 7). Adding them means
   `features_v2`, reported as a variant with and without position.
4. **OQ4 Committed manifest location. DECIDED 2026-10-25 (human):**
   `manifests/split_pixel_map_v1.csv`, tracked in git (12,288 rows: pixel, `lat_band`, `fold`).
   The map is also rebuilt deterministically and checked by S2.
5. **OQ5 Rounding of 70/15/15.** ADR 0003 gives no rounding rule for each band's cut. Interim rule:
   `n_train = floor(0.70·n)`, `n_val = floor(0.15·n)`, and test takes the rest. S3 tolerates ±1.
6. **OQ6 Drop precedence (§3.1).** ADR 0002 lists the drop rules but not their order when a source
   has several candidates with different failures. The order in §3.1 needs confirmation from the
   owner of ADR 0002.
7. **OQ7 SpecObj column names and types.** These are `SPECOBJID` (it may exceed int64, so it is
   stored as a string), `PLUG_RA`/`RA`, and `CLASS`/`SUBCLASS` padding. The DR14Q file name and
   its columns are also unknown. Verify all of them on the downloaded FITS files.
   **Partly resolved 2026-10-26 (A-04b, R-09 "Update"):** the file has `PLUG_RA`, `PLUG_DEC`,
   `SPECPRIMARY`, `SPECOBJID`, `MJD`, `FIBERID`, `CLASS`, `SUBCLASS`, `Z`, `ZWARNING`, and **no**
   `RA`, `DEC` or `SCIENCEPRIMARY`; positions are `PLUG_RA`/`PLUG_DEC`. Still open: the FITS type of
   `SPECOBJID`, `CLASS`/`SUBCLASS` padding, and whether `PLATE` exists (R-09 lists `PLATEID`); the
   DR14Q file.
8. **OQ8 TBV Gaia columns.** `ra`, `dec`, `l`, `b`, `ra_error`, `dec_error`, `parallax_over_error`,
   `pmra_error`, `pmdec_error`, `astrometric_excess_noise_sig`, `phot_g_mean_flux_over_error`, the
   allosmod names, and the names, types and units of the DR13 best_neighbour columns are not in the
   digest. Check them against the TAP schema, and check their nullability against R3/G2 on the
   first ingest.
   (A-04c) On the ADR 0004 path, `ra_error`, `dec_error` (`e_RAdeg`, `e_DEdeg`) and
   `astrometric_excess_noise_sig` (`sepsi`) now have a source; their units and semantics are checked
   in ADR 0004 G-B, and their non-null assumption by G2 at first ingest.
9. **OQ9 5-class combmod.** ADR 0003 open question 3 asks whether the 5-class combmod from
   `astrophysical_parameters` is ingested. v1 includes only the 3 `gaia_source` combmod columns.
10. **OQ10 Unresolved parameters.** `r_match` (E1), `z_star_max` and `z_qso_min` (E3), and the DR15
    galaxy rule (E4) are unresolved. `labels_v1` cannot be produced until they are fixed. If E4
    fails, ADR 0002 trigger 4 applies, and `galaxy_public_by_dr15` and the cut-(b) galaxy rows need
    a v2 definition.
    (A-04c) The objective criterion for `r_match` is ADR 0004 gate G-A2.

## 9. Column provenance under ADR 0004

This section applies only while ADR 0004 (no-login data path) is the active ingest route. Its status
is Proposed, and no file may be built from it before gates G-A and G-B pass. If ADR 0004 is rejected
or superseded, this section is deleted and §2 to §6 apply as written.

### 9.1 Rule
**The canonical name is the Gaia Archive name.** Every column in §2 to §6 keeps its name, type,
nullability and meaning. VizieR names exist only in the ingest adapter and in
`manifests/column_map_vizier_v1.csv`. The adapter renames columns before the explicit schema of
rule 4 is applied, so a missing or misnamed column fails G1. The `Origin` cells of §2 are read
through the table below.

### 9.2 Mapping for `raw`

Sources: `"I/355/gaiadr3"` (g) and `"I/355/paramp"` (p) on the anonymous VizieR TAP, joined on `Source`.
(A-04b) Primary path: one CDS XMatch pass per table, returning these VizieR columns, joined locally on
`Source` (ADR 0004 D4). The full column list is fixed by ADR 0004 gate G-A3.
(A-04c) G-A3 passed on the 100k batch; the table below is its result.

| raw # | Canonical column | VizieR | Status |
|---|---|---|---|
| 1 | `source_id` | g.`Source` | header seen |
| 2-3 | `ra`, `dec` | g.`RA_ICRS`, g.`DE_ICRS` | header seen; epoch checked in G-B. (A-04b) Absent from the XMatch output, which has `RAdeg`, `DEdeg`; their equivalence is checked in G-B. (A-04c) XMatch path: g.`RAdeg`, g.`DEdeg`; G-B confirms they are epoch 2016.0 |
| 4-5 | `l`, `b` | none identified | derived locally from ra/dec (astropy ICRS to Galactic). (A-04c) No `GLON`/`GLAT` in the XMatch output; never NULL |
| 6-7 | `ra_error`, `dec_error` | ~~none identified~~ (A-04c) g.`e_RAdeg`, g.`e_DEdeg` | ~~G-B blocker~~ ~~(A-04b) **to be removed** in a new schema version unless G-A3 finds a source (ADR 0004 D1, user decision)~~ (A-04c) **mapped**, XMatch 100k batch; unit (mas) checked in G-B. Not removed |
| 8-9 | `parallax`, `parallax_error` | g.`Plx`, g.`e_Plx` | header seen |
| 10 | `parallax_over_error` | none identified | derived: `parallax / parallax_error`, NULL if either is NULL. (A-04c) NULL on every `Solved` = 3 row; a zero `e_Plx` gives NULL, never ±inf |
| 11 | `pm` | none identified | derived: `sqrt(pmra² + pmdec²)`, NULL if either is NULL |
| 12-15 | `pmra`, `pmra_error`, `pmdec`, `pmdec_error` | g.`pmRA`, g.`e_pmRA`, g.`pmDE`, g.`e_pmDE` | header seen |
| 16 | `ruwe` | g.`RUWE` | header seen; NULL observed in some rows. (A-04c) NULL share by class in the 100k batch: GALAXY 87.9%, QSO 18.5%, STAR 6.5% |
| 17 | `astrometric_excess_noise` | g.`epsi` | header seen |
| 18 | `astrometric_excess_noise_sig` | ~~none identified~~ (A-04c) g.`sepsi` | ~~G-B blocker~~ ~~(A-04b) **to be removed** in a new schema version unless G-A3 finds a source~~ (A-04c) **mapped**, XMatch 100k batch; semantics checked in G-B. Not removed |
| 19 | `astrometric_params_solved` | ~~none identified~~ (A-04c) g.`Solved` | ~~G-B blocker~~ ~~(A-04b) **to be removed** in a new schema version unless G-A3 finds a source~~ (A-04c) **mapped**, XMatch 100k batch; value 3 on 40.4% of matched rows; R2 checks the encoding. Not removed |
| 20 | `ipd_gof_harmonic_amplitude` | g.`IPDgofha` | header seen |
| 21-22 | `ipd_frac_multi_peak`, `ipd_frac_odd_win` | g.`IPDfmp`, g.`IPDfow` | header seen |
| 23 | `phot_g_mean_mag` | g.`Gmag` | header seen |
| 24 | `phot_g_mean_flux_over_error` | g.`RFG` | header seen |
| 25-26 | `phot_bp_mean_mag`, `phot_rp_mean_mag` | g.`BPmag`, g.`RPmag` | header seen |
| 27 | `bp_rp` | g.`BP-RP` | header seen |
| 28 | `phot_bp_rp_excess_factor` | g.`E(BP/RP)` | header seen; description unread |
| 29-30 | `in_qso_candidates`, `in_galaxy_candidates` | g.`QSO`, g.`Gal` | header seen; source type unverified, cast to bool. (A-04c) **Absent from the XMatch output** (TAP table only): NULL on every XMatch-path row (nullable since A-04c). Not features (§4) |
| 31-33 | `classprob_dsc_combmod_{quasar,galaxy,star}` | g.`PQSO`, g.`PGal`, g.`PSS` | **inference only**; equivalence to the 3 `gaia_source` combmod columns not verified |
| 34-36 | `classprob_dsc_allosmod_{quasar,galaxy,star}` | p.`PQSO-A`, p.`PGal-A`, p.`Pstar-A` | **unverified** (the `-A` = allosmod mapping is convention only). (A-04b) Present in the XMatch paramp output; non-null on 262 of 467 matched rows, so NULL is common and expected (§2.1, R4). (A-04c) 100k batch: non-null on 59.7% of paramp rows |
| 37-40 | `dr13_*` | none identified | interim: NULL on every row; E2 deferred (ADR 0004 D1) |
| 41 | (A-04c) `ap_row_present` | none (local LEFT JOIN) | (A-04c) true if the paramp pass has a row with this `Source`. 100k batch: false for 674 of 52,518 sources (1.3%) |

Notes:
- p.`PQSO` and p.`PGal` (the 5-class combmod) share names with g.`PQSO` and g.`PGal`. Every JOIN
  qualifies and aliases them by table.
- A derived column is allowed only where the Gaia definition is a closed formula over mapped
  columns. Its `Origin` becomes `derived (ADR 0004)`.
- A G-B blocker is never filled with a sentinel. Either G-B finds its VizieR column, or the column is
  removed in a new schema version by a separate decision.
- ~~(A-04b) That decision is made for raw 6, 7, 18 and 19 (ADR 0004 D1). The schema revision is a
  separate task. It touches §2, `features` 12-13, §4 exclusions, R2 and R3 (which use
  `astrometric_params_solved`), and OQ8. Until it lands, no `raw` file may be written under ADR 0004,
  because these non-nullable columns have no source.~~
  (A-04c) Reverted. Raw 6, 7, 18 and 19 are mapped (table above), so no column is removed, no schema
  revision is needed, and the write lock is lifted: `raw` v1 may be written under ADR 0004 once gates
  G-A and G-B pass (the condition at the top of §9). `features` 12-13, R2 and R3 are unchanged.
- (A-04c) No `raw` column is left without a source or a NULL rule: 29-30 and 37-40 are NULL on the
  XMatch path, and 4-5, 10-11 and 41 are derived.
- `labels` and SDSS columns are unchanged: they come from `specObj-dr17.fits` (download pending user
  approval). `features`, `split` and `predictions` read only `raw`, so they are unaffected.
  (A-04b) The download is approved and done; positions are `PLUG_RA`/`PLUG_DEC` (§3 rows 8-9).

### 9.3 Metadata keys added for v1
No v1 file exists yet, so these keys belong to v1 and do not bump the version (rule 10).

- `raw`:
  - `source_service` = `"VizieR/CDS TAP"` ((A-04b) `"CDS XMatch"` when `retrieval_path = xmatch_columns`)
  - `vizier_tables` = `["I/355/gaiadr3","I/355/paramp"]`
  - `column_map` (path and SHA-256 of the map CSV)
  - `retrieval_path` (`upload_sync` | `upload_async` | `source_range` | `xmatch_columns` | `mirror:<name>`)
  - `n_jobs`
  - (A-04b) `xmatch_request`: JSON with the request parameters of each XMatch pass, without the
    uploaded file. Present only when `retrieval_path = xmatch_columns`.
    (A-04c) It includes the upload id column name and its text prefix (`"S"`); ids are always
    uploaded as prefixed text (ADR 0004 D3).

  `adql_query` holds the query template. (A-04b) On the `xmatch_columns` path there is no ADQL, and
  `adql_query` is the empty string.
- `labels`:
  - `match_engine` = `"CDS XMatch"`
  - `xmatch_catalogue` = `"vizier:I/355/gaiadr3"`
  - `r_search_arcsec` (≤ 180, and ≥ `r_match_arcsec`)
  - `sdss_epoch_rule` (the epoch used for propagation; interim rule: `MJD`)

### 9.4 Effect on existing rules and invariants (listed, not changed)
- Rule 3 (missing values): this rule also covers VizieR empty or masked values. They become NULL and
  are never written as 0, an empty string or NaN.
- §2.1 and R4: columns 37-40 are NULL together on every row under the interim rule, so R4 holds.
- §3 `angular_distance_arcsec` and `pm_propagated`: the meaning is unchanged. XMatch returns
  candidates within `r_search`, and the separation is recomputed locally after propagation
  (ADR 0004 D3). L3 and L8 still apply.
- R5 applies to the mapped DSC columns, and G1 and G2 catch mapping failures at write time.
- Not changed: G1-G7, R1-R5, L1-L8, F1-F4, S1-S7 and P1-P4.
  (A-04c) Correction: F3 is widened and R6 is added (§7); the others are unchanged. R2 and R3 apply on
  this path through g.`Solved`. G2 now also covers raw 41 and the mapped raw 6, 7, 18, 19.
  (Amended 2026-11-03, T-DOC-GCUT) R1 is retired and replaced by R7 (§7); G2 no longer covers raw
  23-24 or `features` 17-18, which are nullable.
