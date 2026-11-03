# R-09: no-login data path (VizieR / CDS XMatch / SDSS SAS)

Accessed 2026-10-25 by the researcher agent. Overall confidence: **low to medium**; items marked NOT VERIFIED
need a real test before they are relied on. Column names come from the header of `SELECT TOP 1 *` on the
anonymous VizieR TAP, not from column descriptions (those could not be read: TAP_SCHEMA returned empty and the
VizieR web pages were blocked by an anti-bot page).

## VizieR `"I/355/gaiadr3"` (225 columns, ADQL name needs quotes, join key `Source`)
| Gaia Archive name | VizieR column |
|---|---|
| source_id | `Source` |
| ra, dec | `RA_ICRS`, `DE_ICRS` |
| parallax (+error) | `Plx`, `e_Plx` |
| pmra, pmdec (+errors) | `pmRA`, `e_pmRA`, `pmDE`, `e_pmDE` |
| phot_g/bp/rp_mean_mag | `Gmag`, `BPmag`, `RPmag` |
| phot_*_mean_flux, flux_over_error | `FG`, `FBP`, `FRP`, `RFG`, `RFBP`, `RFRP` |
| bp_rp | `BP-RP` |
| ruwe | `RUWE` |
| astrometric_excess_noise | `epsi` |
| ipd_gof_harmonic_amplitude / phase | `IPDgofha`, `IPDgofhp` |
| ipd_frac_multi_peak / odd_win | `IPDfmp`, `IPDfow` |
| visibility_periods_used | `Nper` |
| phot_bp_rp_excess_factor | `E(BP/RP)` |
| non_single_star | `NSS` |
| in_qso_candidates / in_galaxy_candidates | `QSO`, `Gal` |

Missing from the requested list: `classlabel_espsp` (not in either table). Full diff against `gaia_source`
columns: NOT VERIFIED. Row count: NOT VERIFIED. Built-in crossmatch is SDSS **DR13** only (`SDSS13`).

## VizieR `"I/355/paramp"` (215 columns, join key `Source`)
`classprob_dsc_*` names do **not** exist. Equivalents (mapping by convention, medium-low confidence):
- combmod (5 classes): `PQSO`, `PGal`, `Pstar`, `PWD`, `Pbin`
- specmod (5): `PQSO-S`, `Pgal-S`, `Pstar-S`, `PWD-S`, `Pbin-S`
- allosmod (3, the fair baseline vs LightGBM): `PQSO-A`, `PGal-A`, `Pstar-A`
Must confirm the `-A` suffix really is allosmod before using it as the baseline.

## Limits (as advertised, anonymous vs authenticated not distinguished)
- VizieR TAP `https://tapvizier.cds.unistra.fr/TAPVizieR/tap` (`/sync` tested, `/async` not): MAXREC default 1e9,
  runtime default 12 h, **upload limit 100,000 rows**, ADQL 2.0.
- CDS XMatch `http://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync`: upload 100 MB (~3M rows), max radius **180 arcsec**,
  output truncated at 2M rows, formats votable/json/csv/tsv, sync only, no login mentioned. Fair-use rule: not found.
- SDSS DR17 file `https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits`: 6.7 GB, primaries only, no login.
  No smaller official summary found. SkyServer SQL works without login; DR17 row limit NOT VERIFIED (DR9 was 500,000 rows,
  600 s, 60 queries/min). CasJobs needs registration (training knowledge, not verified).
- ARI Heidelberg TAP: output default 100k / hard 10M, upload 10M rows; AIP TAP: output 5M, upload 10 MB. Neither tested
  anonymously; DR3/DSC presence NOT VERIFIED.

## Proposed route (not yet tested end to end)
1. SDSS labels from `specObj-dr17.fits` (or SkyServer paged CSV), columns extracted locally.
2. CDS XMatch in batches of 1 to 2M rows (id, ra, dec) against `vizier:I/355/gaiadr3`, radius <= 180".
3. VizieR TAP with upload of `Source` ids (100k rows per job, so 10 to 50 jobs) to fetch features and DSC.
4. Versioned column-name mapping table under `manifests/`.

## Risks specific to this route
- Positional match without proper-motion epoch propagation: high proper-motion stars may fail to match, which biases
  the sample against nearby stars. Not measured; must be reported in the selection-bias section.
- 10 to 50 upload jobs instead of one; anonymous limits may differ from advertised ones.
- Sample size with label + Gaia counterpart is unknown until the match is run.

## Anonymous smoke test run by the orchestrator, 2026-10-25 (sync endpoint, small queries only)
- `SELECT COUNT(*) ... WHERE Gmag < 20` on `"I/355/gaiadr3"`: **timed out at 120 s** (a full scan of ~1.8e9 rows over
  sync). Do not count over the whole table; use async or a narrow filter.
- `JOIN "I/355/gaiadr3" g JOIN "I/355/paramp" p ON g."Source" = p."Source"` with `Gmag < 15`, TOP 5: **worked**.
  Columns `RUWE`, `epsi`, `PQSO-A`, `PGal-A`, `Pstar-A` exist and return values (bright stars: `Pstar-A` = 1.0).
  `RUWE` came back empty for some sources (NULL), so missing values must be handled explicitly.
- TAP_UPLOAD: the server accepts the request but requires the uploaded table to be a **VOTable** (a CSV was rejected
  with "not a valid VOTable"). Two attempts with a valid VOTable (3-table join, then gaiadr3 only) **timed out
  (curl exit 28, 90 to 120 s)**, so upload + JOIN is **still unverified**. Next try: `/async`, or filter by a
  key range instead of an upload.

### Update (orchestrator, 2026-10-25, later)
- **Quoting bug in my earlier tests:** PowerShell 5.1 strips embedded double quotes when passing arguments to
  `curl.exe`, so `"I/355/gaiadr3"` reached the server unquoted. Always send the ADQL from a file
  (`--data-urlencode "QUERY@file"` or `-F "QUERY=<file"`). Any ingest code must pass the query through a
  library or a file, never through a shell argument on Windows.
- With correct quoting: a plain `SELECT TOP 200 "Source" ... WHERE "Gmag" < 9` returns in about 3 s on `/sync`.
- Upload (VOTable, 200 ids) joined to `"I/355/gaiadr3"`: `/sync` timed out at 100 s; `/async` stayed `EXECUTING`
  for 180 s and was aborted. So **upload+JOIN is slow or unusable at the moment**, even for 200 ids. Next in the ADR 0004
  contingency: range queries on `Source` without an upload, or the XMatch returning columns.
- **`specObj-dr17.fits` downloaded and verified:** 6,741,023,040 bytes (matches Content-Length), 5,801,200 rows,
  133 columns, in `data/external/sdss/` (gitignored). Present: `PLATEID, MJD, FIBERID, SPECOBJID, CLASS, SUBCLASS, Z,
  Z_ERR, ZWARNING, PLUG_RA, PLUG_DEC, SPECPRIMARY, PRIMTARGET, SURVEY, PROGRAMNAME`. **Absent:** `RA`, `DEC`,
  `SCIENCEPRIMARY`. ADR 0002 must use `PLUG_RA/PLUG_DEC` (or check for other position columns) and `SPECPRIMARY`.

### XMatch smoke test (orchestrator, 2026-10-26 UTC): WORKS over **https**; upload+JOIN no longer needed
- `http://` endpoint: connection reset after 20 s (curl exit 56). `https://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync`
  answered 200 in 6 s for 1,000 positions. Params used: `request=xmatch`, `distMaxArcsec=5`, `selection=best`,
  `RESPONSEFORMAT=csv`, `cat2=vizier:I/355/gaiadr3`, `colRA1=ra`, `colDec1=dec`, `cat1=@file.csv`.
- SDSS eligible spectra in the local file (SPECPRIMARY==1, ZWARNING==0, CLASS in STAR/GALAXY/QSO, finite PLUG_RA/DEC):
  **4,283,507** (STAR 890,744; GALAXY 2,639,535; QSO 753,228).
- Random sample of 1,000 (seed 20261025): **467 matched within 5 arcsec (46.7%)**; by class STAR 84.2%, QSO 59.9%,
  GALAXY 31.1%. Angular distance: median 0.080", p95 0.438", max 4.619" (a 5" radius admits likely false matches;
  r_match is experiment E1). No duplicate `sid` with `selection=best`. Sampling error of these rates is a few points.
  Projection (rough, not a measurement): about 2.0M labelled sources with a Gaia counterpart, before the G < 20 cut.
- The class mix of the matched sample differs strongly from the SDSS mix (match rate depends on class). This is a
  selection effect to document, and it is the reason the realistic-prior regime reweights.
- XMatch against `vizier:I/355/gaiadr3` returned **142 columns** (not 225) and adds `angDist`. Present among the needed
  ones: `Source, Plx, e_Plx, pmRA, pmDE, Gmag, BPmag, RPmag, FG, FBP, FRP, RFG, RUWE, epsi, IPDgofha, Nper, E(BP/RP)`.
  **Absent:** `RA_ICRS`, `DE_ICRS` (the output has `RAdeg`, `DEdeg`) and `NSS`. Need a full comparison of the 142 returned
  columns against the schema list.
- XMatch against `vizier:I/355/paramp` returned 220 columns with `Source`, the same 467 rows, and the DSC columns
  `PQSO, PGal, Pstar, PWD, Pbin, PQSO-S, Pgal-S, Pstar-S, PWD-S, Pbin-S, PQSO-A, PGal-A, Pstar-A`. **Only 262 of 467
  rows (56%) have non-null allosmod probabilities**: the DSC was not run for every source. The DSC baseline must be
  evaluated on the subset where it exists, and the coverage reported (a missing DSC output is not "STAR").
- Output size: about 780 bytes per matched row in CSV (362 KB for 467 rows).

### 100k batch (orchestrator + T18, 2026-10-26): gate G-A1 passed; G-A3 findings
Run: 100,000 random SDSS candidates (seed 20261025), radius 5", `selection=all`, both catalogues, HTTP 200 in 65 s, nothing
near the 2M-row cap. Outputs in `data/interim/xmatch/` (gitignored).
- **Bug found and fixed (ours, not the service's):** the XMatch parses an all-digit upload column as double, so 19-digit
  SPECOBJIDs came back rounded. The script now sends `sid` as `"S" + specobjid`. Re-run verified: all 52,524 returned
  `sid` exist in the candidates, class agrees, echoed ra/dec equal the upload (max diff 2e-6 arcsec).
- Matched objects: **49,724 of 100,000 (49.7%)**. By class: STAR 19,088/20,964 = **91.1%**, QSO 11,473/17,471 = **65.7%**,
  GALAXY 19,163/61,565 = **31.1%**. Pairs per object (all pairs within 5"): median 1, p95 2, max 15.
  Nearest-pair angDist: median 0.0817", p95 0.4756", max 5.0" (the radius cap).
- gaiadr3 pass: 52,524 rows, 142 columns. paramp pass: 51,850 rows, 220 columns. paramp lacks **674 of 52,518** Gaia
  sources (1.3%, fainter: median G 20.02). paramp rows with `PQSO-A` non-null: 59.7% of its rows.
- **Missing astrometry is a large, class-dependent pattern:** `Solved` = 3 (2-parameter solution: no parallax, no proper
  motion, no RUWE) for 40.4% of matched rows. RUWE null share by class: GALAXY 87.9%, QSO 18.5%, STAR 6.5%. Missingness is
  informative (and legitimate at inference time) but must be explicit, documented, and tested for leakage, not imputed.
- **The four "no source" columns DO exist** in the XMatch output: `astrometric_params_solved` = `Solved`,
  `astrometric_excess_noise_sig` = `sepsi`, `ra_error` = `e_RAdeg`, `dec_error` = `e_DEdeg`. The column report missed them
  because the section 9 mapping was incomplete. **No column needs to be removed.** `l`, `b` (no `GLON/GLAT` in the XMatch
  output) and `parallax_over_error` are derived locally. Still not found in the 142 XMatch columns: `QSO`, `Gal`
  (in_qso_candidates, in_galaxy_candidates) and `NSS` (non_single_star); they exist in the full 225-column TAP table.
  The XMatch returns 142 of 225 gaiadr3 columns.
- 6 Gaia sources are matched by two different SDSS objects (one source, two spectra at different plug positions); ADR 0002's
  one-row-per-source rule applies.

### E1 / G-A2 result (T19, verified by the orchestrator, 2026-10-26)
One extra XMatch request (same 100,000 positions shifted +180" in Dec, gaiadr3 pass, radius 5", all pairs). Every cell of
the table below was recomputed independently from the Parquet files and matches.
- Verdict by the fixed ADR 0004 criterion: **r_match = 0.75 arcsec, accepted** (r_c: STAR 0.75, GALAXY 0.25, QSO 0.5;
  false-match fraction at 0.75": STAR 0.114%, GALAXY 0.208%, QSO 0.062%, all <= 1%). No 10" rerun needed.
- **The criterion is weak here.** `N_true(r) = N_real(r) - N_false(r)` is not monotone: it peaks (STAR at 2.0": 18,791;
  GALAXY about 1.0": 18,233; QSO at 0.25 to 0.5") and then falls, because false matches grow about as r^2. N_true(5")
  therefore sits below the peak, so the 0.995 x N_true(5") threshold is met at tiny radii (GALAXY r_c = 0.25). The result
  0.75" is driven by STAR, not by a real plateau. Also, N_true by subtraction slightly undercounts: an object with both a
  true and a random neighbour is subtracted once.
- STAR has a real tail of true matches at 1 to 2": nearest-pair angDist quantiles for STAR are p90 0.40", p99 1.75",
  p99.9 4.5" (GALAXY p99 4.5" is mostly chance matches; QSO p99 0.42"). Going from 0.75" to 2.0" would recover about
  398 STAR objects (2.1% of matched STAR) at a false fraction of 0.79% for STAR but 1.44% for GALAXY at 2".
  These extra stars are plausibly high-proper-motion stars (SDSS epoch vs Gaia epoch 2016), so **r_match = 0.75" biases
  the sample against nearby fast stars**; this belongs in the selection-bias section together with the R-PM measurement.
- N_false(5")/N_real(5") is 4.5% (STAR), 8.8% (GALAXY), 4.4% (QSO): the 5" lot used in the 100k batch is contaminated; always
  filter by `angDist <= r_match` locally, never use the raw 5" output as the label join.
- Not produced (not in the T19 spec): multi-pair rate m_c(r), the 0.1" grid, local epoch propagation of both runs.
- `manifests/column_map_vizier_v1.csv` written: 41 rows (30 mapped, 5 derived, 6 absent_null). The `column_report.json`
  status is stale for the four columns now mapped in the schema; the CSV is the source of truth.
- The label must not depend on class when choosing the radius: `r_match` is one value for all classes (max over classes),
  because the class is the thing being predicted.

## Gaps to close before relying on this
JOIN with TAP_UPLOAD on VizieR; COUNT on `I/355/gaiadr3`; SkyServer DR17 limits and `specObj` schema; column descriptions
(`Pstar`, `-S`, `-A`, `E(BP/RP)`); anonymous upload on ARI/AIP; ASDC.
