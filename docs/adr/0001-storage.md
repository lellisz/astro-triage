# ADR 0001 — Phase 1 storage: Parquet (zstd) + DuckDB in-process

- Status: **Accepted** (2026-10-25, human approval). Every Evidence cell is measured and no
  threshold is broken. Re-measure the Parquet size on the real table after ingest.
- Date: 2026-10-25
- Task: T10
- Classification: **reversible** (< 1 day). Every table is a flat file with a fixed schema
  (`docs/specs/schema.md`); loading it into Postgres later is `COPY` per table.

## Context

- Phase 1 classifies Gaia DR3 sources ~~(G < 20)~~ into STAR / GALAXY / QSO with SDSS spectroscopic
  labels: LightGBM, calibration, Gaia DSC baseline. Later phases put an LLM/VLM on the uncertain
  cases (out of scope here).
  *(Amended 2026-11-03, T-DOC-GCUT.)* The Phase 1 population has **no magnitude cut**; the struck
  "(G < 20)" was never applied by any code path (`src/astro_triage/dataset/build.py` lines 104-105
  already state that the XMatch path has no such selection). The population is: SDSS SpecObj DR17
  rows with `SPECPRIMARY == 1`, `ZWARNING == 0`, `CLASS` in {STAR, GALAXY, QSO}, finite
  `PLUG_RA`/`PLUG_DEC` and `|PLUG_DEC| <= 90` (`src/astro_triage/labels/specobj.py:130`), matched to
  Gaia DR3 within `r_match` = 0.75 arcsec through CDS XMatch (`r_match` taken from
  `dataset_report.json`). Per ADR 0002 section 3 (`resolve_ambiguity`, `build.py:614`), SpecObj rows
  that match more than one Gaia source within `r_match` are dropped: 17,611 pairs in full_v0
  (2,061,857 pairs after `r_match`, 2,044,246 final rows). The figures below are measured on the kept
  rows of the features table, not on a raw table. Measured on `data/interim/full_v0` (2,044,246 rows):
  55.08% of rows have G ≥ 20 (STAR 20.35%, GALAXY 87.10%, QSO 59.45%), the maximum G is 22.64, and
  617 rows have a NULL `phot_g_mean_mag` (STAR 367, GALAXY 165, QSO 85). The Decision, the reversal
  triggers, the Alternatives and the Evidence below are unchanged. The benchmark in E2 draws a
  synthetic G in [12, 20]; that range is a test input, not a statement about the population.
- Scale: 1 to 5 million labelled rows (the real COUNT is not measured yet), about 40 columns,
  mostly float32/float64. Rough size: about 200 B/row uncompressed, so 0.4 GB (2 M) to 1 GB (5 M)
  in memory. This is an estimate, not a measurement.
- Machine: Ryzen 5 5600X (12 threads), 15.9 GB RAM, 83 GB free disk, Windows 11, single user.
- Access pattern: write once per ingest (batch), then read many times: full-column scans for
  training, group-by HEALPix level 5 for spatial CV folds and maps, range filters on magnitude or
  source_id. No concurrent writers, no OLTP, no row-level updates.
- Pinned stack (`requirements.txt`): duckdb 2.0.0, pyarrow 26.0.0, pandas 3.1.0. **No Postgres
  driver (psycopg / psycopg2 / asyncpg / SQLAlchemy) is pinned.** Rule for this task: no new libs.
- Raw data never goes into git (`data/`, `*.parquet`, `*.duckdb` are ignored).
- Missing values are common and meaningful (galaxies often have no parallax or proper motion).
  They must stay explicit, with no imputation.
- Known breaking change (R-00 digest, verified locally): **DuckDB 2.0 relations no longer have
  `fetchone()`.** Use `.fetchall()`, `.limit(1).fetchall()`, `.df()` or `.arrow()`.

## Decision

1. Every Phase 1 table is stored as **Parquet files, zstd compression**, under `data/` (layout and
   columns in `docs/specs/schema.md`). Each table is one file. Rows are sorted by `source_id`, which
   also sorts them by HEALPix pixel, because the HEALPix level-12 index is encoded in the high bits
   of `source_id`. Pixels are not hive-partitioned: 12,288 level-5 partitions of about 400 rows
   each would make tiny files.
2. **DuckDB runs in-process (`duckdb.connect()` with no file) and queries the Parquet files with
   `read_parquet(...)`.** There is no persistent `.duckdb` database file, so there is no second
   copy of the data and no state to migrate. Views are created per session from code.
3. On disk, a missing value is **always a Parquet NULL, never an IEEE NaN.** DuckDB treats NaN
   as a value that is greater than every other number, so `parallax > 0` is TRUE for NaN but
   NULL (filtered out) for NULL. pandas reads NULL in float columns as NaN, which is what LightGBM
   expects.
4. Code calls `fetchall()` / `df()` / `arrow()`. **`fetchone()` is banned** in this codebase
   because of the DuckDB 2.0 break. Reviewers check this with a search for `fetchone(`.

### What would change this decision (reversal triggers)

Revisit and consider Postgres (or another server DB) if **any** of these becomes true:

- The measured benchmark (Evidence) breaks a threshold: 2 M-row Parquet > 500 MB, write > 30 s,
  HEALPix aggregation median > 2 s, range filter median > 1 s, or peak RAM of the query process
  > 4 GB.
- The real labelled row count is > 50 M, or the total Parquet size is > 20 GB.
- A feature needs **concurrent writers** or row-level updates (for example a Streamlit app writing
  LLM verdicts while a pipeline writes predictions). DuckDB/Parquet is single-writer.
- A local positional crossmatch needs a spatial index (q3c / pgSphere). ADR 0002 sets the crossmatch
  as our own positional match against SpecObj DR17, with the engine still open (Gaia Archive upload,
  CDS XMatch or local astropy). This trigger fires only if the local-astropy engine is chosen and is
  too slow without a spatial index. The DR13 `best_neighbour` table is a cross-check only.
- The human wants the data served to other machines or users over a network.

## Alternatives

| Option | Real cost | Why it is not chosen |
|---|---|---|
| **A. Parquet + DuckDB (chosen)** | Two pinned libs already installed and smoke-tested. One write path (pyarrow). No server. Single writer. NaN/NULL rule must be enforced by convention plus a check. | — |
| B. Postgres (the user already uses it) | Needs a running service. Needs a new driver lib (none pinned), which breaks this task's no-new-libs constraint. Row-oriented heap: about 2-4x the disk of zstd Parquet for float columns (an estimate, not measured). Bulk load is `COPY` through a driver. Training still has to pull every row into pandas, so the DB adds a hop and no capability. Repo becomes harder to reproduce: anyone who clones it needs a DB server. | **No Postgres was found on this machine** (see Evidence). Its strengths (concurrency, transactions, spatial indexes) are not needed in Phase 1. |
| C. Parquet + pandas/pyarrow only (no DuckDB) | Fewer moving parts. But group-bys and filters are written by hand in pandas and need the whole table in memory. SQL is lost as the shared, reviewable form of queries. | DuckDB is already pinned, adds SQL over the same files, and costs nothing at runtime. |
| D. A persistent `.duckdb` file | DuckDB's own format, possibly faster on repeated queries. A second copy of the data, with format compatibility tied to the DuckDB version (2.0 just shipped breaking API changes). | It adds state with no measured benefit. Revisit only if `read_parquet` scans turn out to be the bottleneck. |

## Consequences

Good:
- Nothing to install or start. The repo reproduces with `pip install -r requirements.txt` plus the
  ingest script.
- Columnar zstd storage stays small, and scans read only the columns they need.
- Deleting a table means deleting a file. Rebuilding it means rerunning one script.

Bad (accepted):
- One writer at a time. Concurrent writes to the same file are unsafe. **The writer must write to
  a temporary file and then rename it over the target** (spec invariant), so a reader never sees a
  half-written file.
- No foreign keys or constraints. Key uniqueness, join coverage and the "no NaN on disk" rule must
  be checked in code and tests (listed in `docs/specs/schema.md`).
- Schema evolution is manual. A new column means a new versioned file (`*_v2.parquet`), not
  `ALTER TABLE`.
- Pinned to DuckDB 2.0 semantics (the `fetchone()` removal). Upgrading DuckDB means checking its
  changelog for more removals.
- pandas reads a nullable integer column with nulls as float64. That is why the features table
  uses float32 only (see spec).

## Evidence

### E1. Postgres on this machine (fact, checked with read-only file globbing)

- `C:\Program Files\PostgreSQL` does not exist.
- No `psql.exe` under `C:\Program Files (x86)`.
- A scoop search under the user profile directory timed out with no result.
- Checked by the orchestrator on 2026-10-25: `psql` is not on PATH (`Get-Command psql` finds
  nothing) and `Get-Service -Name "*postgres*"` returns no service. The Postgres column below is
  therefore "N/A — not installed". Postgres was not installed to fill it.
- No Postgres driver is pinned in `requirements.txt`. Even with a server present, measuring it from
  Python would need a new lib, which this task forbids.

### E2. Micro-benchmark (Parquet zstd + DuckDB) — EXECUTED 2026-10-25

Run by the orchestrator (the architect's Bash is limited to `graphify`, so it only wrote the script).
Data is **synthetic** (random normal values, 30% NULLs in nullable columns), so the Parquet size is
indicative: real catalogue columns have structure and may compress differently. Re-measure on the real
table after ingest (acceptance threshold: file size still ≤ 500 MB per 2 M rows). Nothing touches the
project: the script and its outputs live in the scratchpad. `read_parquet(?)` with a prepared
parameter works in DuckDB 2.0.0.

Scratch dir: a temporary directory outside the repository (`$env:TEMP\astro_bench`)
(below: `$S`).

Commands (PowerShell):

```powershell
$S  = Join-Path $env:TEMP "astro_bench"; New-Item -ItemType Directory -Force -Path $S | Out-Null
$PY = ".venv\Scripts\python.exe"   # run from the repository root
& $PY -I "$S\bench_storage.py" gen   "$S\bench_2m.parquet"
& $PY -I "$S\bench_storage.py" query "$S\bench_2m.parquet"
psql --version; Get-Service *postgres* -ErrorAction SilentlyContinue
```

Each phase runs in its own process, so each reports its own peak working set.

`$S\bench_storage.py` (benchmark tool, not production code; save it verbatim):

```python
import ctypes, ctypes.wintypes as wt, json, os, statistics, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, duckdb

class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

def peak_mb():
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 2**20, 1)

N = 2_000_000
HP5 = 2**35 * 4**7  # source_id // HP5 = HEALPix level 5 (nested)
F32 = ["ra_error", "dec_error", "parallax_error", "parallax_over_error", "pm", "pmra_error",
       "pmdec_error", "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
       "ipd_gof_harmonic_amplitude", "phot_g_mean_mag", "phot_g_mean_flux_over_error",
       "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "phot_bp_rp_excess_factor",
       "classprob_dsc_combmod_quasar", "classprob_dsc_combmod_galaxy", "classprob_dsc_combmod_star",
       "classprob_dsc_allosmod_quasar", "classprob_dsc_allosmod_galaxy", "classprob_dsc_allosmod_star",
       "sdss_angular_distance"]
F64 = ["ra", "dec", "l", "b", "parallax", "pmra", "pmdec"]
NULLABLE = {"parallax", "parallax_error", "parallax_over_error", "pm", "pmra", "pmra_error",
            "pmdec", "pmdec_error", "ruwe", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp",
            "phot_bp_rp_excess_factor", "classprob_dsc_allosmod_quasar",
            "classprob_dsc_allosmod_galaxy", "classprob_dsc_allosmod_star"}

def gen(path):
    rng = np.random.default_rng(42)
    hp = np.sort(rng.integers(0, 12288, N, dtype=np.int64))
    sid = hp * HP5 + rng.integers(0, HP5, N, dtype=np.int64)
    sid.sort()
    cols = {"source_id": pa.array(sid), "healpix_l5": pa.array((sid // HP5).astype(np.int16))}
    null_mask = rng.random(N) < 0.30  # ~30% sources without 5/6-param astrometry or BP/RP
    for name in F64 + F32:
        dt = np.float64 if name in F64 else np.float32
        v = rng.standard_normal(N).astype(dt)
        if name == "phot_g_mean_mag":
            v = rng.uniform(12, 20, N).astype(dt)
        cols[name] = pa.array(v, mask=null_mask if name in NULLABLE else None)
    cols["astrometric_params_solved"] = pa.array(rng.choice([3, 31, 95], N).astype(np.int8))
    cols["ipd_frac_multi_peak"] = pa.array(rng.integers(0, 100, N).astype(np.int8))
    cols["ipd_frac_odd_win"] = pa.array(rng.integers(0, 100, N).astype(np.int8))
    cols["in_qso_candidates"] = pa.array(rng.random(N) < 0.01)
    cols["in_galaxy_candidates"] = pa.array(rng.random(N) < 0.01)
    cols["sdss_bestobjid"] = pa.array(rng.integers(1237645876861272064, 1237680000000000000, N))
    cols["sdss_number_of_neighbours"] = pa.array(rng.integers(1, 4, N).astype(np.int16))
    cols["sdss_number_of_mates"] = pa.array(rng.integers(0, 3, N).astype(np.int16))
    t = pa.table(cols)
    t0 = time.perf_counter()
    tmp = path + ".tmp"
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, path)
    write_s = time.perf_counter() - t0
    print(json.dumps({"phase": "gen", "rows": t.num_rows, "cols": t.num_columns,
                      "arrow_mem_mb": round(t.nbytes / 2**20, 1),
                      "parquet_mb": round(os.path.getsize(path) / 2**20, 1),
                      "write_s": round(write_s, 2), "peak_ws_mb": peak_mb()}))

def timed(con, sql, p, reps=5):
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter(); rows = con.execute(sql, [p]).fetchall()
        runs.append(time.perf_counter() - t0)
    return {"cold_s": round(runs[0], 3), "median_s": round(statistics.median(runs), 3), "rows": len(rows)}

def query(path):
    con = duckdb.connect()
    q_agg = ("SELECT healpix_l5, count(*), avg(phot_g_mean_mag), avg(bp_rp), "
             "count(parallax) FROM read_parquet(?) GROUP BY healpix_l5")
    q_flt = ("SELECT source_id, parallax, bp_rp FROM read_parquet(?) "
             "WHERE phot_g_mean_mag BETWEEN 18 AND 19 AND parallax IS NOT NULL")
    q_sid = ("SELECT count(*) FROM read_parquet(?) WHERE source_id BETWEEN "
             f"{1000 * HP5} AND {1100 * HP5 - 1}")
    nan = con.execute("SELECT count(*) FROM read_parquet(?) WHERE isnan(parallax)", [path]).fetchall()
    print(json.dumps({"phase": "query", "duckdb": duckdb.__version__,
                      "agg_healpix_l5": timed(con, q_agg, path),
                      "filter_gmag_range": timed(con, q_flt, path),
                      "filter_source_id_range": timed(con, q_sid, path),
                      "nan_on_disk": nan[0][0], "peak_ws_mb": peak_mb()}))

if __name__ == "__main__":
    {"gen": gen, "query": query}[sys.argv[1]](sys.argv[2])
```

Results (raw JSON lines from the two runs; machine: Ryzen 5 5600X, 15.9 GB RAM, Windows 11,
CPython 3.12.5, duckdb 2.0.0, pyarrow 26.0.0):

```json
{"phase": "gen", "rows": 2000000, "cols": 41, "arrow_mem_mb": 341.9, "parquet_mb": 296.7, "write_s": 3.0, "peak_ws_mb": 522.3}
{"phase": "query", "duckdb": "2.0.0", "agg_healpix_l5": {"cold_s": 0.071, "median_s": 0.057, "rows": 12288}, "filter_gmag_range": {"cold_s": 0.096, "median_s": 0.102, "rows": 175070}, "filter_source_id_range": {"cold_s": 0.017, "median_s": 0.015, "rows": 1}, "nan_on_disk": 0, "peak_ws_mb": 174.1}
```

| Metric (2 M rows, 41 cols, zstd) | Parquet + DuckDB 2.0.0 | Postgres | Threshold |
|---|---|---|---|
| Parquet file size (MB) | 296.7 | N/A — not installed (E1) | ≤ 500 |
| Arrow in-memory size (MB) | 341.9 | N/A | info |
| Write time, pyarrow → zstd (s) | 3.0 | N/A | ≤ 30 |
| GROUP BY healpix_l5, median of 5 (s) / cold (s) | 0.057 / 0.071 | N/A | ≤ 2 |
| Range filter gmag 18-19 & parallax NOT NULL, median (s) | 0.102 | N/A | ≤ 1 |
| Range filter on source_id (row-group pruning), median (s) | 0.015 | N/A | info |
| Peak working set, gen process (MB) | 522.3 | N/A | info |
| Peak working set, query process (MB) | 174.1 | N/A | ≤ 4096 |
| `isnan(parallax)` rows on disk | 0 | N/A | = 0 |

Every threshold is met with a margin of at least 5x (the tightest is the range filter: 0.102 s
against 1 s). At 5 M rows the same layout extrapolates to roughly 750 MB of Parquet (a linear
extrapolation, not a measurement). There is no Postgres measurement, so the comparison with
Postgres rests on the qualitative arguments in the Alternatives table, not on numbers.
