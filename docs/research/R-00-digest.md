# Research digest (R-01..R-05)

Date consulted: 2026-10-25. All findings come from researcher subagents using WebFetch/WebSearch
(summarized page text, not raw HTML). Confidence is per claim. Gaps are listed at the end.
This file is a digest, not a primary source: cite the URLs below, not this file.

## R-01 Gaia DR3 tables, DSC, SDSS crossmatch (confidence: medium, ESA servers returned 503)
- `gaiadr3.gaia_source` has: parallax, parallax_error, pm, pmra, pmdec, phot_{g,bp,rp}_mean_mag, bp_rp,
  ruwe, astrometric_excess_noise, ipd_gof_harmonic_amplitude, ipd_frac_multi_peak, ipd_frac_odd_win,
  phot_bp_rp_excess_factor, astrometric_params_solved, in_qso_candidates, in_galaxy_candidates.
  It carries only 3 DSC columns: `classprob_dsc_combmod_{quasar,galaxy,star}`.
- `gaiadr3.astrophysical_parameters` (join on source_id) has 13 `classprob_dsc_*` columns:
  combmod (5 classes, includes global class prior), specmod (5, BP/RP spectra only),
  allosmod (3: quasar/galaxy/star, photometry + astrometry only, no spectra).
- Official Gaia x SDSS crossmatch exists ONLY for SDSS DR13: `gaiadr3.sdssdr13_best_neighbour`
  (128,130,912 rows; no class label), `sdssdr13_neighbourhood`, `sdssdr13_join`.
  Class label (STAR/GALAXY/QSO) must come from SDSS SpecObj. No official DR16/DR17+ crossmatch found.
- DSC training (medium-low, from search summary of ESA docs): quasars ~300k from SDSS-DR14Q
  (Paris+2018, ZWARNING=0); galaxies ~50k from SDSS-DR15 (Aguado+2019); stars ~720k random Gaia
  sources outside those sets (not SDSS-labelled). No training-membership flag found in the archive.
- Citation fix: arXiv:2206.05864 is Apsis I (Creevey+). The DSC paper is Apsis III,
  arXiv:2206.06710 (Delchambre+).
- Primary sources: https://gaia.aip.de/metadata/gaiadr3/ (mirror, not ESA),
  https://arxiv.org/abs/2206.05864, https://arxiv.org/abs/2206.06710

## R-02 Gaia Archive TAP limits and terms (confidence: low-medium)
- Official quota table (cosmos.esa.int/web/gaia-users/archive/collaborate): authenticated = async 120 min,
  unlimited rows, 5 active queries, 50 queued, 20 GB job output, 1 GB user tables; anonymous = 50,000 rows,
  async 1 min. CONFLICT: older sources say anonymous 3,000,000 rows / 30 s sync. Recheck before ingest design.
- ESA recommends bulk download of gaia_source (csv/ecsv) from http://cdn.gea.esac.esa.int/Gaia/ with
  `bulk_download.sh` (FAQ example: 16 parallel jobs). No official healpix-partitioning or backoff guidance found.
- Failed jobs can still consume quota; delete them. Error: "UWS error [DB quota exceeded]".
- `astroquery.gaia` is the recommended Python client. pyvo not mentioned.
- License: CC BY-NC 3.0 IGO. Reference: Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (arXiv:2208.00211).
- NOT obtained: literal acknowledgement and citation text for DR3, and whether Apsis/DSC papers must be cited.

## R-03 SDSS cutouts (confidence: medium)
- Latest public SDSS release is DR20 (2026-07-30). DR17 = 2021-12-06.
- Service: SkyServer `ImgCutout/getjpeg` (params ra, dec, scale arcsec/px default 0.4, width, height, opt).
  Returns JPEG only. No coordinate-based FITS cutout service found.
- Not documented: size limits, rate limit, acceptable-use policy, image-vs-spectrum footprint.
- Image Use Policy: images may be used for any purpose with credit "Sloan Digital Sky Survey".
  No statement about third-party APIs / VLMs.
- Cite Abdurro'uf et al. 2022, ApJS 259, 35 (DR17) and Blanton et al. 2017; acknowledgement text must be
  copied verbatim from https://www.sdss4.org/collaboration/citing-sdss/.

## R-04 Providers layer 2 (confidence: Bedrock IDs high, prices missing)
- Bedrock current: Claude Sonnet 5.5 (`us.anthropic.claude-sonnet-5-5`), Haiku 5.5
  (`us.anthropic.claude-haiku-5-5`). `us.anthropic.claude-sonnet-4-6` is superseded (listed as older).
  Image + text in, text out. Base64 images only, max 5 MB, formats JPEG/PNG/GIF/WebP.
- Image tokens = ceil(w/28) * ceil(h/28), capped (standard 1568 tokens; high-res tier "4.7 and later"
  4784 tokens). Not w*h/750.
- Bedrock per-token prices NOT found (aws.amazon.com/bedrock/pricing not read).
- Gemini: Flash/Flash-Lite models list a free tier. Paid prices per 1M tokens captured from a page summary
  (verify before coding BudgetTracker). Free-tier RPM/RPD/TPM not published; only in AI Studio.
  Docs do not mention 429 `limit: 0`. Free-tier eligibility is per Google Cloud project; restricted
  projects return 403 PERMISSION_DENIED. Input-image token rule for Gemini not found.

## R-05 Ollama (confidence: medium, latency unknown)
- Latest Ollama v0.40.11 (2026-10-24); installed client is 0.34.0.
- Vision models confirmed: qwen3-vl 2b (1.9 GB), 4b (3.3 GB), 8b (6.2 GB); gemma3 4b (3.4 GB), 12b (8.2 GB).
  Classic models (llava, minicpm-v, moondream, llama3.2-vision, qwen2.5vl) not confirmed.
- RX 6600 XT (gfx1032) is not on the official Windows ROCm list. Docs name Vulkan as the fallback for
  RX 6000. HSA_OVERRIDE_GFX_VERSION documented for Linux only. Whether Vulkan works on this card: unknown.
- No published latency for Ryzen 5 5600X or RX 6600 XT.

## Machine (measured locally)
Windows 11, Ryzen 5 5600X (12 threads), 15.9 GB RAM, AMD RX 6600 XT 8 GB, 83 GB free on C:, Python 3.12.5,
no .venv yet, Ollama client 0.34.0 (server not running).

## Round 2 (R-06a/b/c, R-07) and local verification, 2026-10-25
- Gaia acknowledgement (AIP mirror, medium): "This work has made use of data from the European Space Agency
  (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis
  Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been
  provided by national institutions, in particular the institutions participating in the Gaia Multilateral
  Agreement." DR3 DOI format: https://doi.org/10.17876/gaia/dr.3/<table_identifier>.
  No requirement to cite Apsis/DSC was seen on the mirror, but the ESA page (503) was not read.
- Quotas: official table unchanged (authenticated: unlimited rows, 120 min async, 5 active). The current FAQ is
  consistent with 50,000 rows for anonymous users. The 3,000,000-row figure comes from an undated "FAQ old".
  Treat authenticated as unlimited, anonymous as 50,000.
- DSC training also used ~40,000 white dwarfs (Montreal White Dwarf Database). Training-membership flag: not found
  in the parts read (not confirmed absent). Paper: arXiv:2206.06710.
- Class priors used by the Gaia papers (these are the classifiers' priors, not measured sky fractions):
  Bailer-Jones+2019 Sec. 4 (star, quasar, galaxy) = (0.997871, 0.001996, 0.000133).
  Delchambre+2023 Sec. 3.2.2 Table 2: star 0.988728, QSO 0.000989, galaxy 0.000198, WD 0.000198,
  binary 0.009887; for |b| > 11.54 deg: QSO 2.64e-3, galaxy 5.3e-4. No magnitude-dependent prior.
- SDSS labels: `specObj-dr17.fits` is 6.7 GB on the SAS (equals the CAS specObj table; CORRECTION 2026-10-26: it is not primary-only, 688,476 rows have SPECPRIMARY != 1).
  No field or list giving the first release of a spectrum was found; deriving it from plate/mjd is a hypothesis.
  SkyServer SQL row limits, CasJobs/TAP access, DR19/DR20 file names: not confirmed.
  Gaia Archive documents ESDC_CROSSMATCH_POSITIONAL and DISTANCE/CONTAINS joins; user-table upload limits and
  CDS XMatch limits: not confirmed.
- Prices per 1M tokens (Anthropic first-party page; the AWS Bedrock page did not render its Claude table):
  Sonnet 5.5 $2 in / $10 out; Haiku 5.5 $0.10 in / $0.50 out up to 100k prompt tokens, $0.50 / $2.50 above.
  The Anthropic page states regional/multi-region endpoints carry a 10% premium over global; this is not
  confirmed by AWS for `us.` profiles. Derived (low): Sonnet 5.5 $2.20 / $11.00 on `us.`.
- Gemini (page updated 2026-10-22): gemini-2.5-flash-lite $0.10 / $0.40; gemini-2.5-flash $0.30 / $2.50;
  gemini-3.1-flash-lite $0.25 / $1.50. Image input: 258 tokens if both sides <= 384 px, otherwise 258 tokens per
  768x768 tile; the Gemini 3 page describes a crop-unit rule (rounding unspecified). Free-tier RPM/RPD are not on
  the public page (AI Studio only).
- Installed and smoke-tested locally (Windows 11, CPython 3.12.5, binary wheels): numpy 2.5.4, pandas 3.1.0,
  pyarrow 26.0.0, duckdb 2.0.0, lightgbm 4.7.0, scikit-learn 1.9.1, astropy 8.0.2, astropy-healpix 2.0.1,
  astroquery 0.4.11 (pulls pyvo 1.9.1), streamlit 1.66.1, plotly 7.1.0, boto3 1.43.118, pytest 9.1.1, ruff 0.17.2.
  LightGBM trains with NaN inputs under numpy 2.5.4. Breaking change found: DuckDB 2.0 relations no longer have
  `fetchone()`; use `.fetchall()` or `.limit(1).fetchall()`.
- HEALPix level 5 from source_id: `source_id // (2**35 * 4**7)` (range 0..12287).

## Round 3: no-account path (R-08) and first real query (T13), 2026-10-25
- Measured (T13, anonymous async COUNT, 1 attempt): Gaia DR3 sources with `phot_g_mean_mag < 20` =
  1,060,470,828. This is the whole catalogue, not the SDSS-labelled subset.
- CDN (http://cdn.gea.esac.esa.int/Gaia/gdr3/, no account): `gaia_source/GaiaSource_<ini>-<fim>.csv.gz`
  (~3,386 files, inferred from `_MD5SUM.txt`) and
  `Astrophysical_parameters/astrophysical_parameters/AstrophysicalParameters_<ini>-<fim>.csv.gz`
  (same ranges, 1:1 by file name). File ranges are contiguous HEALPix indices up to 786431 (level 8).
  gaia_source total: 1,811,709,771 rows, ~757 GB compressed (search summary, medium confidence).
  Whether `classprob_dsc_*` columns are inside the files was not opened. `bulk_download.sh` is attached to the
  Gaia Archive FAQ (`--parallel 16` supported). The CDN file browser is JavaScript-rendered.
- Not found: a cross-match (SDSS) folder in the CDN; XMatch row/time limits (only 100 MB upload and 2 M matches
  per job, via search summary); a VizieR table for SDSS DR17 (DR16 photometry is `V/154/sdss16`).
- Anonymous TAP quota re-confirmed: 50,000 rows, async timeout 1 min, 100 active queries shared by all anonymous
  users. The 3,000,000-row figure appears on neither current page.
- Arithmetic (not a recommendation): downloading all of gaia_source is ~757 GB of transfer against 83 GB free disk,
  so the no-account route needs stream-filter-delete per file, and a local SDSS crossmatch.

## Open gaps (need a second research round or a human)
1. Gaia ESA docs (credit/citation, DSC page, crossmatch method, xm_flag): retry when gea.esac.esa.int is up.
2. Gaia quota table conflict.
3. Bedrock prices for Sonnet 5.5 / Haiku 5.5; Gemini input-image tokens.
4. Gemini free-tier limits for this account (AI Studio).
5. SDSS rate limit and VLM use: undocumented (contact helpdesk@sdss.org only if the user decides).
6. Direct Gaia DR3 x SDSS (DR17+) SpecObj positional crossmatch: not researched.
