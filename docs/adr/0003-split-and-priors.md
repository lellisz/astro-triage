# ADR 0003: Sky-region split (D3) and two-regime prior protocol (D4)

- Status: Accepted (2026-10-25, human approval). The pixel map is committed at
  `manifests/split_pixel_map_v1.csv` (decision of 2026-10-25, closes schema OQ4).
- Date: 2026-10-25
- Scope: how labelled objects are split into train/val/test and how test metrics are reported.
- Out of scope: label source and circularity with DSC training (ADR 0002), storage layout (ADR 0001).
- Sources: `docs/research/R-00-digest.md` (R-01, round 2). Primary URLs: https://arxiv.org/abs/2206.06710
  (Delchambre+2023, Apsis III / DSC), https://gaia.aip.de/metadata/gaiadr3/. Bailer-Jones+2019 Sec. 4 as cited
  in the digest (no URL captured).

## Context

Neighbouring objects share extinction, Gaia scanning-law systematics, crowding and SDSS plate/targeting
(SDSS plate radius ~1.5 deg). A random per-object split lets the model learn these local signatures and
inflates test scores. Separately, the labelled set (SDSS spectra) is QSO/galaxy-rich, while the Gaia sky is
>99 % stars, so purity on the raw test set says nothing about purity in a real Gaia query.

## D3: split by HEALPix level-5 pixel

### Formula (verified)

Gaia `source_id` stores the level-12 nested HEALPix index in bits 35+ (`source_id >> 35`). Each coarser level
divides by 4, so level 5 = `source_id // (2**35 * 4**7)`, range 0..12287 (nside 32, 12,288 pixels).
Executed 2026-10-25 with `.venv` astropy-healpix 2.0.1 (synthetic `source_id` built per the Gaia encoding):

```
ra,dec 266.41683 -29.00781 p12 118088263 p5 7207 source_id 4057481821135031573
sid>>35 118088263 sid//(2**35*4**7) 7207 match True
random n 100000 all match True
npix5 12288 area deg2 3.357174580844667 res deg 1.8322594196359498
max sid level5 12287
```

The level-5 index from the formula equals `HEALPix(nside=32, order='nested').lonlat_to_healpix` for the
Galactic-centre point and for 100,000 random sky positions.

### Rules

1. Unit of assignment = level-5 nested pixel. All 12,288 pixels are assigned, whether or not they contain
   data, so the map does not depend on the data and stays stable when labels change.
2. Stratification by Galactic latitude of the pixel centre, 3 bands: A `|b| < 11.54 deg` (the Delchambre+2023
   prior boundary), B `11.54 <= |b| < 30`, C `|b| >= 30`. Within each band, pixels sorted by index are permuted
   with `numpy.random.default_rng(20261025)` and cut 70 / 15 / 15 into train / val / test.
   Clarification (2026-10-25, before any model was trained): a **fresh** generator with this seed is created
   for each band, so one band's assignment does not depend on the size or order of the others. Cuts use
   integer arithmetic: `n_train = n*70//100`, `n_val = n*15//100`, test takes the rest.
   Justification: QSO and galaxy priors change ~2.7x across the 11.54 deg boundary (digest, round 2), and the
   SDSS footprint is concentrated at high `|b|`; an unstratified draw can leave the test set with almost no
   low-latitude data by chance, making low-|b| metrics unreportable.
3. No exclusion margin. Justification by scale:
   - Arcsec-scale leakage (duplicates, blends, close pairs): the fraction of pairs at separation s that cross a
     pixel edge is ~ perimeter * s / area = 4 * 1.83 deg * s / 3.36 deg^2; for s = 5 arcsec this is ~0.3 %.
     Negligible.
   - Degree-scale correlation (extinction, SDSS plates, scanning law): a one-pixel buffer does not remove it
     (scanning-law stripes span tens of degrees) and is unaffordable: with 30 % of pixels in val/test, the
     probability that a train pixel has at least one of its 8 neighbours outside train is 1 - 0.7^8 = 94 %,
     so a buffer would delete ~94 % of the training data.
   - Residual degree-scale leakage is measured instead of removed (see trigger T1 below).
4. The pixel-to-split map (12,288 rows: pixel, band, split) is a committed manifest; path per ADR 0001.
5. Reseeding is allowed only once, before any model is trained, and only if a class falls below the reporting
   floor in test. The reason and new seed are recorded as an amendment to this ADR. Reseeding after seeing
   any metric is forbidden.

## D4: two evaluation regimes

Classes are assumed to be {star, qso, galaxy} (SDSS SpecObj classes, R-01). If ADR 0002 changes the label
set, the grid below is re-derived; the rules do not change.

### Regimes

- BALANCED: each class weighted equally, `w_c = (1/K) / n_c`.
- REALISTIC: importance reweighting, `w_c = pi_c / n_c`, evaluated on every point of the prior grid. Raw test
  counts are never reported as "realistic".

### Prior grid (star, qso, galaxy)

These are classifier priors from the Gaia papers, not measured sky fractions (digest, round 2).

| id | source | star | qso | galaxy |
|---|---|---|---|---|
| P0 | uniform (= BALANCED) | 1/3 | 1/3 | 1/3 |
| P1 | Bailer-Jones+2019 Sec. 4 | 0.997871 | 0.001996 | 0.000133 |
| P2 | Delchambre+2023 Tab. 2, all sky; star = star + WD + binary (derived) | 0.998813 | 0.000989 | 0.000198 |
| P3 | Delchambre+2023, `\|b\| > 11.54`; star = 1 - qso - galaxy (derived) | 0.996830 | 0.002640 | 0.000530 |
| S | sensitivity sweep: qso in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x galaxy in {1e-4, 3e-4, 1e-3}; star = complement | | | |

P3 is applied only to test objects in bands B and C; P0..P2 and S to the full test set. Results on S are
plotted as purity vs. qso prior (one line per galaxy prior), not tabulated.

### Metrics

| metric (per class, one-vs-rest) | BALANCED | REALISTIC |
|---|---|---|
| purity (precision) | yes | yes, weighted |
| completeness (recall) | yes | identical to BALANCED for a fixed decision rule (class reweighting cancels within a class); printed once |
| F1 | yes, plus macro-F1 headline | yes, weighted |
| PR-AUC | yes | yes, weighted counts at every threshold |
| confusion matrix | row-normalised (completeness) | weighted; column-normalised (purity) |
| ECE, 15 equal-mass bins | yes, on outputs shifted to P0 | yes, on outputs shifted to the grid prior, weighted |

Prior shift of a posterior to prior `pi'`: `p'(c|x) proportional to p(c|x) * pi'_c / pi_src_c`, renormalised.
`pi_src` = training class fractions for LightGBM (after any class weighting), Delchambre+2023 all-sky prior for
combmod.

### Effect of the combmod embedded prior

`classprob_dsc_combmod_*` already includes a global class prior (R-01; Table 2 of arXiv:2206.06710) dominated
by stars. Consequences:
1. Native outputs at argmax are not comparable: combmod is shrunk towards star (high realistic purity, low QSO
   completeness); a LightGBM trained on the SDSS mix over-predicts QSO/galaxy on a star-dominated sky.
   Reweighting changes the evaluated population, not the scores, so it does not fix this.
2. Therefore two comparisons are reported per grid point: (a) native argmax of each model, labelled "as
   shipped"; (b) prior-matched: both models shifted to the same grid prior, then argmax. Conclusions about which
   model is better use (b). (a) is reported because it is what a user of the Gaia archive gets.
3. `gaia_source` carries only 3 of the 5 combmod classes (R-01). Until ADR 0002 decides otherwise, the 3-class
   combmod star score is `1 - p_qso - p_galaxy` (absorbs WD and binary), matching P2.
4. combmod was trained partly on SDSS QSOs/galaxies (R-01); overlap is a circularity question for ADR 0002 and
   is not handled here.

### Reporting floor

A per-class metric is reported only if, in the evaluated test subset, that class has `n_c >= 100` objects
spread over `>= 20` distinct level-5 pixels. Otherwise the cell shows `n/a (n=..., pixels=...)`.
Rationale: at completeness 0.9, binomial SE at n = 100 is 0.03; fewer than 20 clusters makes the pixel
bootstrap unstable. REALISTIC purity also prints the raw count of contaminating objects from each other class;
if that count is < 10 the cell is flagged `low-count`, and if it is 0 the contamination rate is reported as the
95 % upper bound `3 / n_k` (rule of three) instead of 0.

### Uncertainty

Cluster bootstrap over test pixels: resample level-5 pixels with replacement within each latitude band,
B = 1000, `default_rng(20261025)`. Weights are recomputed on every replicate (n_c changes). 95 % percentile
intervals. Both models use the same replicates, and the CI of the paired difference is reported. A replicate in
which a class with `pi_c > 0` is absent is counted as invalid; if > 1 % of replicates are invalid, the metric is
`n/a`. No per-object bootstrap anywhere.

### SDSS selection bias (documented, not corrected)

SDSS spectroscopic targets come from magnitude limits and colour cuts, so within each class the labelled
objects are not a random sample of Gaia sources. Class reweighting corrects only class proportions; within-class
covariate shift remains. Every REALISTIC table carries the caption "class-prior corrected; within-class
distribution = SDSS spectroscopic targets". The evaluation report includes per-class G-magnitude and `|b|`
histograms of the test set. No covariate-shift correction is attempted.

## Contract for reweight function

```
FILE: src/eval/reweight.py          [create, implemented by the user]
INTENT: per-sample importance weights that map the test class mix onto a target prior
EXPORTS:
  class EmptyClassError(ValueError): ...
  def reweight(y_true: Sequence[str] | numpy.ndarray, prior: Mapping[str, float]) -> numpy.ndarray
DEPENDS: numpy, collections.abc, typing (closed list)
INVARIANTS:
  - output float64, shape (len(y_true),), same order as y_true; input not mutated; deterministic
  - w_i = pi[y_i] / n[y_i]; sum(w) == 1 within 1e-12; sum of w over class c == pi_c
  - prior must sum to 1 within 1e-6 (then renormalised exactly); otherwise ValueError
  - any prior value < 0, NaN or inf -> ValueError
  - label in y_true not present in prior -> ValueError (never silently dropped)
  - class with pi_c > 0 and n_c == 0 -> EmptyClassError naming the class
  - class with pi_c == 0 -> its samples get weight 0 (allowed, present or not)
  - empty y_true -> ValueError
  - BALANCED regime == reweight(y, uniform prior over the label set)
ACCEPTANCE: the test cases below pass under pytest
RISK: float rounding in published priors (handled by the 1e-6 tolerance); bootstrap replicates dropping a rare
  class (handled by EmptyClassError + invalid-replicate rule)
```

Test cases (prose):
1. Basic: `y = [star, star, star, qso]`, prior star 0.5, qso 0.5 -> weights `[1/6, 1/6, 1/6, 1/2]`, sum 1.
2. Empty class: `y = [star, star]`, prior star 0.998, qso 0.002 -> raises `EmptyClassError` mentioning `qso`.
3. Zero prior: `y = [star, galaxy, qso]`, prior star 0.5, qso 0.5, galaxy 0 -> galaxy weight 0, others 0.5 each.
4. Unknown label: `y = [star, wd]`, prior over {star, qso, galaxy} -> `ValueError`.
5. Tolerance: prior summing to 0.9999995 is accepted and renormalised (output sums to 1); prior summing to 0.9
   raises `ValueError`; prior with a negative entry raises `ValueError`.
6. Completeness invariance: on any toy prediction set, weighted completeness per class equals unweighted
   completeness for every prior on the grid.
7. Regime effect (purity): 1000 stars (10 predicted qso) and 1000 qso (950 predicted qso). Unweighted qso purity
   = 950/960 = 0.9896. With prior star 0.998, qso 0.002: purity = 0.0019 / (0.0019 + 0.00998) = 0.1599 (+-1e-4).

## Consequences

- Good: test pixels are unseen sky; numbers reflect a real Gaia query at a stated prior; uncertainty respects
  spatial clustering; model comparison is prior-matched.
- Bad: ~15 % of labelled data idle in test; pixel bootstrap gives wider CIs than per-object bootstrap (honest,
  but looks worse); REALISTIC purity for galaxies may be `n/a` or `low-count` because few contaminants exist;
  the grid multiplies table size (mitigated by plotting S); selection bias remains uncorrected.
- Maintenance: one manifest, one ~40-line function, one bootstrap loop. Reversible in < 1 day except the split
  seed after models are trained (fixed by rule 5).

## What would change my mind

- T1: a val-set check comparing the level-5 split with a level-3 split (nside 8, ~54 deg^2) shows a metric drop
  larger than the bootstrap CI -> degree-scale leakage is material; move to level 3 or add a plate-level group.
- T2: ADR 0002 finds that DSC training objects cannot be excluded and are spatially clustered -> stratify or
  group by SDSS plate instead of latitude.
- T3: band A has < 20 occupied test pixels for any class after the one allowed reseed -> merge bands A and B.
- T4: a published measured class fraction for Gaia at our magnitude range is found -> add it to the grid as the
  primary REALISTIC prior and demote P1..P3 to sensitivity.
- T5: LightGBM prior-shifted ECE > 0.05 in BALANCED -> calibrate (on val) before any REALISTIC comparison.

## Open questions

1. Delchambre+2023 star prior for `|b| > 11.54` is not in the digest; P3 star is derived as the complement.
2. Whether WD/binary should collapse into star (P2, combmod 3-class star) depends on the ADR 0002 label set.
3. The 5-class combmod columns live in `astrophysical_parameters` (R-01); whether ingest fetches them is ADR 0001.
4. SDSS footprint occupancy per latitude band is unknown until data is ingested; the floor may fail in band A.
5. Bailer-Jones+2019 primary URL not captured in the digest.
