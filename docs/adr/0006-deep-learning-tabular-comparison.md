# ADR 0006: Deep learning vs LightGBM on the same tabular signal (comparison protocol, pre-registered)

- Status: **Proposed** (draft, 2026-11-02). Becomes Accepted only after the remaining open gate H-D2
  (section 12) is answered, P-local and P-gpu (section 9.4) are reported, the tests of section 15 pass, the
  `researcher` check T3 is done (done 2026-11-03, amendment A) and the reviewer verdict has no
  BLOCKER/MAJOR (section 16). The constants
  of H-D8 are frozen from 2026-11-03; any change needs a new ADR or a dated amendment decided before any DL
  result exists.
- Date: 2026-11-02; revised 2026-11-03 (owner decisions of 2026-11-02/03: metric, gain rule, seeds, compute;
  H-D4, H-D7 and H-D8 answered by the owner on 2026-11-03)
- Task: T-DL-01; revision T-DL-02
- Related: ADR 0001 (Parquet, single writer, temp+rename), ADR 0002 (labels, cut (c) "same information"),
  ADR 0003 (pixel split, BALANCED/REALISTIC regimes, paired pixel bootstrap, reporting floor, ECE),
  ADR 0004 D7 (missing astrometry is NULL, never imputed in storage; M-full / M-noind ablation),
  ADR 0005 (pre-registered negative-result rule D7.8, used as a template only; it was never executed: Layer 2
  was closed without test evaluation, ADR 0005 Amendment 2, section 18; Kaggle/NVIDIA quota discipline).
- Out of scope: images/CNN (mentioned only, section 13), BP/RP XP spectra (owner, 2026-11-03, H-D1),
  hyperparameter search (B0, section 6), any change to the ADR 0003 split, floor or seed (this ADR adds its
  own primary metric and a block bootstrap on top of ADR 0003, section 4; ADR 0003's outputs are unchanged),
  any new dependency other than `torch`, Layer 2 / LLM, deployment.
- No DL metric from the **test** fold exists or is used. The only test-fold numbers quoted are Phase 1
  figures already published in `data/interim/full_v0/eval/results.md` (`lgbm_full` contaminant counts,
  test class and pixel counts) and figures derived from them, in: section 2.5 (row M2), section 2.6 (row U3,
  "34 blocks"), section 4.2 (power note, "about 34 blocks"), section 4.4 ("about 5,000", which uses the test
  `n_GALAXY / n_STAR`, and the `MIN_BLOCKS` note), section 4.5, section 14 ("likely reportable"), R14 and
  R16. They are feasibility checks only and set no constant. Section 1 marks the provenance of every number.

## 1. Context

- Research question (IC axis): on the same Gaia DR3 objects, the same SDSS labels (STAR/GALAXY/QSO), the
  same split and the same 21 (M-full) / 11 (M-noind) features, does a deep tabular model beat the existing
  LightGBM, and does a combination of both beat either (answered descriptively, no verdict; section 4.3
  H3)? A negative result is a valid outcome.
- What exists (read from code, 2026-11-02):
  - `src/astro_triage/models/lgbm.py`: one fixed hand-set configuration (`PARAMS`: lr 0.05, 31 leaves,
    `min_child_samples` 40, feature/bagging fraction 0.8, `lambda_l2` 1.0, seed 20261025), `MAX_TREES`
    2000, early stopping on **val** multiclass logloss with `PATIENCE` 50, no class weights, one temperature
    fitted on val logits (`fit_temperature`, bounds 0.05..20), predictions for val+test written by
    `predictions_table` to `predictions_<model_id>_v0.parquet` (columns `p_raw_*`, `p_cal_*`, mapped by
    class **name**; parquet metadata carries `train_class_fractions`).
  - **LightGBM tuning budget = zero search trials**: one configuration, one fit per feature set, the only
    val-driven choices are the stopping iteration and T. Whether the `PARAMS` values were themselves picked
    after earlier manual trials is **not measured** (no record in the repository). On full_v0 the stopping
    iterations were 1638 (`lgbm_full`) and 1684 (`lgbm_noind`) (`data/interim/full_v0/train_report.json`,
    val-driven). LightGBM wall-clock training time: **not measured** (not in the report).
  - The seed enters `lgbm.py` only through the module constant `SEED` (`PARAMS["seed"]`, line 95);
    `run_model(spec, folds, outdir)` (line 429) takes no seed argument (read 2026-11-03).
  - `src/astro_triage/evaluation/runner.py`: Phase 1 evaluator. `LGBM_MODELS` is hard-coded to
    `("lgbm_full", "lgbm_noind")`; `load_test_set` aligns everything on `source_id`, checks the committed
    pixel map, reads `p_cal_*` by name; `run_boot` draws pixel replicates within latitude bands with a fresh
    `default_rng(20261025)` per task, so two tasks with the same test rows in the same order get the
    **same replicates** (paired). Floor = `eval.metrics.floor_status` (>= 100 objects in >= 20 pixels).
    A Layer 1 risk-coverage curve exists only in `evaluation/layer2.py` (`risk_coverage`, tied to
    `Layer2Inputs`).
  - `eval.metrics.bootstrap_pixel_draw(pixel, lat_band, rng)` (line 331) resamples whatever ids it is given
    as `pixel` within each band; passing a coarser block id gives a block bootstrap with no new draw code.
  - Phase 1 Kaggle runner (`kaggle/run_kaggle.py`) is CPU, installs `requirements.txt` with pip, and writes
    one `astro_triage_outputs.zip` that contains `full_v0/labels.parquet` and `full_v0/features.parquet`.
- full_v0 sizes (counts only, from `train_report.json`): train 1,433,262; val 318,428. Train class fractions
  STAR 0.380, GALAXY 0.378, QSO 0.242 (= `pi_src`, no weights).
- Facts measured on full_v0 (2,044,246 objects, **covariates only, all folds**, supplied by the
  orchestrator; no model output involved):
  - only **13.3 %** have G < 17.65; by class: STAR 34.3 %, GALAXY 0.1 %, QSO 0.9 % (reviewer quantiles agree;
    the exact share was not recomputed, backlog T-RES-XP);
  - Gaia DR3 XP (BP/RP) coverage, from researcher T-RES-XP (source: De Angeli et al. 2023, arXiv:2206.06143;
    confidence: **partly read**): XP mean spectra are published for ~219 million sources; the main rule is
    G < 17.65 **and** > 15 CCD transits (so G < 17.65 is necessary, not sufficient; the transit count on
    full_v0 is not measured); fainter sources enter only through named lists (~100K quasars, ~17K galaxies,
    ~100K white-dwarf candidates, ~19K ultra-cool dwarfs); availability is flagged by `has_xp_continuous`.
    The XP fraction among faint quasars of full_v0 is **not measured** (`has_xp_continuous` was not
    downloaded). XP is out of scope (owner, 2026-11-03, H-D1); these facts are kept only to justify that.
- Fact supplied by the orchestrator: 82 % of LightGBM M-full errors are STAR<->QSO. **Fold of origin not
  stated to the architect.** It is used here only as motivation; the descriptive metric `e_SQ` of section
  4.2 is justified independently by the physical limit already recorded in ADR 0005 section 1 (compact
  STAR/QSO overlap in photometry/astrometry), so it would be the same without this number.
- **Population (resolved 2026-11-03, H-D9 closed)**: the former "G < 20" text of ADR 0001 was a
  documentation error; no code path ever applied it, and the dataset is not rebuilt. ADR 0001 Context and
  `docs/specs/schema.md` (R1 retired, R7 added) were amended on 2026-11-03: no magnitude cut, null-G rows
  kept (617 NULL `phot_g_mean_mag`). The stale "PROVISIONAL: 100k sample" header of
  `data/interim/full_v0/eval/results.md` / `results.json` is old runner text; the numbers are full_v0's
  (header fixed per the orchestrator, 2026-11-03; no bootstrap rerun).
- **Known limitation, not corrected (owner, 2026-11-03, GCUT-RC3)**: the ADR 0002 section 2
  redshift-consistency rule (`z_star_max` / `z_qso_min`) is **not implemented** in `src/` and full_v0 labels
  did not pass it. The owner keeps full_v0 as built and does not implement the rule now. This protocol
  applies to full_v0 as built; if the rule is implemented later, trigger T7 (section 11) fires.
- Prior expectation (physics, ADR 0002 cut (c)): a different learner on the **same 21 numbers** has no new
  information; published tabular benchmarks (Gorishniy et al., 2021, "Revisiting Deep Learning Models for
  Tabular Data") found no universal winner between GBDT and FT-Transformer. The expected outcome on the
  tabular axis is a tie.

## 2. Options

### 2.1 What enters as the deep model

| # | Option | Real cost |
|---|---|---|
| A1 | **MLP + FT-Transformer on the 21/11 tabular features, PyTorch only** | Two `nn.Module`s (~150 lines), one training loop; GPU needed for FT-Transformer at 1.4M rows (time not measured). Answers "learner effect" with no new signal. |
| A2 | A1 plus `rtdl` / `pytorch-tabular` / `skorch` | New pinned dependencies for code that fits in ~150 lines; hides the training loop that must be checkpointed per epoch. Rejected by the "no dependency other than torch" constraint. |
| A3 | Network over BP/RP XP coefficients as the main axis | New signal (spectra) mixed with a new learner, so the effect cannot be attributed; XP exists for G < 17.65 with > 15 transits (13.3 % of full_v0 objects, 0.1 % of galaxies have G < 17.65) plus named faint lists; faint-QSO XP fraction not measured; needs a new data-acquisition path (`xp_continuous_mean_spectrum`, access path not verified). **Rejected** (owner, 2026-11-03, H-D1: XP out of scope). |

### 2.2 How the deep model reaches the evaluator

| # | Option | Real cost |
|---|---|---|
| E1 | Modify `evaluation/runner.py` to accept extra model ids | Touches the Phase 1 evaluator whose outputs are already reported; every Phase 1 test must be re-run; couples the IC axis to Phase 1 deletion. |
| E2 | **New evaluator module that reuses runner's loaders and metric primitives, reads DL predictions with the same parquet schema** | ~300 lines; Phase 1 untouched; deleting the DL axis deletes two directories and the modules of section 15. Duplicates the small bootstrap loop (needed anyway for purity at fixed completeness and the block bootstrap). |
| E3 | Write DL predictions under the LightGBM file names | Zero code; silently overwrites/aliases Phase 1 artefacts. Rejected (data corruption risk). |

### 2.3 Hyperparameter budget

| # | Option | Real cost |
|---|---|---|
| B0 | **Equal to LightGBM: zero search.** One published/fixed configuration per architecture; val used only for early stopping and T | Cheapest, fair to the existing LightGBM; a DL loss could be blamed on "untuned". **Decided by the owner, 2026-11-03 (H-D3).** |
| B1 | Equal N-trial random search on val for both families (LightGBM retrained with the same N) | N x GPU time (not measured) plus N LightGBM fits; needs a search harness; changes the reference model. Kept as the reversal option (T8). |
| B2 | Search only the DL models | Unequal budget; any DL gain is confounded with tuning effort. Rejected. |

### 2.4 Seeds

| # | Option | Real cost |
|---|---|---|
| S1 | One seed | Cheapest; seed variance of a network on this data is unknown, so a single run cannot support a claim. |
| S5 | **Five seeds per system, DL and LightGBM alike** (section 4.4) | 5x training cost for DL (projected after the pilot) and 5 LightGBM CPU fits per feature set (time not measured, H-D6). LightGBM retraining is **mandatory**: the margin `delta` of section 4.4 is defined from its seed SD. |

### 2.5 Headline metric

| # | Option | Real cost |
|---|---|---|
| M1 | Macro-F1, BALANCED (P0), CI > 0 in 4 of 5 seeds (earlier draft) | Already computed by `eval.metrics`; measures a balanced mix that no Gaia user queries; a gain can come from STAR, the class nobody selects. **Replaced** (owner, 2026-11-02/03). |
| M2 | **Purity of QSO and of GALAXY at fixed completeness (0.80 primary, 0.90 secondary), thresholds from val, test reweighted to P2** | New code (threshold per system on val, weighted purity, block bootstrap); answers the user question "how clean is a QSO/GALAXY sample I select from Gaia"; under P2 the GALAXY purity hangs on a few STAR contaminants; with the total low-count reading the GALAXY cell is likely reportable (section 4.5). |

### 2.6 Bootstrap unit for the headline CI

| # | Option | Real cost |
|---|---|---|
| U5 | Level-5 pixels within bands (ADR 0003, as Phase 1) | Zero new code; ~3.4 deg^2 clusters are smaller than the degree-scale correlations ADR 0003 T1 worries about, so the CI may be too narrow. Kept as a **descriptive** CI. |
| U3 | **Level-3 parent blocks (~54 deg^2) of the level-5 test pixels, within bands** | One integer division; fewer clusters -> wider, more honest CI; at least 34 blocks per target class on test by pigeonhole (R14). Design choice proposed by the architect (owner, H-D8). |

## 3. Decision (D1): systems, features, preprocessing

1. Systems (model id = file stem; `s<k>` = seed index, seed = 20261025 + k, k = 0..4):

| id | Family | Features | Role |
|---|---|---|---|
| `lgbm_full`, `lgbm_noind` | LightGBM (existing Phase 1 files, never overwritten) | 21 / 11 | descriptive context only |
| `lgbm_full_s<k>`, `lgbm_noind_s<k>` | LightGBM retrained, `PARAMS` unchanged except the seed (section 3.8) | 21 / 11 | reference of the gain rule; source of `delta` |
| `mlp_full_s<k>`, `mlp_noind_s<k>` | MLP | 21 / 11 | main axis (a) |
| `ftt_full_s<k>`, `ftt_noind_s<k>` | FT-Transformer | 21 / 11 | main axis (b) |
| `ens_mlp_full_s<k>`, `ens_ftt_full_s<k>` | LightGBM + network (section 3.6) | 21 | extra (c), descriptive only (owner, 2026-11-03) |
| `xp_*` | network on BP/RP | XP subset | **retired** (owner, 2026-11-03, H-D1: XP out of scope) |

   Gaia DSC (`classprob_dsc_combmod_*`) appears only as context in the report, never in a gain pair.
2. Feature sets are exactly `lgbm.MODEL_FULL.features` and `lgbm.MODEL_NOIND.features`, imported, never
   re-listed.
3. **Preprocessing, fitted on train only** (one object per feature set, serialized next to the model):
   - continuous features: `sklearn.preprocessing.QuantileTransformer(output_distribution="normal",
     n_quantiles=1000, subsample=1_000_000, random_state=20261025)`; chosen as in Gorishniy et al. (2021),
     not tuned. StandardScaler is **not** an arm (one preprocessing, no search);
     (Amended 2026-11-03, T-DL-02 amendment A, trigger T3) the quantile transform is consistent with the
     paper's DL default (Sec. 4.3 / App. B.2). As in the paper, Gaussian noise N(0, 1e-3) is added to the
     **train** numerical values only while fitting the quantiles; the original (noise-free) values of every
     fold are then transformed. Implementation detail for the coder: noise drawn from
     `default_rng(20261025)`, SD 1e-3, non-null cells only (whether the paper's 1e-3 is an SD or a variance
     is not verified). `output_distribution="normal"` and `n_quantiles=1000` are **not verified** against
     the paper;
   - `astrometric_params_solved` (M-full only): categorical {3, 31, 95}, one-hot for the MLP, one
     categorical token for the FT-Transformer; unseen code -> `TrainError`;
   - missing values: for every feature of the set with >= 1 NULL/NaN **in train**, a binary indicator
     column `<name>__isna` is appended and the transformed value is set to 0.0 (the median in normal space).
     Indicators are never created from val/test. **Same masks on both sides**: LightGBM reads NaN natively,
     the network reads the indicator, so both see exactly the NULL cells of `features.parquet`; a NULL in
     val or test in a feature that has no indicator -> `TrainError` (the masks would differ; covariates
     only, no label read). In M-noind the nine astrometric columns are absent, so no astrometric
     missingness indicator exists there (ADR 0004 D7.4 ablation preserved). (Amendment A) The indicator
     masks are this project's own design: neither supported nor contradicted by the paper excerpts read;
   - the preprocessor stores the train-fitted state and its sha256; the predictions parquet records it.
4. **Loss and labels**: unweighted cross-entropy over `CLASSES` order (`STAR, GALAXY, QSO`); no class
   weights, no resampling, so `pi_src` = train class fractions, identical to LightGBM's (asserted).
5. **Calibration**: one temperature per trained model, fitted on val logits with
   `lgbm.fit_temperature` (same bounds); test never enters a fit. A temperature at a bound is flagged as in
   `lgbm.run_model`. The LightGBM seed models get the same calibration through `lgbm.run_model`.
6. **Ensemble (extra c)**: `p = w * p_cal(lgbm_full_s<k>) + (1 - w) * p_cal(net_full_s<k>)`, `w` on the grid
   0.0..1.0 step 0.1 minimizing **val** logloss, ties to the larger `w` (prefer the existing model); then
   one temperature fitted on val on `log p`; stored with `p_raw_*` = the mixture before T. No stacking
   model, no other weights.
7. Architectures (fixed, B0). (Amended 2026-11-03, T-DL-02 amendment A, trigger T3: checked by
   `researcher` T-RES-DL against Gorishniy et al. 2021, arXiv:2106.11959,
   https://arxiv.org/html/2106.11959; the paper wins where it differs.)
   - FT-Transformer: the paper "default" FT-Transformer (Table 12), **confirmed**: 3 blocks, d_token 192,
     8 heads, FFN hidden factor 4/3 with ReGLU, attention dropout 0.2, FFN dropout 0.1, residual dropout
     0.0, AdamW lr 1e-4, weight decay 1e-5. Added from the paper: weight decay **0.0** for the Feature
     Tokenizer, LayerNorm and bias parameters (Table 12 note; the paper's rule is adopted; the package
     README's `make_parameter_groups()` does not name the parameters it protects, so it is not relied on);
     the learned [CLS] token embedding also goes in the weight-decay 0.0 group, with the Feature Tokenizer
     (**project choice**: not settled by the paper text; reference code (rtdl) not consulted);
     **PreNorm** variant (Sec. 3.3) **with the first normalisation of the first Transformer block removed**
     (Sec. 3.3: "In the PreNorm setting, we also found it to be necessary to remove the first normalization
     from the first Transformer layer to achieve good performance."); **Kaiming** initialisation
     (Table 12). Not verified in the excerpts read: the tokenizer detail "one token per numeric feature
     plus one categorical token for `astrometric_params_solved`" (kept as this ADR's design).
   - MLP: 3 hidden layers x 512, ReLU, dropout 0.1, AdamW lr 1e-3, weight decay 1e-5. The paper gives no
     MLP default, only a tuning space (Table 15: layers 1-8 / 1-16, size 1-512 / 1-1024, dropout
     {0, U[0, 0.5]}, lr LogUniform[1e-5, 1e-2], weight decay {0, LogUniform[1e-6, 1e-3]}); these values are
     **project choices inside the paper's ranges**, not tuned.
   - Both: batch 1024 (**project choice**: the paper uses 256 / 512 / 1024 by dataset, Table 7; 1024 matches
     its large-dataset group), `max_epochs` 100 (this ADR's own cost bound; no 100-epoch cap found in the
     excerpts read, not verified), early stopping on val logloss with patience **16** epochs (the paper's
     value for all algorithms, Sec. 4.3; supersedes the earlier 10), best-epoch weights restored, no LR
     schedule (the **paper's setting**: "We do not apply learning rate schedules"). `max_epochs` may
     be **lowered for all systems at once** by the human after the pilot (cost only, decided before any val
     result of the batch is seen).
8. **LightGBM seeds (mandatory, replaces the optional H-D5)**: `lgbm_full_s<k>` and `lgbm_noind_s<k>`,
   k = 0..4, are fitted on CPU with `PARAMS`, `MAX_TREES`, `PATIENCE` and calibration unchanged, only
   `seed` = 20261025 + k. Files go to `data/interim/<release>/lgbm_seeds/` and never overwrite the
   existing `lgbm_*` artefacts. Wall time of the first fit (`lgbm_full_s0`) is measured and reported to
   the human before the other nine run (H-D6). `lgbm_full_s0` uses the same seed as `lgbm_full`; whether
   it reproduces it is reported, not assumed.

## 4. Decision (D2): metrics, hypotheses, criterion of gain (pre-registered)

### 4.1 Population and outputs

Headline numbers: test fold of full_v0 (as built, section 1), population `a_all` (all test rows),
REALISTIC regime with prior **P2** (Delchambre+2023 all-sky, ADR 0003) as the **primary** prior. P1 (all
test rows) and P3 (test rows in bands B and C only, as ADR 0003 prescribes) are **descriptive sensitivity
analyses** with the same rule computed and labelled "sensitivity", never headline. BALANCED (P0) and the
`astrometric_params_solved` strata are descriptive. Gaia DSC appears only as context.

Score used for selection: the calibrated one-vs-rest probability `s_m,c(x) = p_cal_c(x)` of system `m`,
**not** prior-shifted. Reason: the operating point is fixed by completeness, which does not depend on the
class prior (ADR 0003, Metrics table); the prior enters only through the purity weights. One rule for all
systems, so the comparison stays paired (decision reversible in < 1 day).

### 4.2 Metrics

- **Primary**: purity of QSO and of GALAXY at fixed completeness `C* = 0.80` under P2 (formulas F1-F3).
- **Secondary (same rule, reported, not headline)**: the same at `C* = 0.90`.
- **Descriptive only** (no gain claim): macro-F1 (BALANCED, prior_matched, `eval.metrics`); STAR<->QSO
  confusion `e_SQ = (R[STAR->QSO] + R[QSO->STAR]) / 2`, where `R[a->b]` is the BALANCED row-normalized
  confusion rate (fraction of true `a` predicted `b`), lower is better, reported with the floor of STAR and
  QSO only; accuracy; per-class purity/completeness/F1/PR-AUC; ECE mean (15 equal-mass bins,
  prior_matched); a risk-coverage (abstention) curve per system on the `tau` grid 0.50..0.99 step 0.01
  with confidence = max `p_cal_*`; the realised **test** completeness of every system and seed at its val
  threshold, printed next to `D_c` (section 4.4); the raw contaminant counts of every selection, per
  contaminating class (for GALAXY: STAR and QSO separately); costs (section 8).

Formulas (fold `F` is val or test; `m` a system at one seed; `c` in {QSO, GALAXY}; `pi` a prior of the
ADR 0003 grid):

```
F1  threshold, fitted on VAL only, frozen in val_freeze.json:
      S = scores s_m,c(x_i) for i in val with y_i = c, sorted descending; n = |S|
      k = ceil(C* * n);  t_m,c(C*) = S[k]   (1-based: the k-th largest)
      selected(i) = [ s_m,c(x_i) >= t_m,c(C*) ]      (ties included, so val completeness >= C*)
F2  weighted purity on fold F:
      w_i = pi[y_i] / n_F[y_i]          (= reweight(y_F, pi), ADR 0003; recomputed per bootstrap replicate)
      P_m,c(F; pi, C*) = sum_{i in F} w_i [y_i = c] selected(i) / sum_{i in F} w_i selected(i)
      undefined (n/a) when the denominator is 0
F3  seed mean of a family X (k = 0..4, N_SEEDS = 5):
      Pbar_X,c(test; pi, C*) = (1/5) * sum_k P_{X_s<k>},c(test; pi, C*)
      each seed with its own threshold t_{X_s<k>},c(C*); probabilities are never averaged across seeds
F4  point difference for the pair X vs L (L = LightGBM seeds of the same feature set):
      D_c = Pbar_X,c(test; P2, 0.80) - Pbar_L,c(test; P2, 0.80)
F5  margin, from LightGBM only, on VAL, before any DL run (written to lgbm_seeds/delta.json):
      v_k = P_{L_s<k>},c(val; pi, C*)      k = 0..4, threshold t_{L_s<k>},c(C*) from F1
      delta_f,c(pi, C*) = 2 * sd(v_0..v_4, ddof = 1)
                        = 2 * sqrt( sum_k (v_k - mean(v))^2 / (5 - 1) )      (numpy.std(v, ddof=1))
      one value per feature set f in {full, noind}, class c, prior pi in {P1, P2, P3}, C* in {0.80, 0.90};
      the headline uses delta_full,c(P2, 0.80)
```

(Clarified 2026-11-04, T-DL-04, clarification, no number changes) F5 populations: for the P3
sensitivity cell the threshold `t_{L_s<k>},c(C*)` is fitted on ALL val rows of class `c` (F1 does not
depend on the prior), while the purity `v_k`, the floor check and the block count are taken on the val
rows of latitude bands B and C only (the P3 population of ADR 0003 D4); P1 and P2 use all val rows for
all three. The headline `delta_full,c(P2, 0.80)` is unaffected. Reproducibility: val is processed in
ascending `source_id` order, so `delta.json` is bit-for-bit reproducible; `delta.json` records
`labels_sha256` and the sha256 of the 10 LightGBM seed prediction files, and `val_freeze.json` hashes
`delta.json`.

What the margin means (stated so the owner approves it knowingly, H-D8). `delta` is the spread of a
**single** seed, while `D_c` compares two **5-seed means**. If DL and LightGBM have the same seed SD `s`,
the seed part of `D_c` has `SD(D_c) = s * sqrt(1/5 + 1/5) = s * sqrt(2/5) ~= 0.63 s`, so `delta = 2 s` is
about `3.2 SD(D_c)`, i.e. about `sqrt(2.5) ~= 1.58` times a nominal 2-SD margin on `D_c`. `delta` rests on
4 degrees of freedom: the relative standard error of a sample SD is about `1 / sqrt(2 * 4) ~= 35 %`.
Consequences: a false "win" from seed noise alone is very unlikely (type-I error very low, before the CI
condition is even applied), and power is reduced: a true gain near `2 s` is declared a win at most about
half the time (this counts only the seed term `P(D_c > delta_hat)`; the block-bootstrap CI condition
(about 34 blocks, possibly wide) and the noise of `delta_hat` lower or shift it; power was not computed).

**SD fold (proposal, flagged for the owner, H-D8): val.** On val, threshold and purity come from the same
rows, so `v_k` is in-sample at the threshold. Measuring on test would be more faithful but would use the
test fold to set a constant of the rule that is then applied on test. Val keeps test untouched until the
single evaluation. Cost: the val SD may differ from the test SD; the factor 2 absorbs part of that.

Limitations of thresholds fitted on val (stated in every report):
- the sampling variance of the val threshold `t_m,c(C*)` is **not** in the bootstrap CI (thresholds are
  frozen; only the test rows are resampled);
- a system whose val threshold lands at a **lower** realised test completeness selects fewer, more certain
  objects and gets a higher test purity; the realised test completeness per system and seed is therefore
  printed next to `D_c` as a descriptive column. It does **not** gate the verdict (no tolerance is applied).

### 4.3 Hypotheses (stated before training)

| id | Hypothesis | Pair(s), first minus second | Expected |
|---|---|---|---|
| H1 | MLP ties LightGBM on the tabular signal | `mlp_full - lgbm_full` (5-seed means, both sides) | no gain |
| H2 | FT-Transformer ties LightGBM on the tabular signal | `ftt_full - lgbm_full` (5-seed means, both sides) | no gain |
| H3 | The ensemble may gain over LightGBM (errors partially decorrelated) | `ens_ftt_full - lgbm_full`, `ens_mlp_full - lgbm_full` | possible gain, small; **descriptive only** (owner, 2026-11-03) |
| H4 | The missing-astrometry pattern helps both families alike | `ftt_full - ftt_noind`, `mlp_full - mlp_noind` next to `lgbm_full - lgbm_noind` (seed means) | same sign as LightGBM |
| H5 | Retired 2026-11-03 (BP/RP, XP out of scope, H-D1) | none | none |

`lgbm_full` in H1-H4 means the five `lgbm_full_s<k>`; the Phase 1 `lgbm_full` file is context only.
Only H1 and H2 (pure MLP, pure FT-Transformer versus LightGBM) can produce a verdict (owner, 2026-11-03).

### 4.4 Criterion of gain (decision rule, fixed now; code constants in section 15)

This rule **replaces** the earlier draft's macro-F1 rule "CI lower bound > 0 in >= 4 of 5 seeds" (owner,
2026-11-02/03; the replacement was confirmed by the owner on 2026-11-03, H-D4).

Block bootstrap (headline CI; design choice proposed by the architect, flagged for the owner, H-D8):
- block id of a test object: `block_i = pixel5_i // 16` (= `pixel5_i >> 4`), the level-3 nested parent of
  its level-5 pixel (nested scheme: one level coarser = integer division by 4);
- latitude band = the ADR 0003 band of the object's **level-5** pixel; a level-3 block that crosses a band
  boundary is split into its per-band parts, each a block;
- one replicate = `eval.metrics.bootstrap_pixel_draw(block, lat_band, rng)`; `rng =
  numpy.random.default_rng(20261025)` created once per population; B = 1000; the **same replicates** for
  every system and seed (paired); thresholds are not refitted; weights are recomputed per replicate;
- `D_c` is recomputed on each replicate; 95 % percentile interval `[lo_c, hi_c]`; a replicate where class c
  is absent, or where any system's F2 denominator is 0, is invalid; > 1 % invalid -> the cell is `n/a`;
- this CI covers sky-sampling variation with the models and thresholds fixed; seed variation enters
  through `delta`;
- `MIN_BLOCKS` = 20 mirrors the ADR 0003 floor of >= 20 pixels (`eval.metrics.MIN_PIXELS` = 20; ADR 0003:
  "fewer than 20 clusters makes the pixel bootstrap unstable"), applied to blocks; it is fixed here before
  any DL result; the 34-block bound (R14) was seen only afterwards, as a feasibility check;
- the ADR 0003 level-5 paired CI of the same `D_c` is reported next to it as descriptive.

Per headline pair `X - L` (H1, H2 only) and class c in {QSO, GALAXY}, at P2 and `C* = 0.80`:
- the cell is **reportable** iff (i) class c passes the ADR 0003 floor on test (>= 100 objects in >= 20
  level-5 pixels), (ii) class c occupies >= 20 blocks on test (`MIN_BLOCKS`, see above), (iii) no system or
  seed in the pair has a `low-count` flag for c, where `low-count` means: the SUM of the raw contaminant
  counts over the other classes in the selection is < 10 (ADR 0003, total reading, exactly as
  `src/eval/metrics.py:124-128`); a contaminant class absent from the test set adds 0 and does not
  trigger the flag by itself (owner's decision, 2026-11-03), (iv) invalid replicates <= 1 %, (v)
  `delta_full,c(P2, 0.80)` is finite and its five val cells pass the same checks (i)-(iii) on val;
- `win_c` iff reportable **and** `D_c > delta_c` **and** `lo_c > 0` (pointwise above the margin, CI above
  zero; the CI is **not** required to exceed `delta`);
- `worse_c` iff reportable **and** `D_c < -delta_c` **and** `hi_c < 0` (descriptive label only).

Verdict per pair: **"X wins"** iff `win_QSO` **and** `win_GALAXY`. Otherwise **"tie: LightGBM stays"**
(this includes a CI containing zero, a point difference <= `delta`, and any non-reportable cell, section
4.5). `worse_c` is printed per class; it never changes the verdict. The same rule at `C* = 0.90` and at
P1/P3 is printed as secondary/sensitivity and never overrides the headline. Seed variation is reported as
min / median / max of the per-seed purities and of `v_k`. Next to every `D_c` the report prints the
realised test completeness per system and seed and the raw contaminant counts per contaminating class
(GALAXY selection: STAR and QSO separately); these columns are descriptive and never change the verdict.
Under P2 one STAR contaminant weighs about 5,000 GALAXY objects (reviewer's figure; exactly
`(0.998813 / 0.000198) * (n_GALAXY / n_STAR)` on the evaluated rows), so the GALAXY purity hangs on a few
stars.

Four fronts, never mixed: (1) model effect with full signal `net_full - lgbm_full`; (2) model effect with
reduced signal `net_noind - lgbm_noind`; (3) signal effect within the network `net_full - net_noind`;
(4) signal effect within LightGBM `lgbm_full - lgbm_noind`. A pair that changes both model and signal
(e.g. `ftt_full - lgbm_noind`) is not computed. Only H1 and H2 are headline. The ensemble pairs (H3) and
fronts (2)-(4) are computed with the same formulas and printed as **descriptive**: they never declare a
win and never enter the verdict (owner, 2026-11-03, for the ensembles). No multiplicity correction is
applied across the two headline pairs; requiring both classes is the only guard against chance wins. The
report says so.

Negative-result sentence (emitted when H1 and H2 are both "tie: LightGBM stays"): "On the same 21 Gaia DR3
features, MLP and FT-Transformer (5 seeds, untuned defaults) do not raise the purity of QSO and GALAXY at
completeness 0.80 under the Delchambre+2023 prior beyond 2 seed-SDs of LightGBM with a block-bootstrap CI
above zero." It names which cells were not reportable; the ensemble results are reported next to it as
descriptive. No configuration is changed after test is seen to rescue it; any new attempt is a new
`MODEL_VERSION` evaluated on val first and declared as post-test.

### 4.5 Cells that are n/a or low-count

Rule (proposal, owner decision H-D8): a cell that is not reportable (section 4.4) **cannot support a win
and counts as a tie for that class**; therefore a win requires a reportable cell in **both** QSO and
GALAXY. The low-count part of the rule uses the total reading (owner's decision, 2026-11-03).

Feasibility, from Phase 1 numbers already published (`data/interim/full_v0/eval/results.md`; reviewer's
reading, `lgbm_full`; these are **prior_matched argmax** cells, **not** threshold-based cells at `C*`):
- GALAXY test: 112,571 objects in 542 level-5 pixels (results.md:224); QSO: 541 pixels;
- predicted-GALAXY contaminants: 12 STAR + 159 QSO at GALAXY completeness ~0.68 (results.md:298) and
  141 STAR + 730 QSO at ~0.99 (results.md:260).

Under the total reading both counts sum to far more than 10, so a GALAXY cell at `C* = 0.80` under P2 is
**likely reportable** (not guaranteed: the threshold-based counts are not measured). A DL win is
therefore hard but possible. Reversal option, documented and not adopted: decide on QSO alone when GALAXY
is not reportable (cheaper to win, weaker claim).

## 5. Decision (D3): output files and how the evaluator receives them

1. Directory: `data/interim/<release>/dl/` (git-ignored like all `data/`). Files:
   - `predictions_<model_id>_v0.parquet`: **exactly** the schema of `lgbm.predictions_table` (`source_id`
     int64 sorted, `model_version`, `p_raw_{star,qso,galaxy}`, `p_cal_{star,qso,galaxy}` float32), rows =
     val + test ids, written with temp+rename, zstd. Metadata = the LightGBM keys (`schema_name`,
     `schema_version`, `model_id`, `train_class_fractions`, `calibration`, `calibration_fit_fold`, ...) plus
     `family`, `arch_config` (JSON), `seed`, `best_epoch`, `epochs_run`, `temperature`,
     `preprocessor_sha256`, `torch_version`, `device`, `feature_set`;
   - `models/<model_id>.pt` (best weights) and `models/<model_id>.preproc.pkl`; manifests via
     `ingest.manifest` like LightGBM;
   - `train_report_dl.json`: per model val metrics only (logloss before/after T, ECE, macro-F1 on val,
     purity at `C*` on val under P2), costs, temperature, best epoch. **It contains no test metric**
     (unlike `train_report.json`).
   - `val_freeze.json`: written once, after all planned systems exist, listing sha256 of every DL and
     LightGBM-seed predictions parquet and of `lgbm_seeds/delta.json`, every threshold `t_m,c(C*)`
     (system, seed, class, `C*`), the constants of section 4.4 and the human approval line; refused if it
     exists.
   - LightGBM seeds: `data/interim/<release>/lgbm_seeds/predictions_lgbm_{full,noind}_s<k>_v0.parquet`
     (same schema), `train_report_lgbm_seeds.json` (val metrics and wall time only) and `delta.json`
     (every `delta` of F5 with its five `v_k`, thresholds and floor checks; val only), written once
     **before any DL batch**; refused if it exists.
2. Evaluator input: `outdir` (release dir) + `dl/` + `lgbm_seeds/`. It calls `runner.load_test_set(outdir)`
   (gives the test rows, pixels, bands and the two Phase 1 LightGBM systems) and aligns every DL and
   LightGBM-seed parquet on that `source_id` order, reading `p_cal_*` by name. It refuses (exit 1) when: a
   parquet does not cover exactly the val+test ids; probabilities are not finite or do not sum to 1 (1e-3);
   `train_class_fractions` differs from LightGBM's by > 1e-9; `val_freeze.json` is missing or any sha256
   differs; a seed listed in the freeze is missing.
3. Outputs: `dl/eval/results_dl.json`, `dl/eval/results_dl.md` (header PROVISIONAL as Phase 1 if below
   floor), `dl/eval/risk_coverage_dl.csv`.

## 6. Decision (D4): tuning budget

**B0** (equal to LightGBM: zero search), **decided by the owner on 2026-11-03 (H-D3)**: one fixed
configuration per architecture; val is used only for early stopping, temperature and the ensemble weight.
Limitation stated in every report: a DL loss or tie could be attributed to the lack of tuning. Reversal
option: B1 (equal N-trial search for both families, LightGBM retrained with the same N), only through
trigger T8 and declared as post-test if test was already seen.

## 7. Decision (D5): seeds and determinism

S5: seeds 20261025 + k, k = 0..4, set for Python, NumPy, torch (CPU and CUDA) and the DataLoader generator;
for LightGBM, `PARAMS["seed"]` = 20261025 + k (section 3.8).
`torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; GPU
non-determinism that remains is recorded as a limitation, not hidden (bit-identical reruns are **not** an
acceptance criterion on GPU; on CPU they are).

## 8. Decision (D6): costs reported per system

Measured, never estimated in the report: train wall time per epoch (mean, max) and total; epochs run; time
to predict val+test (s) and per 1M rows; peak GPU memory (`torch.cuda.max_memory_allocated`); parameter
count; checkpoint size; device name. Peak host RSS: measured on Linux (Kaggle) via `resource`, "not
measured" on Windows. For LightGBM: training wall time per seed is measured by `scripts/train_lgbm_seeds.py`
(H-D6, first on `lgbm_full_s0`); inference time measured by the evaluator only if the human asks.

## 9. Decision (D7): compute plan and Kaggle logistics

0. Compute, in this order (owner, 2026-11-03): **(A) Kaggle GPU** if the quota allows (quota and session
   length not read yet, H-D2); **(B) free Colab** (GPU type, session length and idle limits **not
   verified**; the owner supplies the data to the session; no Colab-specific file is planned, the owner
   runs `scripts/train_dl.py` from a notebook cell; checkpoints written to persistent storage the owner
   chooses); **(C) CPU on a train subsample**: `scripts/train_dl.py --plan-c-subsample N`, outputs under
   `data/interim/<release>/dl_sub<N>/`; the LightGBM seeds are retrained on the **same** subsample
   (`lgbm_seeds_sub<N>/`, same row draw, `dl_data.subsample_index`) so the comparison stays equal; val and
   test stay complete; every result is labelled "train subsample N". N is decided by the owner after the
   P-local timing. Checkpoint and deterministic resume (item 3) apply to A, B and C.
   (Amended 2026-11-03, T-DL-02 amendment A) **GPU session check**: every Kaggle or Colab GPU session
   prints `torch.__version__`, `torch.version.cuda` and the GPU name **before any training**, and aborts
   with a clear message if the GPU is a P100 (sm_60) or the torch build does not list the device's
   architecture (`torch.cuda.get_arch_list()`); the three values are recorded in `environment.json` and
   in the run report. The pin is read from the **first** GPU session of the platform in use (Kaggle or
   Colab) and recorded per platform in `environment.json`; within a platform, a later session whose torch
   differs from that platform's recorded pin stops before training; there is no cross-platform assert
   (Kaggle and Colab runs are separate and labelled). `requirements-dl.txt` carries the pin read from the
   first session, not guessed (H-D7).
1. Private kernel(s) named `astro-triage-dl-*`, accelerator GPU (type: not verified), internet on only for
   pip of the package (torch is **not** pip-installed: the preinstalled version is recorded and asserted
   against the Kaggle pin recorded under item 0).
2. Inputs: (a) the private package dataset built by `scripts/make_kaggle_package.py` (it already ships all of
   `src/`, `scripts/`, `kaggle/`); (b) the Phase 1 kernel output (`astro_triage_outputs.zip` with
   `full_v0/labels.parquet`, `features.parquet`, `predictions_lgbm_*_v0.parquet`, `train_report.json`) if
   still available as a notebook output — **to be confirmed in the Kaggle interface**; if not, the human
   decides how to supply it (no agent uploads data). The LightGBM seeds (`lgbm_seeds/`) are trained on CPU
   (locally or in a CPU kernel) and supplied the same way.
3. Checkpoint per epoch and per seed: `ckpt/<model_id>/last.pt` (model, optimizer, epoch, best val loss,
   patience counter, RNG states, preprocessor sha) overwritten by temp+rename each epoch, `best.pt` on
   improvement. Restart = add the previous version's output as input; finished model ids (parquet present
   and sha recorded) are skipped; a checkpoint whose preprocessor sha or config differs is refused.
4. One output: `astro_triage_dl_outputs.zip` (dl/ tree, checkpoints of unfinished models only,
   `environment.json`, `run_log.txt`); `/kaggle/working` ends with the zip and the log only.
5. The bootstrap evaluation (B = 1000) runs **off the GPU**: locally (CPU, `scripts/evaluate_dl.py`) or in
   a separate CPU kernel.
6. Limits: weekly GPU hours and maximum session length are **not verified**; the plan does not depend on a
   number until the human reads them in the interface. (Amendment A) Official numbers were not found;
   third parties report ~30 h/week and 12 h per session (low confidence); the owner's real quota is unknown.
7. Rules: only resources named `astro-triage-*`, all private; uploads contain code only; never touch other
   datasets of the account. **Every write to Kaggle (dataset/kernel create or version) and every output
   download requires the human's explicit "yes".**

### 9.4 Smoke pilot (before any batch)

- P-local (CPU, no Kaggle): every system on 10,000 train rows drawn with `default_rng(20261025)`, full val,
  1 epoch; correctness only (shapes, files, schema, resume after a kill); timings here are not used.
- P-gpu (Kaggle, after the human's "yes"; or Colab under plan B): `ftt_full_s0`, `mlp_full_s0`,
  `ftt_noind_s0`, `mlp_noind_s0`, **1 epoch on the full train fold**, then val prediction. Measures
  sec/epoch, peak GPU memory and inference time. Its val numbers are not reported as results. It starts
  with the GPU session check of section 9, item 0, and its printed torch version fixes the H-D7 pin.
- Projection to the human: `sec/epoch x max_epochs x 5 seeds x 4 systems` as an **upper bound** (early
  stopping makes it lower), next to the GPU quota the human reads. Nothing measured is invented; anything
  not measured is printed as "not measured".
- Batch only after an explicit "yes" to: systems, seeds, `max_epochs`, number of sessions. If the budget is
  short, priority order: `ftt_full`, `mlp_full` (headline), then `*_noind`.

## 10. Risks

| id | Risk | Mitigation / measurement |
|---|---|---|
| R1 | GPU quota/time insufficient for 20 runs (not measured, not verified) | Pilot projection; priority order; plan B/C (section 9, item 0); seeds may be cut by the human before any DL run, then both families use the same N >= 3 and F3/F5 use that N (ddof = 1) |
| R2 | Unequal effort masked as model effect (defaults vs hand-set LightGBM) | B0 for both (owner, H-D3); stated as limitation; B1 via T8 |
| R3 | FT-Transformer defaults misremembered | Checked 2026-11-03 by `researcher` T-RES-DL (amendment A, section 3.7); residual: the tokenizer detail and `output_distribution` / `n_quantiles` are not verified and are labelled so |
| R4 | Missingness indicators let the network read the class through the astrometric NULL pattern | Same information as LightGBM M-full (it sees `astrometric_params_solved` and the same NULL cells, section 3.3); front (2) with `noind` measures it |
| R5 | QuantileTransformer fit on 1.4M rows: memory/time on Kaggle (not measured) | `subsample=1_000_000`; measured in P-local and P-gpu |
| R6 | GPU non-determinism -> seed runs not reproducible bit-for-bit | Recorded; claims rest on 5 seeds, not on one run |
| R7 | Test seen before all seeds finish, then a "fix" | `val_freeze.json` gate; evaluator refuses without it |
| R8 | Retired 2026-11-03 (BP/RP extra removed, H-D1) | none |
| R9 | Phase 1 Kaggle output no longer available as input | Confirm in the interface; human decides the supply path |
| R10 | `torch` version on Kaggle differs from the local pin | The pin is read from the first GPU session of the platform in use and recorded per platform in `environment.json`; within a platform, a later session whose torch differs from that platform's recorded pin stops before training; no cross-platform assert (Kaggle and Colab runs are separate and labelled) (section 9, item 0). (Amendment A, not verified) Kaggle GPU image torch is reported as 2.10.0+cu128 only by a community comment dated 2026-06-14 on github.com/Kaggle/docker-python issue 1546 (confidence low-medium; no image tag/date found); Colab free runtime torch: not found; newest cp312 win_amd64 CPU wheel seen on download.pytorch.org: 2.9.1+cpu (a 2.10.0 entry not established) |
| R11 | Pinning `torch` in `requirements.txt` makes the Phase 1 CPU runner pip-install torch | Pin in `requirements-dl.txt` instead (H-D7) |
| R12 | Closed 2026-11-03: the ADR 0001 G < 20 text was a documentation error (H-D9) | none |
| R13 | ADR 0002 section 2 redshift rule not applied (GCUT-RC3); some labels may be inconsistent with their redshift | Documented limitation (section 1); the protocol applies to full_v0 as built; T7 |
| R14 | Few level-3 blocks per class -> wide CI or `n/a` | Measured bound: GALAXY test spans 542 level-5 pixels and QSO 541 (results.md:224), and a level-3 block holds at most 16 level-5 pixels, so each class occupies >= ceil(542/16) = 34 and >= ceil(541/16) = 34 blocks (pigeonhole; band splitting only adds blocks) >= `MIN_BLOCKS` 20; block counts per class printed; level-5 CI printed next to it |
| R15 | `delta` from 5 values is itself noisy (4 degrees of freedom, relative SE ~35 %); val SD may differ from test SD | Factor 2 (~3.2 SD of `D_c`, section 4.2); `v_k` printed; SD fold flagged (H-D8); T5 |
| R16 | GALAXY purity under P2 not reportable -> a DL win impossible by construction | Conditional on the low-count reading; resolved by the owner's total reading (2026-11-03): published `lgbm_full` prior_matched argmax cells show 12 STAR + 159 QSO contaminants at completeness ~0.68 and 141 + 730 at ~0.99 (results.md:298, 260; not threshold-based), so the cell at `C* = 0.80` is likely reportable; QSO-only decision kept as reversal option (section 4.5) |
| R17 | Val-threshold sampling variance is not in the CI; a system whose threshold lands at lower test completeness gets higher test purity | Stated as limitation (section 4.2); realised test completeness printed next to `D_c`; no tolerance gating |
| R18 | (Amendment A) Current torch builds reported failing on the Kaggle P100 (sm_60) while the T4 (sm_75) works (same docker-python issue 1546; not verified) | GPU session check aborts on a P100 or a build without the device's architecture (section 9, item 0) |

## 11. Reversal triggers

- T1. P-gpu projection exceeds the human's GPU budget even for `ftt_full` + `mlp_full` with 3 seeds ->
  drop FT-Transformer to `noind`-free headline only, or drop the GPU and run MLP on CPU (human decides).
- T2. On val (before test), a network fails to beat the trivial `pi_src` logloss after `max_epochs` ->
  implementation bug suspected; stop, fix, rerun; no test evaluation of that system.
- T3. Researcher shows FT-Transformer defaults differ from section 3.7 -> amend before training. Fired
  2026-11-03 (T-RES-DL); amendment A applied to sections 3.3 and 3.7 before any training.
- T4. Retired 2026-11-03 (XP out of scope, H-D1).
- T5. `delta.json` (before any DL run) shows `delta_full,c(P2, 0.80) = 0` for a class (identical purity on
  the five LightGBM seeds) or a non-finite `delta` for QSO -> stop; the human re-approves the rule (H-D8)
  before any DL batch.
- T6. Closed 2026-11-03: the G < 20 question was a documentation error (H-D9); it cannot fire.
- T7. The ADR 0002 section 2 redshift-consistency rule (`z_star_max` / `z_qso_min`, GCUT-RC3) is
  implemented -> the labels change; the LightGBM seeds, `delta.json`, every DL run and `val_freeze.json` are
  redone on the new labels before any comparison; results on full_v0 as built are not mixed with them.
- T8. The owner wants to rule out "untuned" after a tie or loss -> B1 (equal N-trial search for both
  families) as a new `MODEL_VERSION`, evaluated on val first, declared as post-test.

## 12. Decisions for the human

- H-D1. BP/RP XP: **decided by the owner (2026-11-03): out of scope.** Recommendation: none needed;
  reopen only through a new ADR (new signal).
- H-D2. Compute: confirm the Kaggle accelerator, weekly quota and session length read in the interface;
  approve P-gpu; else plan B or C (section 9, item 0). Open. **Before P-gpu.** Recommendation: read the
  quota first, run P-gpu, decide A/B/C from its projection. (Amendment A) Official Kaggle numbers were
  not found; third-party figures (~30 h/week, 12 h per session) are low confidence; the owner's real quota
  is unknown.
- H-D3. Tuning budget: **decided by the owner (2026-11-03): B0**, zero search, val only for early stopping
  and temperature; limitation recorded; B1 is the reversal option (T8).
- H-D4. Seeds: 5 for LightGBM and for each DL system (confirmed by the owner). **Confirmed by the owner
  (2026-11-03)**: the earlier "4 of 5 seeds" rule is replaced by the delta/CI rule of section 4.4.
  **Before training.** Recommendation: none (decided).
- H-D5. Retrain LightGBM with the same 5 seeds: now **mandatory** (section 3.8), because `delta` needs its
  seed SD; new files only, never overwriting `lgbm_*`. Recommendation: run after H-D6 is measured.
- H-D6. Measure LightGBM training wall time (first on `lgbm_full_s0`) before the other nine seed fits.
  Open, not measured. Recommendation: measure one fit, then approve the rest.
- H-D7. Pin `torch` in a separate `requirements-dl.txt` (proposed, R11) instead of `requirements.txt`.
  **Decided by the owner (2026-11-03).** Version to pin: the one preinstalled in the Kaggle GPU image, **not
  verified**; local CPU wheel for CPython 3.12 win_amd64 of that version: **not verified** (researcher
  handoff). Recommendation: separate file; the pin is read from the first GPU session of the platform in
  use and recorded per platform; within a platform a later mismatch stops before training; no
  cross-platform assert (plan C on CPU: the version is recorded only). (Amendment A, not verified)
  Reported only by a community comment (2026-06-14, Kaggle/docker-python issue 1546): Kaggle GPU torch
  2.10.0+cu128; newest cp312 win_amd64 CPU wheel seen: 2.9.1+cpu; Colab: not found. The pin is the
  version printed by the first GPU session of the platform in use (section 9, item 0).
- H-D8. Re-approve the **new** pre-registered constants before any test number exists: `C*` 0.80 / 0.90,
  P2 primary (P1, P3 sensitivity), F1-F5 (thresholds on val, seed means, `delta` = 2 x SD with ddof = 1),
  the **SD fold** (proposal: val), the **block bootstrap** (level-3 parents within bands, B = 1000,
  `default_rng(20261025)`, `MIN_BLOCKS` = 20, mirroring ADR 0003 `MIN_PIXELS` = 20), the **n/a rule**
  (non-reportable cell = tie for that class, so a win needs both classes reportable), the seed-mean reading
  of "pointwise difference", and T5.
  Already decided by the owner on 2026-11-03 and recorded here: **low-count = total reading** (the SUM of
  the raw contaminant counts over the other classes < 10, as `src/eval/metrics.py:124-128`; a contaminant
  class absent from the test set does not trigger the flag); the **ensemble pairs are descriptive only**
  (the verdict covers pure MLP and pure FT-Transformer versus LightGBM). Real numbers of the margin, to be
  approved knowingly: with equal seed SD `s` on both sides, `SD(D_c) ~= 0.63 s`, so `delta = 2 s ~= 3.2
  SD(D_c)` (~1.58 x a nominal 2-SD margin); `delta` rests on 4 degrees of freedom (relative SE of the SD
  ~35 %); type-I error very low, power reduced (a true gain near `2 s` wins at most about half the time;
  this counts only the seed term `P(D_c > delta_hat)`; the block-bootstrap CI condition (about 34 blocks,
  possibly wide) and the noise of `delta_hat` lower or shift it; power was not computed).
  **Approved as written by the owner (2026-11-03).** Recommendation: none (decided); the documented
  alternative remains QSO-only when GALAXY is `n/a`.
- H-D9. ADR 0001 G < 20 anomaly: **closed 2026-11-03** (documentation error, no rebuild; ADR 0001 and
  schema amended). Recommendation: none.

## 13. Non-goals

Images/CNN on Gaia or SDSS cutouts (a different signal and a different data path; not planned); BP/RP XP
spectra; deep ensembles across seeds; hyperparameter search (B0; B1 only via T8); class weights or
resampling; any change to split, seed or floor of ADR 0003; overwriting the existing `lgbm_*` files (the
5-seed retrain writes new files); BALANCED or P1/P3 headline claims; a win claim for the LightGBM+network
ensembles; Gaia DSC as a compared system; implementing the ADR 0002 section 2 redshift rule (GCUT-RC3);
Layer 2 interaction; publishing any Kaggle resource; evaluating partially finished seeds.

## 14. Consequences

- Good: the DL axis is one more producer of the same predictions schema; every comparison uses the same
  rows and replicates for all systems; the gain rule, its margin formula and the negative sentence exist
  before training; the margin is fixed from LightGBM on val before any DL run; deleting the axis =
  deleting `dl/`, `lgbm_seeds/` and the files of section 15.
- Bad: GPU cost unknown until the pilot; untuned defaults invite the "not tuned" objection (B0 decided);
  5 seeds x 4 systems multiply cost, plus 10 LightGBM CPU fits (time not measured); GPU runs are not
  bit-reproducible; one more evaluator module partly mirrors `runner.run_boot`; the GALAXY purity under P2
  hangs on a few STAR contaminants and its cell, although likely reportable, is not guaranteed to be
  (section 4.5); `delta` is conservative (~3.2 SD of `D_c`) and noisy (4 degrees of freedom), so power is
  reduced; the val-threshold variance is outside the CI; labels carry the unapplied redshift rule (R13).
- Maintenance cost: 5 source modules (~800 lines total, including a small `lgbm.py` change), 3 scripts,
  1 Kaggle runner (~200 lines), 5 test files, 1 requirements file; re-pin torch when the Kaggle image
  changes.
- Reversibility: everything < 1 day **except** test exposure: after `evaluate_dl.py` runs on test, the
  configurations of section 3 and the constants of section 4 cannot be changed and re-evaluated on the
  same test without declaring it.

## 15. Implementation plan (per file)

Code constants (in `evaluation/dl_compare.py`): `HEADLINE_METRIC = "purity_at_completeness"`,
`TARGET_CLASSES = ("QSO", "GALAXY")`, `PRIMARY_COMPLETENESS = 0.80`, `COMPLETENESS_LEVELS = (0.80, 0.90)`,
`PRIMARY_PRIOR = "P2"`, `SENSITIVITY_PRIORS = ("P1", "P3")`, `DELTA_SD_MULTIPLIER = 2.0`, `DELTA_DDOF = 1`,
`DELTA_FOLD = "val"`, `BLOCK_LEVEL = 3`, `BLOCK_DIVISOR = 16`, `MIN_BLOCKS = 20`, `N_SEEDS = 5`,
`LOW_COUNT_RULE = "total"`, `HEADLINE_PAIRS = (("mlp_full", "lgbm_full"), ("ftt_full", "lgbm_full"))`,
`DESCRIPTIVE_PAIRS = (("ens_mlp_full", "lgbm_full"), ("ens_ftt_full", "lgbm_full"))` (each name = its five
`_s<k>` files). `SEED_MAJORITY` is removed.

```
FILE: src/astro_triage/models/lgbm.py                       [modify, small]
INTENT: let a caller fit with another seed and model id without changing the default behaviour
EXPORTS:
  def run_model(spec: ModelSpec, folds: dict[str, Fold], outdir: Path, *, seed: int = SEED,
                model_id: str | None = None) -> dict[str, object]
      # seed replaces PARAMS["seed"] for this fit only, in a per-fit copy of PARAMS; the manifest params
      # string must be built from that copy in BOTH places: lgbm.py:473 (json.dumps(PARAMS,
      # sort_keys=True), which embeds PARAMS["seed"]) and lgbm.py:475 (seed={SEED});
      # model_id overrides spec.model_id in file names, metadata and manifests
DEPENDS: unchanged
INVARIANTS: PARAMS is never mutated (copy per fit); default call keeps today's behaviour
ACCEPTANCE: default call on the synthetic release, compared with the same call before the change ->
  identical data columns (source_id, model_version, p_raw_*, p_cal_*) and identical parquet metadata
  EXCEPT the volatile keys `created_utc` and `git_sha` (set at lgbm.py:451-452); OR, equivalently,
  identical booster model string (`models/<model_id>.txt`) and identical probabilities. A byte-identical
  parquet is NOT required (the volatile keys change it). seed=20261026 -> metadata/manifest seed 20261026
  and file predictions_<model_id>_v0.parquet; the per-fit params copy is the one serialised in the
  manifest: for seed=20261026 neither the `params={...}` JSON (line 473) nor `seed=` (line 475) contains
  20261025, so lgbm_full_s1..s4 never say seed 20261025 inside params
RISK: fit_booster reads PARAMS directly; the seed must be threaded through it, not set globally
```

```
FILE: scripts/train_lgbm_seeds.py                           [create, target <= 120 lines]
INTENT: CLI: --outdir --features {full,noind} --seeds 0-4 [--plan-c-subsample N] [--delta]; fits the
  LightGBM seeds into lgbm_seeds/ (or lgbm_seeds_sub<N>/), records wall time per fit; --delta writes
  delta.json from val only (F1, F2, F5)
DEPENDS: argparse, json, sys, time, pathlib, astro_triage.models.lgbm (load_folds, run_model, MODEL_FULL,
  MODEL_NOIND, TrainError), astro_triage.models.dl_data (subsample_index),
  astro_triage.evaluation.dl_compare (compute_delta) (closed list)
INVARIANTS: never writes into the release root (no lgbm_* overwrite); refuses an existing target parquet or
  an existing delta.json; delta.json is computed from val rows only; seed = 20261025 + k;
  train_report_lgbm_seeds.json drops the test metrics that run_model computes (lgbm.py:441): no key
  containing "test"
ACCEPTANCE: synthetic release, 2 seeds: two parquets in lgbm_seeds/, release-root lgbm_* sha unchanged;
  delta.json identical when test probabilities are replaced by NaN; report has no key containing "test"
RISK: CPU time of 10 full fits (not measured, H-D6)
```

```
FILE: src/astro_triage/models/dl_data.py                    [create, target <= 190 lines]
INTENT: train-only preprocessing (quantile, NaN indicators, solved code) and tensors per fold
EXPORTS:
  class Preprocessor:
      def __init__(self, features: tuple[str, ...]) -> None
      def fit(self, train: pd.DataFrame) -> "Preprocessor"
      def transform(self, frame: pd.DataFrame) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64] | None]
          # (continuous + indicators, solved category index or None for noind)
      n_continuous: int; n_categories: int; indicator_columns: tuple[str, ...]
      def state_sha256(self) -> str
  def save_preprocessor(p: Preprocessor, path: Path) -> None     # temp+rename
  def load_preprocessor(path: Path) -> Preprocessor
  def subsample_index(n_train: int, n: int, seed: int = 20261025) -> npt.NDArray[np.int64]
      # plan C: first n of default_rng(seed).permutation(n_train), sorted; shared with LightGBM seeds
DEPENDS: numpy, pandas, sklearn.preprocessing.QuantileTransformer, pickle, hashlib, os, tempfile, pathlib,
  astro_triage.models.lgbm (TrainError, INDICATOR_COLUMN) (closed list)
INVARIANTS: fit sees train rows only; indicators decided from train NULLs only; transform never refits;
  a NULL in a feature without indicator -> TrainError; unseen solved code -> TrainError; output finite;
  columns read by name; no torch import; (amendment A) quantile-fit noise N(0, 1e-3) on train non-null
  cells only, from default_rng(20261025), never applied in transform
ACCEPTANCE: fit on train, transform(val) twice -> identical arrays; a NaN only present in val in a feature
  without indicator -> TrainError; noind set yields no solved categories and no astrometric indicator; sha
  stable across save/load; subsample_index deterministic and within range; (amendment A) two fits on the
  same train give the same state sha (deterministic noise) and transform(train) is not the noisy input
RISK: QuantileTransformer on 1.4M x 21 memory; constant columns (quantile edge collapse)
```

```
FILE: src/astro_triage/models/dl_nets.py                    [create, target <= 180 lines]
INTENT: MLP and FT-Transformer modules returning logits in CLASSES order
EXPORTS:
  MLP_CONFIG: Final[dict[str, object]]; FTT_CONFIG: Final[dict[str, object]]   # section 3.7
  class MLP(torch.nn.Module): def __init__(self, n_cont: int, n_cat: int, config: Mapping[str, object]); forward(x_cont, x_cat) -> Tensor
  class FTTransformer(torch.nn.Module): same signature   # PreNorm, no attention-input norm in block 0, Kaiming init (amendment A)
  def build_model(arch: Literal["mlp", "ftt"], n_cont: int, n_cat: int) -> torch.nn.Module
  def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]
      # amendment A: weight_decay 0.0 for Feature Tokenizer, [CLS] token embedding (project choice: not
      # settled by the paper text; reference code (rtdl) not consulted), LayerNorm and bias parameters (FTT)
DEPENDS: torch, typing, collections.abc (closed list)
INVARIANTS: output shape (batch, 3); no softmax inside; configs are the only hyperparameters
ACCEPTANCE: forward on random (8, n) input -> (8, 3) finite for both archs, with and without categories;
  parameter count logged; for FTT every tokenizer, [CLS] embedding, LayerNorm and bias parameter is in a
  group with weight_decay 0.0 (the [CLS] embedding asserted explicitly) and every other parameter in the
  group with 1e-5, each parameter exactly once; FTT block 0 has NO normalisation on its attention input
  while blocks 1.. have one
RISK: ReGLU dimension arithmetic; parameter groups mis-assigned (tokenizer / [CLS] / LayerNorm / bias)
```

```
FILE: src/astro_triage/models/dl_train.py                   [create, target <= 300 lines]
INTENT: fit one (arch, feature set, seed) with early stopping on val, checkpoint per epoch, calibrate on
  val, write predictions + manifest + report entry
EXPORTS:
  @dataclass(frozen=True) class DLSpec: arch: str; feature_set: Literal["full", "noind"]; seed_index: int
      @property model_id -> str        # f"{arch}_{feature_set}_s{seed_index}"
  def train_one(spec: DLSpec, folds: dict[str, Fold], outdir: Path, ckpt_dir: Path, *, device: str,
                max_epochs: int, patience: int, train_subset: int | None = None) -> dict[str, object]
DEPENDS: torch, numpy, pandas, pyarrow, pyarrow.parquet, json, time, os, resource (Linux only, guarded),
  astro_triage.models.{dl_data,dl_nets}, astro_triage.models.lgbm (load_folds, Fold, fit_temperature,
  predictions_table, MODEL_FULL, MODEL_NOIND, TrainError, _atomic, _manifest, _git_sha) (closed list)
INVARIANTS: test rows used only after weights and T are frozen, for the predictions table; returned dict
  holds val metrics and costs only; unweighted CE; seeds per section 7; resume from last.pt reproduces the
  epoch counter and patience; refuses a checkpoint with another config/preprocessor sha; optimizer built
  from dl_nets.parameter_groups (amendment A)
ACCEPTANCE: 300-row synthetic release, 2 epochs CPU: parquet schema equals predictions_table schema;
  p_cal rows sum to 1; kill after epoch 1 + resume == uninterrupted run (CPU, bit-identical); report has
  no key containing "test"
RISK: importing private helpers of lgbm.py (accepted: guarantees identical schema; if they move, this
  breaks loudly); GPU OOM at batch 1024 for FTT (measured in P-gpu)
```

```
FILE: src/astro_triage/models/ensemble.py                   [create, target <= 90 lines]
INTENT: LightGBM + network mixture weight and temperature, both on val
EXPORTS:
  W_GRID: Final[tuple[float, ...]]    # 0.0..1.0 step 0.1
  def fit_weight(p_lgbm_val: Floats, p_net_val: Floats, y_val: Ints) -> float
  def mix(p_lgbm: Floats, p_net: Floats, w: float) -> Floats
DEPENDS: numpy, astro_triage.models.lgbm (fit_temperature, log_loss) (closed list)
INVARIANTS: never receives test rows for fitting; ties -> larger w; output rows sum to 1
ACCEPTANCE: identical inputs -> w = 1.0; a perfect net on val -> w = 0.0; sums within 1e-9
RISK: log of zero probability (clip at 1e-12 before log)
```

```
FILE: scripts/train_dl.py                                   [create, target <= 130 lines]
INTENT: CLI: --outdir --arch {mlp,ftt} --features {full,noind} --seeds 0-4 --max-epochs --patience
  (default 16) --device --ckpt-dir --train-subset (pilot) --plan-c-subsample N --ensemble (build ens_*
  from finished nets and lgbm_full_s<k>) --freeze (write val_freeze.json via dl_compare.build_freeze,
  requires --approval-text)
DEPENDS: argparse, json, sys, pathlib, astro_triage.models.{dl_train,ensemble,dl_data},
  astro_triage.evaluation.dl_compare (build_freeze)
INVARIANTS: --train-subset runs write under dl/pilot/ and never into dl/; --plan-c-subsample writes under
  dl_sub<N>/ only; --freeze refuses an existing freeze, a missing planned model or a missing delta.json;
  exit 1 on TrainError; (amendment A) with --device cuda it prints torch.__version__, torch.version.cuda
  and the GPU name before any training, records them in environment.json and the report, and exits 1 with
  a clear message on a P100 (sm_60) or when torch.cuda.get_arch_list() lacks the device's architecture
ACCEPTANCE: pilot run leaves dl/ without predictions_*; freeze lists one sha per planned model (DL and
  LightGBM seeds), the delta.json sha and every threshold; the GPU check, with a faked device name
  "Tesla P100-PCIE-16GB" and capability (6, 0), exits 1 before any training step
RISK: CLI drift from the Kaggle runner (runner calls this script, no duplicate logic)
```

```
FILE: src/astro_triage/evaluation/dl_compare.py             [create, target <= 340 lines]
INTENT: purity at fixed completeness, delta, block bootstrap and the section 4.4 verdict on test
EXPORTS:
  constants of section 15 header
  def threshold_at_completeness(y: StrArr, score: FloatArr, cls: str, completeness: float) -> float   # F1
  def weighted_purity(y: StrArr, score: FloatArr, cls: str, t: float, prior: Mapping[str, float]) -> float  # F2, nan if empty
  def contaminant_counts(y: StrArr, score: FloatArr, cls: str, t: float) -> dict[str, int]   # raw, per other class
  def is_low_count(counts: Mapping[str, int]) -> bool         # sum(counts.values()) < 10 (total reading)
  def compute_delta(outdir: Path, lgbm_dir: Path) -> dict[str, object]              # F5, val rows only
  def build_freeze(outdir: Path, dl_dir: Path, lgbm_dir: Path, approval: str) -> dict[str, object]  # val rows only
  def block_ids(pixel5: NDArray[np.int64]) -> NDArray[np.int64]                     # pixel5 // BLOCK_DIVISOR
  def load_dl_systems(outdir: Path, ts: runner.TestSet) -> dict[str, runner.ModelData]
  def e_sq(y: StrArr, pred: StrArr, idx: NDArray[np.int64]) -> float
  def run_compare(outdir: Path, n_boot: int = M.N_BOOT, workers: int = 4) -> dict[str, object]
  def class_verdict(d: float, lo: float, hi: float, delta: float, reportable: bool) -> str   # "win"|"worse"|"tie"
  def pair_verdict(per_class: Mapping[str, str]) -> str        # "X wins" | "tie: LightGBM stays"
  def risk_coverage_curve(y: StrArr, proba: FloatArr) -> list[dict[str, float]]
  def render_markdown(res: dict) -> str
DEPENDS: numpy, pandas, json, hashlib, math, concurrent.futures, pathlib,
  astro_triage.evaluation.runner (load_test_set, ModelData, make_cell, variant_proba, GRID, EvalError),
  eval.metrics as M (bootstrap_pixel_draw, floor_status, N_BOOT, SEED), eval.reweight (reweight)
  (closed list)
INVARIANTS: thresholds and delta read from val_freeze.json / delta.json, never refitted on test;
  compute_delta and build_freeze read val rows only; refuses without val_freeze.json or on sha mismatch;
  DL pi_src must equal LightGBM's; same replicates for all systems and seeds (one rng =
  default_rng(M.SEED) per population); purity averaged over seeds, probabilities never; pairs only within
  a front; a non-reportable cell is a tie; std uses ddof = DELTA_DDOF; only HEADLINE_PAIRS get a verdict,
  DESCRIPTIVE_PAIRS are printed without one; realised test completeness and per-class contaminant counts
  are printed and never read by class_verdict
ACCEPTANCE: synthetic release where every DL parquet is a copy of the LightGBM seed of the same k -> D = 0
  and "tie: LightGBM stays"; D > delta with lo > 0 in QSO only -> tie; in both -> "X wins"; D > 0 but
  D <= delta with lo > 0 -> tie; GALAXY below floor or low-count -> tie with the reason printed;
  is_low_count({"STAR": 4, "QSO": 5}) is True and is_low_count({"STAR": 0, "QSO": 12}) is False (absent
  contaminant class does not trigger); an ensemble pair built to beat LightGBM by far more than delta on
  both classes -> no verdict field, result listed as descriptive; threshold_at_completeness on a hand list
  gives the k-th largest with k = ceil(C* n); weighted_purity equals ADR 0003 case 7 (0.1599 +-1e-4) at the
  matching threshold; compute_delta equals 2 * numpy.std(v, ddof=1) on hand values; compute_delta and
  build_freeze unchanged when test probabilities are NaN; block_ids(7207) == 450; e_sq on a hand
  confusion matrix equals the formula; DL parquet with one missing id -> EvalError
RISK: memory of 30 systems x B=1000 in workers (pass only per-class scores of the population); few blocks
  per class (R14)
```

(Noted 2026-11-04, T-DL-04, no number or constant changes) (i) The delta writer lives in
`scripts/compute_delta.py`, not as `--delta` of `scripts/train_lgbm_seeds.py`: `tests/test_lgbm_seeds.py`
is at its 200-line cap. The `--delta` acceptance item "delta.json identical when test probabilities are
NaN" is covered by `tests/test_dl_compare.py::test_test_rows_never_affect_output`. (ii) `dl_compare.py`
imports `pyarrow` / `pyarrow.parquet` and `astro_triage.split` beyond the closed DEPENDS list:
`pyarrow.parquet` reads labels and predictions with a fold / source_id filter so test rows are never
loaded (`runner.load_test_set` reads test); `split.pixel_l5` and `split.build_pixel_map` check that every
val row lies in a val pixel and give the latitude band of the P3 population. (iii) The 340-line target
cannot hold; the module is split along the val/test boundary WHEN the missing functions are implemented,
not now: `evaluation/dl_compare.py` = constants + pure functions (F1, F2, contaminant_counts,
is_low_count, block_ids, class_verdict, pair_verdict, e_sq, risk_coverage_curve; no IO);
`evaluation/dl_val.py` = val IO only (loaders, `_cell`, compute_delta, build_freeze);
`evaluation/dl_eval.py` = test side (load_dl_systems, run_compare, render_markdown). No re-export from
dl_compare (it would create an import cycle).

```
FILE: scripts/evaluate_dl.py                                [create, target <= 60 lines]
INTENT: CLI: --outdir --n-boot --workers; writes dl/eval/results_dl.{json,md}, risk_coverage_dl.csv
DEPENDS: argparse, sys, pathlib, astro_triage.evaluation.dl_compare
ACCEPTANCE: exit 1 with "dl evaluation refused" for every refusal of section 5.2
```

```
FILE: kaggle/run_kaggle_dl.py                               [create, target <= 220 lines]
INTENT: GPU runner: package, pip (requirements.txt minus PIP_SKIP, never torch), assert torch version,
  restore Phase 1 inputs and earlier DL outputs/checkpoints, run scripts/train_dl.py per planned spec,
  one zip
DEPENDS: stdlib only (as run_kaggle.py)
INVARIANTS: never runs evaluate_dl (bootstrap off GPU); never uploads; skips finished model ids; refuses
  if Phase 1 labels/features/predictions are not found; (amendment A) the GPU session check of section 9,
  item 0 runs (through scripts/train_dl.py) before any training
ACCEPTANCE: dry-run mode on a fake /kaggle tree (tmp dir) lists the plan and exits 0 without training
RISK: Kaggle paths of a notebook-output input (to be confirmed in the interface)
```

```
FILE: requirements-dl.txt                                   [create, 2 lines]
INTENT: -r requirements.txt + torch==<version printed by the first Kaggle/Colab GPU session (section 9,
  item 0); not guessed, not verified yet>
```

```
FILE: tests/test_dl_data.py, tests/test_dl_train.py, tests/test_ensemble.py, tests/test_dl_compare.py,
      tests/test_lgbm_seeds.py                              [create, each <= 200 lines]
INTENT: the ACCEPTANCE items above, CPU only, synthetic data, no network
INVARIANTS: torch-dependent tests use pytest.importorskip("torch"); acceptance of this ADR requires them
  to run (not skip) on the local .venv with requirements-dl.txt installed
```

BP/RP XP: no files (H-D1, out of scope).

## 16. Acceptance criteria (to leave Proposed)

1. H-D2 answered (the remaining open gate; H-D3, H-D4, H-D7 and H-D8 answered by the owner, 2026-11-03);
   FT-Transformer defaults checked by `researcher` (T3; done 2026-11-03, amendment A). The constants of
   H-D8 are frozen from 2026-11-03; any change needs a new ADR or a dated amendment decided before any DL
   result exists.
2. All tests of section 15 pass locally with torch installed (none skipped); `ruff` clean.
3. P-local done; P-gpu done and its measured sec/epoch, peak GPU memory and projection delivered to the
   human; no number in that report is estimated without saying so.
4. `lgbm_seeds/delta.json` exists before the first DL batch; `val_freeze.json` exists before
   `evaluate_dl.py` is run on test for the first time.
5. The DL report gives a verdict only for H1 and H2 (pure MLP, pure FT-Transformer versus LightGBM); the
   ensemble pairs appear as descriptive with no verdict; realised test completeness and per-class
   contaminant counts are printed next to every `D_c`.
6. Reviewer verdict with no BLOCKER/MAJOR open.
