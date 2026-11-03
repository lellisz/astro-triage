# ADR 0006: Deep learning vs LightGBM on the same tabular signal (comparison protocol, pre-registered)

- Status: **Proposed** (draft, 2026-11-02). Becomes Accepted only after the human answers the decisions of
  section 12 that are marked "before training" and the smoke pilot (section 9.4) has been reported.
- Date: 2026-11-02
- Task: T-DL-01
- Related: ADR 0001 (Parquet, single writer, temp+rename), ADR 0002 (labels, cut (c) "same information"),
  ADR 0003 (pixel split, BALANCED/REALISTIC regimes, paired pixel bootstrap, reporting floor, ECE),
  ADR 0004 D7 (missing astrometry is NULL, never imputed in storage; M-full / M-noind ablation),
  ADR 0005 (pre-registered negative-result rule; Kaggle/NVIDIA quota discipline).
- Out of scope: images/CNN (mentioned only, section 13), hyperparameter search beyond the budget of
  section 6, any change to the ADR 0003 split, metrics, floor or seed, any new dependency other than
  `torch`, Layer 2 / LLM, deployment.
- Nothing in this ADR uses a model metric from the **test** fold. Section 1 marks the provenance of every
  number.

## 1. Context

- Research question (IC axis): on the same Gaia DR3 objects, the same SDSS labels (STAR/GALAXY/QSO), the
  same split and the same 21 (M-full) / 11 (M-noind) features, does a deep tabular model beat the existing
  LightGBM, and does a combination of both beat either? A negative result is a valid outcome.
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
  - `src/astro_triage/evaluation/runner.py`: Phase 1 evaluator. `LGBM_MODELS` is hard-coded to
    `("lgbm_full", "lgbm_noind")`; `load_test_set` aligns everything on `source_id`, checks the committed
    pixel map, reads `p_cal_*` by name; `run_boot` draws pixel replicates within latitude bands with a fresh
    `default_rng(20261025)` per task, so two tasks with the same test rows in the same order get the
    **same replicates** (paired). Floor = `eval.metrics.floor_status` (>= 100 objects in >= 20 pixels).
    A Layer 1 risk-coverage curve exists only in `evaluation/layer2.py` (`risk_coverage`, tied to
    `Layer2Inputs`).
  - Phase 1 Kaggle runner (`kaggle/run_kaggle.py`) is CPU, installs `requirements.txt` with pip, and writes
    one `astro_triage_outputs.zip` that contains `full_v0/labels.parquet` and `full_v0/features.parquet`.
- full_v0 sizes (counts only, from `train_report.json`): train 1,433,262; val 318,428. Train class fractions
  STAR 0.380, GALAXY 0.378, QSO 0.242 (= `pi_src`, no weights).
- Facts measured on full_v0 (2,044,246 objects, **covariates only, all folds**, supplied by the
  orchestrator; no model output involved):
  - only **13.3 %** have G < 17.65, the limit below which Gaia DR3 guarantees published XP (BP/RP)
    spectra; by class: STAR 34.3 %, GALAXY 0.1 %, QSO 0.9 %;
  - for fainter sources DR3 publishes XP only for a subset, focused on red sources; the XP fraction among
    faint quasars is **not measured** (`has_xp_continuous` was not downloaded).
- Fact supplied by the orchestrator: 82 % of LightGBM M-full errors are STAR<->QSO. **Fold of origin not
  stated to the architect.** It is used here only as motivation; the secondary metric of section 4 is
  justified independently by the physical limit already recorded in ADR 0005 section 1 (compact
  STAR/QSO overlap in photometry/astrometry), so it would be the same without this number.
- **Anomaly, pending verification (not resolved here)**: ADR 0001 states a G < 20 selection, but the median
  G is 20.9 (GALAXY) and 20.2 (QSO). Lead for whoever checks it: `dataset/build.py` lines 104-105 state that
  the XMatch path has no G < 20 selection. Until checked, every statement in this ADR about magnitude
  ranges refers to the data as built, not to ADR 0001's text.
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
| A3 | Network over BP/RP XP coefficients as the main axis | New signal (spectra) mixed with a new learner, so the effect cannot be attributed; covers 13.3 % of objects and ~0.1 % of galaxies; needs a new data-acquisition path (`xp_continuous_mean_spectrum`, access path not verified). Kept as **optional extra (d)**. |

### 2.2 How the deep model reaches the evaluator

| # | Option | Real cost |
|---|---|---|
| E1 | Modify `evaluation/runner.py` to accept extra model ids | Touches the Phase 1 evaluator whose outputs are already reported; every Phase 1 test must be re-run; couples the IC axis to Phase 1 deletion. |
| E2 | **New evaluator module that reuses runner's loaders and metric primitives, reads DL predictions with the same parquet schema** | ~250 lines; Phase 1 untouched; deleting the DL axis deletes one directory and four modules. Duplicates the small bootstrap loop (needed anyway for the STAR<->QSO metric). |
| E3 | Write DL predictions under the LightGBM file names | Zero code; silently overwrites/aliases Phase 1 artefacts. Rejected (data corruption risk). |

### 2.3 Hyperparameter budget

| # | Option | Real cost |
|---|---|---|
| B0 | **Equal to LightGBM: zero search.** One published/fixed configuration per architecture; val used only for early stopping and T | Cheapest, fair to the existing LightGBM; a DL loss could be blamed on "untuned". |
| B1 | Equal N-trial random search on val for both families (LightGBM retrained with the same N) | N x GPU time (not measured) plus N LightGBM fits; needs a search harness; changes the reference model. |
| B2 | Search only the DL models | Unequal budget; any DL gain is confounded with tuning effort. Rejected. |

### 2.4 Seeds

| # | Option | Real cost |
|---|---|---|
| S1 | One seed | Cheapest; seed variance of a network on this data is unknown, so a single run cannot support a claim. |
| S5 | **Five seeds per DL system**, claim rule over seeds (section 4.4) | 5x training cost (projected after the pilot). LightGBM stays at its one existing seed (its seed variance: not measured; human may add it, section 12). |

## 3. Decision (D1): systems, features, preprocessing

1. Systems (model id = file stem; `s<k>` = seed index, seed = 20261025 + k, k = 0..4):

| id | Family | Features | Role |
|---|---|---|---|
| `lgbm_full`, `lgbm_noind` | LightGBM (existing files, not retrained) | 21 / 11 | reference |
| `mlp_full_s<k>`, `mlp_noind_s<k>` | MLP | 21 / 11 | main axis (a) |
| `ftt_full_s<k>`, `ftt_noind_s<k>` | FT-Transformer | 21 / 11 | main axis (b) |
| `ens_mlp_full_s<k>`, `ens_ftt_full_s<k>` | LightGBM + network (section 3.6) | 21 | extra (c) |
| `xp_*` | network on BP/RP | XP subset | extra (d), **not planned until H-D1 is answered** |

2. Feature sets are exactly `lgbm.MODEL_FULL.features` and `lgbm.MODEL_NOIND.features`, imported, never
   re-listed.
3. **Preprocessing, fitted on train only** (one object per feature set, serialized next to the model):
   - continuous features: `sklearn.preprocessing.QuantileTransformer(output_distribution="normal",
     n_quantiles=1000, subsample=1_000_000, random_state=20261025)`; chosen as in Gorishniy et al. (2021),
     not tuned. StandardScaler is **not** an arm (one preprocessing, no search);
   - `astrometric_params_solved` (M-full only): categorical {3, 31, 95}, one-hot for the MLP, one
     categorical token for the FT-Transformer; unseen code -> `TrainError`;
   - missing values: for every feature of the set with >= 1 NULL/NaN **in train**, a binary indicator
     column `<name>__isna` is appended and the transformed value is set to 0.0 (the median in normal space).
     Indicators are never created from val/test. In M-noind the nine astrometric columns are absent, so no
     astrometric missingness indicator exists there (ADR 0004 D7.4 ablation preserved);
   - the preprocessor stores the train-fitted state and its sha256; the predictions parquet records it.
4. **Loss and labels**: unweighted cross-entropy over `CLASSES` order (`STAR, GALAXY, QSO`); no class
   weights, no resampling, so `pi_src` = train class fractions, identical to LightGBM's (asserted).
5. **Calibration**: one temperature per trained model, fitted on val logits with
   `lgbm.fit_temperature` (same bounds); test never enters a fit. A temperature at a bound is flagged as in
   `lgbm.run_model`.
6. **Ensemble (extra c)**: `p = w * p_cal(lgbm_full) + (1 - w) * p_cal(net_full_s<k>)`, `w` on the grid
   0.0..1.0 step 0.1 minimizing **val** logloss, ties to the larger `w` (prefer the existing model); then
   one temperature fitted on val on `log p`; stored with `p_raw_*` = the mixture before T. No stacking
   model, no other weights.
7. Architectures (fixed, B0):
   - FT-Transformer: the default configuration of Gorishniy et al. (2021) (paper "default" FT-Transformer:
     3 blocks, d_token 192, 8 heads, FFN hidden factor 4/3 with ReGLU, attention dropout 0.2, FFN dropout
     0.1, residual dropout 0.0, AdamW lr 1e-4, weight decay 1e-5). **These values are from the architect's
     memory and must be confirmed by `researcher` against the paper/reference code before the pilot**; if
     they differ, the paper wins and this list is amended before any training.
   - MLP: 3 hidden layers x 512, ReLU, dropout 0.1, AdamW lr 1e-3, weight decay 1e-5. Chosen by the
     architect, not tuned, stated as such.
   - Both: batch 1024, `max_epochs` 100, early stopping on val logloss with patience 10 epochs, best-epoch
     weights restored, no LR schedule. `max_epochs` may be **lowered for all systems at once** by the human
     after the pilot (cost only, decided before any val result of the batch is seen).

## 4. Decision (D2): metrics, hypotheses, criterion of gain (pre-registered)

### 4.1 Population and outputs

All headline numbers: test fold of full_v0, population `a_all` (all test rows), regime BALANCED (P0),
variant **prior_matched** (ADR 0003: conclusions use prior-matched outputs; with identical `pi_src` the
shift is the same for every system). REALISTIC P1-P3 purity and the `astrometric_params_solved` strata are
reported as **descriptive**, with no gain claim.

### 4.2 Metrics

- **Primary**: macro-F1 (mean of per-class F1 under BALANCED weights, `eval.metrics`).
- **Key secondary**: STAR<->QSO confusion `e_SQ = (R[STAR->QSO] + R[QSO->STAR]) / 2`, where `R[a->b]` is the
  BALANCED row-normalized confusion rate (fraction of true `a` predicted `b`). Lower is better. Reported
  with the floor of STAR and QSO only.
- Also reported: accuracy, per-class purity/completeness/F1/PR-AUC, ECE mean (15 equal-mass bins,
  prior_matched), a risk-coverage (abstention) curve per system on the `tau` grid 0.50..0.99 step 0.01 with
  confidence = max `p_cal_*` (descriptive, no claim), costs (section 8).
- Uncertainty: paired pixel bootstrap, B = 1000, `default_rng(20261025)`, within latitude bands, the
  **same replicates** for every system (test rows taken from `runner.load_test_set` in its row order).
  Floor: >= 100 objects in >= 20 pixels per class, else `n/a (n=..., pixels=...)`.

### 4.3 Hypotheses (stated before training)

| id | Hypothesis | Pair(s), first minus second | Expected |
|---|---|---|---|
| H1 | MLP ties LightGBM on the tabular signal | `mlp_full - lgbm_full` | no gain |
| H2 | FT-Transformer ties LightGBM on the tabular signal | `ftt_full - lgbm_full` | no gain |
| H3 | The ensemble may gain over LightGBM (errors partially decorrelated) | `ens_ftt_full - lgbm_full`, `ens_mlp_full - lgbm_full` | possible gain, small |
| H4 | The missing-astrometry pattern helps both families alike | `ftt_full - ftt_noind`, `mlp_full - mlp_noind` next to the existing `lgbm_full - lgbm_noind` | same sign as LightGBM |
| H5 | BP/RP can only help on the XP subset (G < 17.65 plus a faint subset), where GALAXY is ~0.1 % of objects | extra (d) only, restricted population | not testable on macro-F1 (GALAXY below floor expected); only `e_SQ` on that subset |

### 4.4 Criterion of gain (decision rule, fixed now; code constants in section 14)

For a headline pair `X - lgbm_full` (H1, H2, H3), per seed `k`:
- `gain_k` iff the macro-F1 diff cell passes the floor **and** the lower bound of its 95 % paired CI > 0;
- `loss_k` iff the cell passes the floor and the upper bound < 0.

Verdict over seeds: **"X better"** iff `gain_k` for >= 4 of 5 seeds; **"LightGBM better"** iff `loss_k`
for >= 4 of 5 seeds; otherwise **"no measurable difference"**. The same rule, with signs reversed (upper
bound < 0 means fewer STAR<->QSO errors), gives the separate `e_SQ` verdict. Seed variation is reported as
min / median / max of the point diffs and the 5 intervals side by side; probabilities are **never averaged
across seeds** (that would be a deep ensemble, a different system, not planned).

Four fronts, never mixed: (1) model effect with full signal `net_full - lgbm_full`; (2) model effect with
reduced signal `net_noind - lgbm_noind`; (3) signal effect within the network `net_full - net_noind`;
(4) signal effect within LightGBM `lgbm_full - lgbm_noind` (exists). A pair that changes both model and
signal (e.g. `ftt_full - lgbm_noind`) is not computed. Only H1-H3 are headline; fronts (2)-(4) are
descriptive. No multiplicity correction is applied; the report says so.

Negative-result sentence (emitted when H1-H3 are all "no measurable difference" or "LightGBM better"):
"On the same 21 Gaia DR3 features, MLP and FT-Transformer (5 seeds, untuned defaults) add no measurable
macro-F1 over LightGBM; the LightGBM+network ensemble adds none either." No configuration is changed after
test is seen to rescue it; any new attempt is a new `MODEL_VERSION` evaluated on val first and declared as
post-test.

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
   - `train_report_dl.json`: per model val metrics only (logloss before/after T, ECE, macro-F1 on val),
     costs, temperature, best epoch. **It contains no test metric** (unlike `train_report.json`).
   - `val_freeze.json`: written once, after all planned systems exist, listing sha256 of every DL
     predictions parquet, the constants of section 4.4 and the human approval line; refused if it exists.
2. Evaluator input: `outdir` (release dir) + `dl/`. It calls `runner.load_test_set(outdir)` (gives the
   test rows, pixels, bands and the two LightGBM systems) and aligns every DL parquet on that `source_id`
   order, reading `p_cal_*` by name. It refuses (exit 1) when: a DL parquet does not cover exactly the val+test
   ids; probabilities are not finite or do not sum to 1 (1e-3); `train_class_fractions` differs from
   LightGBM's by > 1e-9; `val_freeze.json` is missing or any sha256 differs; a seed listed in the freeze is
   missing.
3. Outputs: `dl/eval/results_dl.json`, `dl/eval/results_dl.md` (header PROVISIONAL as Phase 1 if below
   floor), `dl/eval/risk_coverage_dl.csv`.

## 6. Decision (D4): tuning budget

**B0** (equal to LightGBM: zero search). Val is used only for early stopping, temperature and the ensemble
weight. Changing the budget to B1 is a human decision (H-D3) and must be taken before any batch training;
under B1 LightGBM gets the same number of trials on the same val.

## 7. Decision (D5): seeds and determinism

S5: seeds 20261025 + k, k = 0..4, set for Python, NumPy, torch (CPU and CUDA) and the DataLoader generator.
`torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; GPU
non-determinism that remains is recorded as a limitation, not hidden (bit-identical reruns are **not** an
acceptance criterion on GPU; on CPU they are).

## 8. Decision (D6): costs reported per system

Measured, never estimated in the report: train wall time per epoch (mean, max) and total; epochs run; time
to predict val+test (s) and per 1M rows; peak GPU memory (`torch.cuda.max_memory_allocated`); parameter
count; checkpoint size; device name. Peak host RSS: measured on Linux (Kaggle) via `resource`, "not
measured" on Windows. For LightGBM: training time **not measured**; inference time measured by the
evaluator only if the human asks (H-D6).

## 9. Decision (D7): Kaggle logistics

1. Private kernel(s) named `astro-triage-dl-*`, accelerator GPU (type: not verified), internet on only for
   pip of the package (torch is **not** pip-installed: the preinstalled version is recorded and asserted
   against `requirements-dl.txt`).
2. Inputs: (a) the private package dataset built by `scripts/make_kaggle_package.py` (it already ships all of
   `src/`, `scripts/`, `kaggle/`); (b) the Phase 1 kernel output (`astro_triage_outputs.zip` with
   `full_v0/labels.parquet`, `features.parquet`, `predictions_lgbm_*_v0.parquet`, `train_report.json`) if
   still available as a notebook output — **to be confirmed in the Kaggle interface**; if not, the human
   decides how to supply it (no agent uploads data).
3. Checkpoint per epoch and per seed: `ckpt/<model_id>/last.pt` (model, optimizer, epoch, best val loss,
   patience counter, RNG states, preprocessor sha) overwritten by temp+rename each epoch, `best.pt` on
   improvement. Restart = add the previous version's output as input; finished model ids (parquet present
   and sha recorded) are skipped; a checkpoint whose preprocessor sha or config differs is refused.
4. One output: `astro_triage_dl_outputs.zip` (dl/ tree, checkpoints of unfinished models only,
   `environment.json`, `run_log.txt`); `/kaggle/working` ends with the zip and the log only.
5. The bootstrap evaluation (B = 1000) runs **off the GPU**: locally (CPU, `scripts/evaluate_dl.py`) or in
   a separate CPU kernel.
6. Limits: weekly GPU hours and maximum session length are **not verified**; the plan does not depend on a
   number until the human reads them in the interface.
7. Rules: only resources named `astro-triage-*`, all private; uploads contain code only; never touch other
   datasets of the account. **Every write to Kaggle (dataset/kernel create or version) and every output
   download requires the human's explicit "yes".**

### 9.4 Smoke pilot (before any batch)

- P-local (CPU, no Kaggle): every system on 10,000 train rows drawn with `default_rng(20261025)`, full val,
  1 epoch; correctness only (shapes, files, schema, resume after a kill); timings here are not used.
- P-gpu (Kaggle, after the human's "yes"): `ftt_full_s0`, `mlp_full_s0`, `ftt_noind_s0`, `mlp_noind_s0`,
  **1 epoch on the full train fold**, then val prediction. Measures sec/epoch, peak GPU memory and
  inference time. Its val numbers are not reported as results.
- Projection to the human: `sec/epoch x max_epochs x 5 seeds x 4 systems` as an **upper bound** (early
  stopping makes it lower), next to the GPU quota the human reads. Nothing measured is invented; anything
  not measured is printed as "not measured".
- Batch only after an explicit "yes" to: systems, seeds, `max_epochs`, number of sessions. If the budget is
  short, priority order: `ftt_full`, `mlp_full` (headline), then `*_noind`.

## 10. Risks

| id | Risk | Mitigation / measurement |
|---|---|---|
| R1 | GPU quota/time insufficient for 20 runs (not measured, not verified) | Pilot projection; priority order; seeds may be cut by the human (then rule 4.4 changes to a stated majority before training, e.g. 3 of 3) |
| R2 | Unequal effort masked as model effect (defaults vs hand-set LightGBM) | B0 for both; stated as limitation; B1 is H-D3 |
| R3 | FT-Transformer defaults misremembered | `researcher` verifies before the pilot (handoff) |
| R4 | Missingness indicators let the network read the class through the astrometric NULL pattern | Same information as LightGBM M-full (it sees `astrometric_params_solved`); front (2) with `noind` measures it |
| R5 | QuantileTransformer fit on 1.4M rows: memory/time on Kaggle (not measured) | `subsample=1_000_000`; measured in P-local and P-gpu |
| R6 | GPU non-determinism -> seed runs not reproducible bit-for-bit | Recorded; claims rest on 5 seeds, not on one run |
| R7 | Test seen before all seeds finish, then a "fix" | `val_freeze.json` gate; evaluator refuses without it |
| R8 | BP/RP extra: 0.1 % of galaxies have guaranteed XP; faint-QSO XP fraction not measured; selection by XP availability is class- and colour-dependent | Extra (d) gated by H-D1; first step measures `has_xp_continuous` on train/val only; report restricted to `e_SQ` on the XP subset with the subset's class/G histograms |
| R9 | Phase 1 Kaggle output no longer available as input | Confirm in the interface; human decides the supply path |
| R10 | `torch` version on Kaggle differs from the local pin | Runner records and asserts it; mismatch stops the run before training |
| R11 | Pinning `torch` in `requirements.txt` makes the Phase 1 CPU runner pip-install torch | Pin in `requirements-dl.txt` instead (H-D7) |
| R12 | ADR 0001 G < 20 anomaly: magnitude-dependent statements could be wrong | Pending check (section 1); no claim in this ADR depends on G < 20 |

## 11. Reversal triggers

- T1. P-gpu projection exceeds the human's GPU budget even for `ftt_full` + `mlp_full` with 3 seeds ->
  drop FT-Transformer to `noind`-free headline only, or drop the GPU and run MLP on CPU (human decides).
- T2. On val (before test), a network fails to beat the trivial `pi_src` logloss after `max_epochs` ->
  implementation bug suspected; stop, fix, rerun; no test evaluation of that system.
- T3. Researcher shows FT-Transformer defaults differ from section 3.7 -> amend before training.
- T4. `has_xp_continuous` on train/val shows XP coverage of QSO >= 20 % and of the faint subset not
  colour-confined -> reopen BP/RP as a main-axis candidate in a new ADR (new signal, separate from this one).
- T5. Seed spread of the point macro-F1 diff (max - min) on val exceeds the width of a typical paired CI ->
  the 4-of-5 rule is too weak; raise seeds (human) before test.
- T6. The G < 20 anomaly reveals a data-build error -> stop; rebuild the dataset; all predictions (LightGBM
  included) are redone before any comparison.

## 12. Decisions for the human

- H-D1. Scope of BP/RP (extra d): not at all / measure `has_xp_continuous` only / full extra. Open.
  **Before any XP download.**
- H-D2. GPU: confirm accelerator, weekly quota and session length read in the interface; approve P-gpu.
  Open. **Before P-gpu.**
- H-D3. Tuning budget B0 (proposed) or B1 with N trials for both families. Open. **Before training.**
- H-D4. Number of seeds: 5 (proposed) and the 4-of-5 rule; or fewer, with the majority stated now. Open.
  **Before training.**
- H-D5. Retrain LightGBM with the same 5 seeds to measure its seed variance (CPU cost, not measured). Open.
- H-D6. Measure LightGBM inference/training time for the cost table. Open.
- H-D7. Pin `torch` in a separate `requirements-dl.txt` (proposed, R11) instead of `requirements.txt`.
  Open. Version to pin: the one preinstalled in the Kaggle GPU image, **not verified**; local CPU wheel for
  CPython 3.12 win_amd64 of that version: **not verified** (researcher handoff).
- H-D8. Approve the pre-registered constants of section 4.4 before any test number exists. Open.
- H-D9. Check the ADR 0001 G < 20 anomaly (section 1). Open, independent of this ADR.

## 13. Non-goals

Images/CNN on Gaia or SDSS cutouts (a different signal and a different data path; not planned); deep
ensembles across seeds; hyperparameter search beyond B0/B1; class weights or resampling; any change to
split, seed, floor, bootstrap or metrics of ADR 0003; retraining or overwriting `lgbm_*` files; REALISTIC
headline claims; Layer 2 interaction; publishing any Kaggle resource; evaluating partially finished seeds.

## 14. Consequences

- Good: the DL axis is one more producer of the same predictions schema; every comparison uses the same
  rows, replicates and floor as Phase 1; the gain rule and the negative sentence exist before training;
  deleting the axis = deleting `dl/` and the files of section 15.
- Bad: GPU cost unknown until the pilot; untuned defaults invite the "not tuned" objection; 5 seeds x 4
  systems multiply cost; GPU runs are not bit-reproducible; one more evaluator module partly mirrors
  `runner.run_boot`; the XP extra may be unreportable for GALAXY.
- Maintenance cost: 4 source modules (~700 lines total), 2 scripts, 1 Kaggle runner (~200 lines), 4 test
  files, 1 requirements file; re-pin torch when the Kaggle image changes.
- Reversibility: everything < 1 day **except** test exposure: after `evaluate_dl.py` runs on test, the
  configurations of section 3 cannot be changed and re-evaluated on the same test without declaring it.

## 15. Implementation plan (per file)

Code constants (in `evaluation/dl_compare.py`): `HEADLINE_METRIC = "macro_f1"`,
`SECONDARY_METRIC = "e_sq"`, `HEADLINE_PAIRS = (("mlp_full", "lgbm_full"), ("ftt_full", "lgbm_full"),
("ens_mlp_full", "lgbm_full"), ("ens_ftt_full", "lgbm_full"))`, `SEED_MAJORITY = 4`, `N_SEEDS = 5`.

```
FILE: src/astro_triage/models/dl_data.py                    [create, target <= 180 lines]
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
DEPENDS: numpy, pandas, sklearn.preprocessing.QuantileTransformer, pickle, hashlib, os, tempfile, pathlib,
  astro_triage.models.lgbm (TrainError, INDICATOR_COLUMN) (closed list)
INVARIANTS: fit sees train rows only; indicators decided from train NULLs only; transform never refits;
  unseen solved code -> TrainError; output finite; columns read by name
ACCEPTANCE: fit on train, transform(val) twice -> identical arrays; a NaN only present in val yields 0.0
  and no new column; noind set yields no solved categories and no astrometric indicator; sha stable
  across save/load
RISK: QuantileTransformer on 1.4M x 21 memory; constant columns (quantile edge collapse)
```

```
FILE: src/astro_triage/models/dl_nets.py                    [create, target <= 160 lines]
INTENT: MLP and FT-Transformer modules returning logits in CLASSES order
EXPORTS:
  MLP_CONFIG: Final[dict[str, object]]; FTT_CONFIG: Final[dict[str, object]]   # section 3.7
  class MLP(torch.nn.Module): def __init__(self, n_cont: int, n_cat: int, config: Mapping[str, object]); forward(x_cont, x_cat) -> Tensor
  class FTTransformer(torch.nn.Module): same signature
  def build_model(arch: Literal["mlp", "ftt"], n_cont: int, n_cat: int) -> torch.nn.Module
DEPENDS: torch, typing, collections.abc (closed list)
INVARIANTS: output shape (batch, 3); no softmax inside; configs are the only hyperparameters
ACCEPTANCE: forward on random (8, n) input -> (8, 3) finite for both archs, with and without categories;
  parameter count logged
RISK: FTT defaults misremembered (R3); ReGLU dimension arithmetic
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
  epoch counter and patience; refuses a checkpoint with another config/preprocessor sha
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
FILE: scripts/train_dl.py                                   [create, target <= 100 lines]
INTENT: CLI: --outdir --arch {mlp,ftt} --features {full,noind} --seeds 0-4 --max-epochs --patience
  --device --ckpt-dir --train-subset (pilot) --ensemble (build ens_* from finished nets) --freeze
  (write val_freeze.json, requires --approval-text)
DEPENDS: argparse, json, sys, pathlib, astro_triage.models.{dl_train,ensemble}
INVARIANTS: --train-subset runs write under dl/pilot/ and never into dl/; --freeze refuses an existing
  freeze or a missing planned model; exit 1 on TrainError
ACCEPTANCE: pilot run leaves dl/ without predictions_*; freeze lists one sha per planned model
RISK: CLI drift from the Kaggle runner (runner calls this script, no duplicate logic)
```

```
FILE: src/astro_triage/evaluation/dl_compare.py             [create, target <= 300 lines]
INTENT: paired comparison of DL systems vs LightGBM on test under the section 4 rule
EXPORTS:
  constants of section 15 header
  def load_dl_systems(outdir: Path, ts: runner.TestSet) -> dict[str, runner.ModelData]
  def e_sq(y: StrArr, pred: StrArr, idx: NDArray[np.int64]) -> float
  def run_compare(outdir: Path, n_boot: int = M.N_BOOT, workers: int = 4) -> dict[str, object]
  def seed_verdict(cells: list[dict[str, object]], higher_is_better: bool) -> str
  def risk_coverage_curve(y: StrArr, proba: FloatArr) -> list[dict[str, float]]
  def render_markdown(res: dict) -> str
DEPENDS: numpy, pandas, json, hashlib, math, concurrent.futures, pathlib,
  astro_triage.evaluation.runner (load_test_set, ModelData, make_cell, variant_proba, GRID, EvalError),
  eval.metrics as M (closed list)
INVARIANTS: refuses without val_freeze.json or on sha mismatch; DL pi_src must equal LightGBM's; same
  replicates for all systems (one rng = default_rng(M.SEED) per population); pairs only within a front;
  floor applied per cell; probabilities never averaged across seeds
ACCEPTANCE: synthetic release where a DL parquet is a copy of lgbm_full -> every diff cell point 0 and
  verdict "no measurable difference"; DL parquet with one missing id -> EvalError; 4 seeds with CI > 0
  and 1 without -> "X better"; 3 and 2 -> "no measurable difference"; e_sq on a hand confusion matrix
  equals the formula; same replicate indices as runner.run_boot on the same rows
RISK: memory of 20 systems x B=1000 in workers (pass only pred labels/proba of the population)
```

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
  if Phase 1 labels/features/predictions are not found
ACCEPTANCE: dry-run mode on a fake /kaggle tree (tmp dir) lists the plan and exits 0 without training
RISK: Kaggle paths of a notebook-output input (to be confirmed in the interface)
```

```
FILE: requirements-dl.txt                                   [create, 2 lines]
INTENT: -r requirements.txt + torch==<version preinstalled on the Kaggle GPU image; not verified>
```

```
FILE: tests/test_dl_data.py, tests/test_dl_train.py, tests/test_ensemble.py, tests/test_dl_compare.py
                                                            [create, each <= 200 lines]
INTENT: the ACCEPTANCE items above, CPU only, synthetic data, no network
INVARIANTS: torch-dependent tests use pytest.importorskip("torch"); acceptance of this ADR requires them
  to run (not skip) on the local .venv with requirements-dl.txt installed
```

Extra (d) BP/RP: no files until H-D1. Its first step would be a train/val-only `has_xp_continuous`
measurement; the data path needs a `researcher` report first (access to `xp_continuous_mean_spectrum`
without login: not verified).

## 16. Acceptance criteria (to leave Proposed)

1. H-D2, H-D3, H-D4, H-D7, H-D8 answered; FT-Transformer defaults confirmed by `researcher` (T3).
2. All tests of section 15 pass locally with torch installed (none skipped); `ruff` clean.
3. P-local done; P-gpu done and its measured sec/epoch, peak GPU memory and projection delivered to the
   human; no number in that report is estimated without saying so.
4. `val_freeze.json` exists before `evaluate_dl.py` is run on test for the first time.
5. Reviewer verdict with no BLOCKER/MAJOR open.
