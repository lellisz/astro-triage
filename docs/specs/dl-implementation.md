# Spec T-DL-05: deep-learning axis implementation plan (ADR 0006)

Status: proposed (2026-11-04), architect. Authority: ADR 0006 (sections 3-5, 7-9, 15, 16, amendment A, the two
2026-11-04 notes), `.claude/CONTRACT.md`. **No coder runs before the owner approves this spec.** Every constant
is the ADR's; where the ADR is silent or contradicts itself the item is listed under "ADR gaps" with a
recommended default, not resolved silently.

## 0. Facts read (not assumed)
- torch is **not installed** in `.venv`: no `.venv/Lib/site-packages/torch/__init__.py` (only the
  `array_api_compat/torch` shims inside scipy and sklearn); sklearn and lightgbm are installed. Checked by
  file listing; `python -c "import torch"` was **not run** (architect Bash is limited to graphify).
- Local release `data/interim/full_v0/` has `labels.parquet`, `features.parquet`, the Phase 1 predictions,
  all 10 `lgbm_seeds/predictions_lgbm_{full,noind}_s{0..4}_v0.parquet`, `train_report_lgbm_seeds.json` and
  `lgbm_seeds/delta.json` (ADR §16.4 first half already met).
- `evaluation/dl_compare.py` (312 lines, commit 2eb5b5d) holds constants + pure F1/F2/low-count/block code
  (lines 28-116) and val IO (`_sha256`, `_json_float`, `_load_val_labels`, `_val_scores`, `_cell`,
  `compute_delta`, lines 119-312).
- Importers of `dl_compare`: `scripts/compute_delta.py:16` (`compute_delta`), `tests/test_dl_compare.py:19`
  (`dc`); `tests/test_compute_delta.py:11` imports `make_release, random_data` from `test_dl_compare`.
  Attributes the tests use on `dc`: `threshold_at_completeness, weighted_purity, contaminant_counts,
  is_low_count, block_ids, FEATURE_SETS, N_SEEDS, compute_delta, pq` (the `pq` spy, line 443).
- `tests/test_dl_compare.py` is 536 lines (already above the 200-line habit).
- `lgbm.run_model` already has `seed=` / `model_id=` (ADR §15 lgbm.py item is done); `load_folds` returns
  all three folds with labels in memory; `predictions_table(source_id, z, T)` gives the schema;
  `_atomic`, `_manifest`, `_git_sha`, `fit_temperature`, `log_loss`, `evaluate_split` exist.
- `scripts/train_lgbm_seeds.py` has **no** `--plan-c-subsample` (ADR §15 lists it).
- `runner.load_test_set` reads test rows + Phase 1 predictions; asserts solved codes in {3, 31, 95}.

## 1. Global rules for every slice
- Production code: <= 250 new/changed lines per slice; each new test file <= 200 lines; CPU only; no
  network; synthetic data under `tmp_path`; torch tests start with `pytest.importorskip("torch")`.
- Output paths: every CLI resolves `--outdir` and every derived dir and refuses (exit 1) unless they are
  `is_relative_to((REPO_ROOT / "data").resolve())` (pattern: `compute_delta.py:28-42`). Tests monkeypatch
  `REPO_ROOT` as `tests/test_compute_delta.py` does.
- Refuse-overwrite: a target parquet / model / report entry / `delta.json` / `val_freeze.json` that exists
  -> error naming it (pattern: `train_lgbm_seeds.run` clash list); single-shot JSON written via
  `mkstemp` + `os.link` (no overwrite race), rolling files via `_atomic` (temp + `os.replace`).
- No report key containing "test" (pattern: `train_lgbm_seeds._has_test_key`, duplicated, not imported).
- Errors: training side `lgbm.TrainError`; evaluation side `runner.EvalError`; CLIs map them (and
  `OSError, ValueError, KeyError, pa.ArrowException`) to exit 1 with a one-line message.
- **Leakage invariants (security, every slice):** QuantileTransformer, indicator set and solved-code
  vocabulary are fitted on TRAIN rows only; val is used only for early stopping, temperature, ensemble
  weight and thresholds; no test label is used by any computation and no test-derived metric is computed,
  written or printed before `val_freeze.json` exists; test *features* enter only the forward pass of a
  frozen model to fill the predictions table (ADR §5.1). `scripts/train_dl.py` hands `train_one` a
  label-stripped test fold (see S8).
- Seeds: `seed = 20261025 + k`, k = 0..4 (`lgbm.SEED + k`).

## 2. Slices (dependency order)

Track E (evaluation, no torch): S1 -> S2. Track T (training): S3 (no torch) -> S4 -> S5 -> S6 -> S7 -> S8.
Join: S9 -> S10 -> S11 -> S12 -> S13. Kaggle: S14. Conditional: S15 (plan C), S16 (pin, after P-gpu).
Torch-dependent slices (S4-S6, S8 tests, S10 tests) cannot be **run** before the owner approves the local
torch install (A1, section 6); they can be written and reviewed.

### S1 — split `dl_compare` (pure refactor, no new behaviour). Moment: first slice, before any missing
function is written (ADR note iii: "WHEN the missing functions are implemented"); green baseline required.
```
FILE: src/astro_triage/evaluation/dl_compare.py   [modify]
INTENT: constants + pure functions only, no IO.
KEEPS: all constants of lines 28-63 except LABEL_COLUMNS and DELTA_FORMULA; threshold_at_completeness,
  weighted_purity, contaminant_counts, is_low_count, block_ids; _purity_from_weights renamed
  purity_from_weights (public: dl_val uses it; no test references the old name).
DEPENDS: math, collections.abc, typing, numpy, numpy.typing, runner (GRID, EvalError, FloatArr, StrArr),
  eval.metrics as M, eval.reweight (closed). No pyarrow, no pathlib, no hashlib, no split.
FILE: src/astro_triage/evaluation/dl_val.py   [create, ~200 lines = moved code]
INTENT: val-fold IO: compute_delta (unchanged behaviour), later build_freeze (S10).
EXPORTS: LABEL_COLUMNS, DELTA_FORMULA, sha256_file(path) (was _sha256), json_float (was _json_float),
  load_val_labels(outdir) (was _load_val_labels), val_scores(path, val_ids) (was _val_scores),
  _cell(...) (private, unchanged), compute_delta(outdir, lgbm_dir) -> dict (byte-identical output).
DEPENDS: hashlib, math, pathlib, numpy, pandas, pyarrow, pyarrow.parquet (module attribute `pq` kept: the
  test spy patches `dv.pq.read_table`), split (build_pixel_map, pixel_l5), eval.metrics as M,
  eval.reweight, runner (GRID, EvalError, FloatArr, StrArr), evaluation.dl_compare (constants and pure
  functions) (closed).
FILE: scripts/compute_delta.py   [modify, 1 line]  import -> from astro_triage.evaluation.dl_val import compute_delta
INVARIANTS: dl_compare never imports dl_val/dl_eval; no re-export (ADR note iii: cycle); delta.json
  produced from the same inputs is byte-identical before and after.
```
Tester (same slice, import-only edits, no assertion changes): in `tests/test_dl_compare.py` add
`from astro_triage.evaluation import dl_val as dv`; replace the 13 `dc.compute_delta` sites by
`dv.compute_delta` and `dc.pq` (line 443) by `dv.pq`. `tests/test_compute_delta.py` unchanged (it patches
the script's own `compute_delta` name). New `tests/test_dl_split.py` (<= 60 lines): `dl_compare` source
imports neither `pyarrow` nor `dl_val`; `compute_delta` on the module `release` fixture returns the same
dict as a JSON snapshot taken before the refactor (the tester stores the expected dict from the pre-S1
commit in the test as a sha256 of `json.dumps(doc, sort_keys=True)`, excluding nothing).
ACCEPTANCE: full pytest suite passes with the same number of tests as before S1 (the tester records the
count from `pytest --collect-only -q` before and after; the caller's figure "95" is not verified here);
`ruff` clean; `grep -n "import" dl_compare.py` shows no IO module.
RISK: missing a monkeypatch site -> spy silently stops spying (test 433 would then fail its
`len(label_calls) == 1` assertion, so it is loud).

### S2 — remaining pure functions in `dl_compare`
```
EXPORTS (add):
  def class_verdict(d: float, lo: float, hi: float, delta: float, reportable: bool) -> str
      # "win" iff reportable and d > delta and lo > 0; "worse" iff reportable and d < -delta and hi < 0;
      # else "tie"; any non-finite input with reportable=True -> "tie"
  def pair_verdict(per_class: Mapping[str, str]) -> str
      # "X wins" iff per_class[c] == "win" for every c in TARGET_CLASSES, else "tie: LightGBM stays";
      # missing class key -> EvalError
  def e_sq(y: StrArr, pred: StrArr, idx: npt.NDArray[np.int64]) -> float
      # BALANCED row-normalised: (R[STAR->QSO] + R[QSO->STAR]) / 2 on rows idx; nan if STAR or QSO absent
  def risk_coverage_curve(y: StrArr, proba: FloatArr) -> list[dict[str, float]]
      # tau = 0.50..0.99 step 0.01 built as np.round(np.arange(50, 100) / 100, 2); confidence = max p_cal;
      # per tau: {"tau", "coverage", "risk" (error rate on kept rows, nan if none kept), "n_kept"}
```
DEPENDS: unchanged from S1. INVARIANTS: no IO; `class_verdict` never reads completeness or counts.
Tests `tests/test_dl_verdict.py` (<= 150 lines), hand-computed: d=0.03, delta=0.02, lo=0.001 -> win;
d=0.02 (= delta) -> tie; lo=0 -> tie; reportable False with huge d -> tie; d=-0.03, hi=-0.001 -> worse;
pair: {"QSO": "win", "GALAXY": "tie"} -> tie; both win -> "X wins"; e_sq on a 3x3 hand confusion (e.g.
STAR 10 rows with 2 -> QSO, QSO 5 rows with 1 -> STAR => (0.2 + 0.2)/2 = 0.2); risk_coverage on 4 hand
rows; tau grid has 50 values, first 0.5, last 0.99, no float drift (`0.57 in taus`).

### S3 — `models/dl_data.py` (no torch)
```
FILE: src/astro_triage/models/dl_data.py   [create, <= 190 lines]
EXPORTS (ADR §15, unchanged): Preprocessor(features).fit(train_df) / .transform(frame)
  -> (X_cont float32 (n, n_continuous), x_cat int64 (n,) or None); attributes n_continuous, n_categories,
  indicator_columns, continuous_columns, categories (tuple[int, ...]); state_sha256();
  save_preprocessor(p, path); load_preprocessor(path); subsample_index(n_train, n, seed=20261025).
CONSTANTS: QT_N_QUANTILES = 1000, QT_SUBSAMPLE = 1_000_000, QT_RANDOM_STATE = 20261025,
  FIT_NOISE_SD = 1e-3, FIT_NOISE_SEED = 20261025.
DATA FLOW fit(train):
  continuous = features minus INDICATOR_COLUMN (order of `features`)
  X = train[continuous].to_numpy(float64); null = isnan(X)
  indicator_columns = continuous names with null.any() (train only)
  noisy = X.copy(); nn = ~null; noisy[nn] += default_rng(FIT_NOISE_SEED).normal(0.0, FIT_NOISE_SD, nn.sum())
  qt = QuantileTransformer(output_distribution="normal", n_quantiles=QT_N_QUANTILES,
       subsample=QT_SUBSAMPLE, random_state=QT_RANDOM_STATE).fit(noisy)   # sklearn ignores NaN in fit
  if INDICATOR_COLUMN in features: categories = sorted unique train codes; any NULL code -> TrainError
transform(frame):
  null in a continuous column not in indicator_columns -> TrainError (names the column)
  Z = qt.transform(X) (noise-free X); Z[null] = 0.0; append null[:, indicator] as float32 columns
  x_cat = index of code in categories; unseen or NULL code -> TrainError; noind -> None
  non-finite output -> TrainError
state_sha256: sha256 of json.dumps({features, continuous, indicators, categories, n_quantiles_,
  noise sd/seed}, sort_keys=True).encode() + quantiles_.astype("<f8").tobytes()
  + references_.astype("<f8").tobytes()
save/load: pickle via lgbm._atomic; load only from a path the caller already contained in data/ (or the
  Kaggle working dir) — pickle executes code; never load a .pkl from an untrusted input.
subsample_index: n <= 0 or n > n_train -> TrainError; np.sort(default_rng(seed).permutation(n_train)[:n])
DEPENDS: numpy, pandas, sklearn.preprocessing.QuantileTransformer, pickle, hashlib, json, pathlib,
  astro_triage.models.lgbm (TrainError, INDICATOR_COLUMN, _atomic) (closed; `_atomic` replaces ADR's
  os/tempfile, same behaviour).
```
Tests `tests/test_dl_data.py` (<= 200 lines; ADR §15 items plus): fit on a 300-row train with a NaN column
-> indicator present; val with NaN in a column without indicator -> TrainError; transform(val) twice
identical; two independent fits same sha (deterministic noise); `transform(train)` differs from the noisy
input and NaN cells are 0.0; noind features -> `x_cat is None`, no astrometric indicator; unseen code 5 ->
TrainError; save/load round-trip same sha and same arrays; subsample_index deterministic, sorted, unique,
in range, errors on 0 and n_train+1. **Leakage test:** fit on train, then mutate val/test frames (scale x10,
add NaN to an indicator column) and assert `state_sha256()` unchanged and `fit` was never called with them
(spy on `QuantileTransformer.fit` records the row count == n_train).
RISK: n_quantiles > rows in tiny tests (sklearn is expected to cap it with a warning; not verified here:
tests filter that warning explicitly, never globally); constant column (quantiles collapse; must stay finite).

### S4 — `models/dl_nets.py` (torch)
```
FILE: src/astro_triage/models/dl_nets.py   [create, <= 180 lines]
EXPORTS (ADR §15): MLP_CONFIG, FTT_CONFIG (Final dicts), MLP, FTTransformer, build_model, parameter_groups.
MLP_CONFIG = {"n_layers": 3, "d_hidden": 512, "activation": "relu", "dropout": 0.1, "lr": 1e-3,
  "weight_decay": 1e-5, "batch_size": 1024}
FTT_CONFIG = {"n_blocks": 3, "d_token": 192, "n_heads": 8, "ffn_factor": 4/3, "activation": "reglu",
  "attention_dropout": 0.2, "ffn_dropout": 0.1, "residual_dropout": 0.0, "lr": 1e-4,
  "weight_decay": 1e-5, "batch_size": 1024, "norm": "prenorm", "first_block_attention_norm": False,
  "init": "kaiming"}
MLP.forward(x_cont, x_cat): input = cat(x_cont, one_hot(x_cat, n_cat).float()) (no one-hot if n_cat == 0);
  3 x [Linear -> ReLU -> Dropout(0.1)] -> Linear(512, 3).
FTTransformer: tokenizer: numeric token_j = x_j * W_j + b_j (W, b: (n_cont, d)); categorical token =
  Embedding(n_cat, d)(x_cat) + b_cat; learned [CLS] (1, d) appended last; block i:
  h = x + drop_res(MHA(norm_a_i(x))) with norm_a_0 = Identity; x = h + drop_res(FFN(norm_f_i(h)));
  FFN = Linear(d, 2*256) -> ReGLU (a * relu(b)) -> Dropout(0.1) -> Linear(256, d) (256 = 192 * 4/3);
  MHA = nn.MultiheadAttention(d, 8, dropout=0.2, batch_first=True); head on the [CLS] output:
  LayerNorm -> ReLU -> Linear(d, 3) (see gap G4).
parameter_groups(model, wd): FTTransformer -> [{"params": decay, "weight_decay": wd},
  {"params": no_decay, "weight_decay": 0.0}] where no_decay = every parameter of the tokenizer, the [CLS]
  parameter, every LayerNorm parameter and every parameter whose name ends with "bias"; MLP -> one group
  with wd (ADR §3.7 gives the MLP no exemption). Each parameter exactly once.
DEPENDS: math, torch, typing, collections.abc (closed).
INVARIANTS: output (batch, 3) logits in CLASSES order, no softmax; configs are the only hyperparameters.
```
Tests `tests/test_dl_nets.py` (<= 160 lines, importorskip): (8, n) random -> (8, 3) finite for both archs,
with and without categories; FTT groups as ADR §15 acceptance (the `cls` parameter asserted by identity);
union of groups == `set(model.parameters())`, no duplicates; FTT block 0 attention norm is `nn.Identity`,
blocks 1-2 are `nn.LayerNorm`; ReGLU hidden = 256; parameter count is a positive int and is logged;
`torch.manual_seed(s)` twice -> identical initial `state_dict`.
RISK: ReGLU split dimension; name-based bias matching catching a non-bias parameter.

### S5 — `models/dl_train.py` part 1: seeding, loop, checkpoint, resume (torch)
```
EXPORTS (part 1):
  @dataclass(frozen=True) class DLSpec: arch: Literal["mlp", "ftt"]; feature_set: Literal["full", "noind"];
      seed_index: int;  model_id -> f"{arch}_{feature_set}_s{seed_index}"; seed -> lgbm.SEED + seed_index
  def set_seeds(seed: int) -> torch.Generator     # random, numpy, torch (+cuda) seeds;
      # torch.use_deterministic_algorithms(True, warn_only=True); returns the shuffle generator
  def compat_key(spec, config, patience, preproc_sha, train_ids_sha, device_type, torch_version) -> dict
  def fit_network(model, opt, tensors, *, ckpt: Path, compat: dict, max_epochs: int, patience: int,
                  device: str, gen: torch.Generator) -> dict[str, object]
      # returns best_epoch, epochs_run, best_val_logloss, epoch_seconds list; best weights loaded
  def _on_epoch_end(epoch: int) -> None    # no-op hook, called after last.pt is written (test kill point)
DATA FLOW per epoch: perm = torch.randperm(n_train, generator=gen); batches of config batch_size
  (last partial batch kept); unweighted nn.CrossEntropyLoss; val logloss in eval mode, no_grad, inference
  batch 8192; improved iff val < best (strict) -> best.pt; else counter += 1; stop when counter >= patience
  or epoch == max_epochs; last.pt = {model, optimizer, epoch, best_val_loss, best_epoch, counter,
  rng: {torch_cpu, torch_cuda (list|None), shuffle: gen.get_state()}, epoch_seconds, compat}
  written by temp + os.replace every epoch.
RESUME: last.pt present -> torch.load(weights_only=True) (only tensors/ints/floats/str/dicts are stored, so
  no pickle code runs); compat != stored compat -> TrainError naming the differing keys; restore all
  states; continue at epoch + 1. compat EXCLUDES max_epochs (cost bound, may be lowered, ADR §3.7).
DEPENDS (whole file, closed, = ADR §15): torch, numpy, pandas, pyarrow, pyarrow.parquet, json, time, os,
  random, resource (Linux, guarded import), astro_triage.models.{dl_data, dl_nets}, astro_triage.models.lgbm
  (load_folds, Fold, fit_temperature, predictions_table, evaluate_split, MODEL_FULL, MODEL_NOIND,
  TrainError, SEED, MODEL_VERSION, _atomic, _manifest, _git_sha). (`random`, `evaluate_split`, `SEED`,
  `MODEL_VERSION` added: gap G10.)
```
Tests `tests/test_dl_train_loop.py` (<= 200 lines, importorskip, 300-row synthetic tensors, CPU):
2-epoch run twice -> identical best state_dict (bit); kill via monkeypatched `_on_epoch_end` raising at
epoch 1, then rerun -> final state_dict and epoch counter bit-identical to the uninterrupted run; a last.pt
with another preproc sha / seed / patience -> TrainError; changed max_epochs resumes; patience=1 with a
val loss forced up (monkeypatched val function) stops after 2 epochs and restores epoch-1 weights;
`torch.load` called with `weights_only=True` (spy).

### S6 — `models/dl_train.py` part 2: `train_one`
```
def train_one(spec: DLSpec, folds: dict[str, Fold], outdir: Path, ckpt_dir: Path, *, device: str,
              max_epochs: int, patience: int, train_subset: int | None = None) -> dict[str, object]
DATA FLOW: train fold (optionally rows subsample_index(n_train, train_subset)) -> Preprocessor.fit ->
  tensors(train, val) -> set_seeds -> build_model -> AdamW(parameter_groups(model, cfg wd), lr) ->
  fit_network -> z_val -> T = fit_temperature(z_val, val.labels) -> (weights and T frozen) ->
  z_test = forward(test features) -> predictions_table(ids val+test, z, T) -> parquet sorted by
  source_id, zstd, metadata = run_model's keys (with model_id) + family, arch_config (json), seed,
  best_epoch, epochs_run, temperature, preprocessor_sha256, torch_version, device, feature_set,
  train_subset -> models/<id>.pt (best state_dict), models/<id>.preproc.pkl -> 2 manifests (_manifest).
RETURNS: model_id, seed, val metrics = evaluate_split(val, z_val, T, pi_src, False), temperature,
  temperature_at_bound, train_class_fractions, beats_prior_logloss (best val logloss < logloss of the
  constant pi_src on val; T2), costs (ADR §8: epoch s mean/max/total, epochs_run, predict s val+test and
  per 1M rows, peak GPU bytes or "not measured", parameter count, checkpoint bytes, device name, peak RSS
  via resource on Linux else "not measured"), prediction_file, model_file. No key containing "test".
INVARIANTS: refuses an existing target parquet/model (TrainError); pi_src == LightGBM's train fractions
  (same rows: asserted when train_subset is None); test fold labels never read (works with a
  label-stripped test fold); returns no array.
```
Tests `tests/test_dl_train.py` (<= 200 lines, importorskip; synthetic release with 300 rows built like
`tests/test_lgbm*.py` fixtures, 2 epochs, CPU): parquet schema == `predictions_table` schema; p_cal rows
sum to 1 (1e-6); ids == val ∪ test sorted; metadata has every ADR §5.1 key; report has no key containing
"test" (recursive); second call refuses overwrite; test fold passed with `labels = -1` and no
`sdss_class` column still works (proves test labels unused); test feature values replaced by other finite
values -> val metrics, T and best_epoch identical (test never influences fitting).

### S7 — `models/ensemble.py`
```
EXPORTS: W_GRID = tuple(round(i / 10, 1) for i in range(11)); TIE_ABS_TOL = 1e-12 (gap G5)
  def fit_weight(p_lgbm_val, p_net_val, y_val) -> float    # scan W_GRID from 1.0 down; replace best only
      # when loss < best - TIE_ABS_TOL; loss = lgbm.log_loss(log(clip(mix, 1e-12)), y)
  def mix(p_lgbm, p_net, w) -> Floats                       # w*p_lgbm + (1-w)*p_net, renormalised rows
  def build_ensemble(outdir: Path, dl_dir: Path, lgbm_dir: Path, net: Literal["mlp", "ftt"], k: int)
      -> dict[str, object]   # reads p_cal of lgbm_full_s<k> and <net>_full_s<k> (val+test), val labels
      # via dl_val.load_val_labels; w on val; T = fit_temperature(log(clip(mix_val)), y_val);
      # predictions_table(ids, log(clip(mix_all)), T) -> dl_dir/predictions_ens_<net>_full_s<k>_v0.parquet
      # metadata: family "ensemble", weight, temperature, component file sha256s, train_class_fractions
      # (both components must agree within 1e-9, else TrainError)
DEPENDS: numpy, json, pathlib, pyarrow, pyarrow.parquet, astro_triage.models.lgbm (fit_temperature, log_loss,
  predictions_table, TrainError, MODEL_VERSION, _atomic, _manifest), astro_triage.evaluation.dl_val
  (load_val_labels, sha256_file) (gap G11).
```
Tests `tests/test_ensemble.py` (<= 160 lines, no torch): ADR acceptance (identical inputs -> 1.0; perfect
net -> 0.0; sums 1e-9); hand case with 3 rows where w = 0.7 is the unique minimum; ties: two w with equal
loss -> larger; build_ensemble on a synthetic release: test-row p replaced by other valid p -> w and T
unchanged; refuses existing target.

### S8 — `scripts/train_dl.py` (training CLI, GPU session check)
```
CLI: --outdir --arch {mlp,ftt} --features {full,noind} --seeds (parse like train_lgbm_seeds) --max-epochs
  (default 100) --patience (default 16) --device {cpu,cuda} --platform {local,kaggle,colab} (required with
  cuda) --ckpt-dir --train-subset N (pilot) --plan-c-subsample N
EXPORTS: main(argv) -> int; check_gpu_session(version: str, cuda: str | None, name: str,
  capability: tuple[int, int], arch_list: list[str], pinned: str | None) -> None (pure, raises TrainError);
  gpu_facts() -> dict (thin torch wrapper); strip_test_labels(fold: Fold) -> Fold.
DIRS: normal -> <outdir>/dl/; --train-subset -> <outdir>/dl/pilot/ (never dl/); --plan-c-subsample N ->
  <outdir>/dl_sub<N>/; ckpt default <dir>/ckpt/; all contained in data/ (or the Kaggle working dir when
  --platform kaggle, see S14).
FLOW: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" before importing torch; device cuda -> print
  torch.__version__, torch.version.cuda, GPU name; check_gpu_session (P100 by name or capability (6, 0),
  or f"sm_{maj}{min}" not in arch_list, or platform pin recorded in <dir>/environment.json differs ->
  exit 1 before any training); record into environment.json {platform: {torch, cuda, gpu, first_seen}}
  -> load_folds -> strip_test_labels(test) -> per (spec): skip if model id is in train_report_dl.json
  with matching parquet sha, else train_one -> add val purity from dl_val.val_purity_summary (S10;
  until then omitted) -> write train_report_dl.json (no "test" key) -> T2: beats_prior_logloss False
  outside pilot -> exit 1 after writing the report entry without predictions (gap G12).
DEPENDS: argparse, json, os, sys, pathlib, astro_triage.models.{dl_train, dl_data, lgbm},
  astro_triage.evaluation.dl_val (S10 functions; gap G1).
```
Tests `tests/test_train_dl_cli.py` (<= 200 lines): check_gpu_session with ("Tesla P100-PCIE-16GB", (6, 0))
-> TrainError; T4 (7, 5) with "sm_75" listed -> ok; arch missing -> TrainError; pin mismatch -> TrainError;
`main` with a faked `gpu_facts` returning P100 exits 1 and `train_one` (monkeypatched to fail) never
called; --train-subset run leaves `dl/` without `predictions_*`; outdir outside data/ -> exit 1;
existing target -> exit 1; report has no "test" key; strip_test_labels drops `sdss_class`, labels all -1.

### S9 — `dl_val.build_freeze` + `val_purity_summary`
```
def val_purity_summary(outdir: Path, pred_path: Path) -> dict[str, object]
    # per class in TARGET_CLASSES, C* in COMPLETENESS_LEVELS: threshold (F1, all val rows) and P2 purity
def build_freeze(outdir: Path, dl_dir: Path, lgbm_dir: Path, approval: str) -> dict[str, object]
PLANNED SYSTEMS: lgbm_{full,noind}_s<k>, {mlp,ftt}_{full,noind}_s<k>, ens_{mlp,ftt}_full_s<k>, k in
  range(N_SEEDS) (40 files). Missing any -> EvalError naming them.
DOC: schema_name "astro_triage.dl_val_freeze", schema_version 1, approval (non-empty, else EvalError),
  labels_sha256, delta_sha256 (lgbm_dir/delta.json), every file sha256, thresholds[system][cls][C*]
  (F1 on all val rows, val ids sorted), train_class_fractions per system (DL vs LightGBM |diff| <= 1e-9 else
  EvalError), constants of ADR §15 header and §4.4 (B = M.N_BOOT, rng seed M.SEED, MIN_BLOCKS, ...).
  delta.json's prediction_sha256 must equal the current LightGBM seed file sha256s (else EvalError:
  seeds refitted after delta).
```
Tests `tests/test_dl_freeze.py` (<= 200 lines; reuse `make_release` from `test_dl_compare`, extended by a
helper that copies lgbm seed files to dl/ names): thresholds equal `threshold_at_completeness` on hand
scores; doc identical when test-row p are NaN / garbage (val only); one planned file missing -> EvalError;
pi_src mismatch -> EvalError; tampered seed file after delta -> EvalError; empty approval -> EvalError.

### S10 — `train_dl.py --ensemble` and `--freeze`
`--ensemble --seeds 0-4` calls `ensemble.build_ensemble` for mlp and ftt; `--freeze --approval-text "..."`
calls `dl_val.build_freeze` and writes `<dir>/val_freeze.json` via mkstemp + os.link (refuses existing,
refuses missing delta.json); val purity added to train_report_dl.json entries. Tests appended to a new
`tests/test_train_dl_freeze.py` (<= 150 lines): freeze lists one sha per planned model + delta sha +
every threshold; second freeze exit 1; missing approval exit 1; pilot dir freeze is written under
`dl/pilot/` only.

### S11 — `evaluation/dl_eval.py` part 1: `load_dl_systems` + refusals
```
def load_freeze(dl_dir: Path) -> dict    # missing -> EvalError("val_freeze.json missing")
def load_dl_systems(outdir: Path, ts: runner.TestSet, *, dl_dir: Path, lgbm_dir: Path)
    -> dict[str, runner.ModelData]
REFUSALS (ADR §5.2, all EvalError): dl_dir path contains a "pilot" component; any freeze sha differs;
  a planned file missing; ids != val ∪ test (val ids from dl_val.load_val_labels, test ids from ts);
  non-finite p_cal or row sum off by > runner.SUM_TOLERANCE; train_class_fractions differs from LightGBM's
  by > 1e-9. (Amended 2026-11-09, owner) the test-exposure marker `dl/test_opened.json` exists -> EvalError("test already opened"), checked before any test read. Aligned on ts.source_id, p_cal read by name.
DEPENDS: json, pathlib, numpy, pandas, pyarrow.parquet, runner (TestSet, ModelData, EvalError,
  SUM_TOLERANCE), evaluation.dl_val (sha256_file, load_val_labels), evaluation.dl_compare (constants).
```
Tests `tests/test_dl_eval_load.py` (<= 200 lines): one test per refusal (message asserted); happy path
aligns rows in ts order; **ordering test**: with no `val_freeze.json`, the spy on `pq.read_table` /
`runner.load_test_set` shows no test read occurs before the refusal (the freeze is checked first).

### S12 — `dl_eval.run_compare` (block bootstrap + verdict)
Populations: P2 and P1 on all test rows, P3 on bands B and C (ADR §4.1, §4.2 note). Per population, one
`rng = default_rng(M.SEED)`; block id = `block_ids(pixel) * 4 + band_index` (as `dl_val._cell`); per
replicate b: `idx = M.bootstrap_pixel_draw(block, band, rng)`; multiplicity `m = bincount(idx, n)`;
for every (system, seed, class, C*) the frozen boolean `selected` vector; per replicate
```
n_k = sum_i m_i [y_i = k];  cnt_k(sel) = sum_i m_i sel_i [y_i = k]
P(sel) = (pi_c / n_c) cnt_c / sum_{k: n_k > 0} (pi_k / n_k) cnt_k      (contaminant with n_k = 0 adds 0, G6)
invalid iff n_c = 0 or any denominator of the pair's systems = 0
D_c = mean_k P_X_s<k> - mean_k P_L_s<k>
```
Never materialise a B x n matrix (peak memory target <= 2 GB per worker). Same replicate stream for every
system and seed. CI = 2.5/97.5 percentiles of valid replicates; invalid share > 1 % -> cell n/a.
Reportable per ADR §4.4 (i)-(v) with delta from `delta.json` ([fs][c]["0.80"]["P2"], and its
`val_checks_pass`); verdict only for HEADLINE_PAIRS via `class_verdict` / `pair_verdict`; DESCRIPTIVE_PAIRS
and fronts (2)-(4) get D, CI, no `verdict` key (G7). Level-5 descriptive CI with its own
`default_rng(M.SEED)`. Printed next to every D_c: realised test completeness per system and seed,
contaminant counts per contaminating class, seed min/median/max of purities and of v_k.
Tests `tests/test_dl_eval_compare.py` (<= 200 lines, tiny synthetic test fold, n_boot 50): the ADR §15
dl_compare acceptance list (copies of LightGBM -> D = 0 and tie; QSO-only win -> tie; both -> "X wins";
0 < D <= delta -> tie; GALAXY below floor / low-count -> tie with reason; ensemble beating by far -> no
verdict field); same seed -> identical results twice; replicates identical across systems (spy); n/a when
> 1 % invalid; thresholds read from the freeze, never refitted (spy on threshold_at_completeness: 0 calls).

### S13 — descriptive metrics, report, `scripts/evaluate_dl.py`
Descriptive per system and seed: macro-F1 / accuracy / per-class metrics / ECE via `runner.flat_metrics`
(seed mean of the metric, never of probabilities), `e_sq`, risk-coverage curve; costs copied from
`train_report_dl.json`. `render_markdown(res)` with the PROVISIONAL header rule, the negative-result sentence
of §4.4 verbatim when H1 and H2 are both ties, the B0 limitation, R17 limitation, non-reportable cells
named. Location: `evaluation/dl_report.py` (pure formatting, gap G19; fallback: inside dl_eval.py).
`scripts/evaluate_dl.py` (<= 60 lines): --outdir --n-boot --workers [--dl-dirname --seeds-dirname];
writes `dl/eval/results_dl.{json,md}`, `risk_coverage_dl.csv` via `_atomic`; refuses existing outputs; (amended 2026-11-09, owner) only after both report files are written successfully, writes the marker `dl/test_opened.json` via `_atomic` (no metrics: timestamp, git sha, freeze sha); a failed run writes no marker; tests: marker present -> exit 1, refused; exception before the report is written -> no marker;
exit 1 "dl evaluation refused: <reason>" on EvalError.
Tests `tests/test_evaluate_dl.py` (<= 200 lines): exit 1 + message for every §5.2 refusal; md contains the
negative sentence on a tie fixture and never a verdict for ensemble rows; `allow_nan=False` JSON.

### S14 — `kaggle/run_kaggle_dl.py` (stdlib only)
Plan stages mirror `run_kaggle.py` seeds mode (log `[stage] start/done`): pacote, pip (requirements.txt
minus PIP_SKIP, never torch), torch assert (via train_dl's session check), inputs (Phase 1 parquets or
rebuild path: see gap G14), restore dl/ + ckpt/ + environment.json from earlier outputs without overwrite,
train per planned spec via `scripts/train_dl.py --platform kaggle --device cuda`, zip
`astro_triage_dl_outputs.zip` (dl/ tree, unfinished checkpoints, environment.json, run_log.txt; refuses
`labels.parquet` / `features.parquet` members). Never runs evaluate_dl, never uploads. `--dry-run` prints
the plan. Tests `tests/test_kaggle_dl.py` (<= 160 lines): dry run on a fake /kaggle tree exits 0 and calls
no subprocess; finished ids skipped; missing Phase 1 inputs -> fail; zip member filter.

### S15 (conditional, only if the owner chooses plan C) — `train_lgbm_seeds.py --plan-c-subsample N`
(writes `lgbm_seeds_sub<N>/`, train Fold rebuilt from `dl_data.subsample_index`) and
`compute_delta.py --seeds-dirname`; tests in a **new** file `tests/test_lgbm_seeds_sub.py` because
`tests/test_lgbm_seeds.py` is at its cap.

### S16 (BLOCKED until P-gpu prints a version) — `requirements-dl.txt`: `-r requirements.txt` +
`torch==<printed version>`. Then rerun all torch tests locally with that pin when a matching win cp312 CPU
wheel exists (ADR §16.2).

## 3. Pilots
**P-local (this machine: 15.9 GB RAM, AMD RX 6600 XT, no CUDA -> `--device cpu`).** Preconditions: A1
(torch install) approved; S1-S10 merged; full suite green with torch present, nothing skipped; ruff clean.
1. `train_dl.py --outdir data/interim/full_v0 --arch {mlp,ftt} --features {full,noind} --seeds 0
   --max-epochs 1 --device cpu --train-subset 10000` (4 runs; 10,000 train rows from
   `subsample_index(n_train, 10000, 20261025)`, full val, val+test predicted) -> `dl/pilot/`.
2. Resume check: `--arch mlp --features noind --seeds 1 --max-epochs 2 --train-subset 10000`, interrupt
   after the "epoch 1" log line, rerun the same command -> it resumes at epoch 2 and finishes.
3. (Corrected 2026-11-09) Not run in P-local: the S7 guard refuses mixtures whose `train_class_fractions`
   differ, and the pilot models use a 10,000-row subsample while `lgbm_full_s0` uses the full train fold.
   The guard is correct and kept. `--ensemble` runs only over full-train-fold models (P-gpu or batch);
   the mixing logic is covered by the S7 tests.
4. Check: every parquet schema equals `predictions_lgbm_full_s0_v0.parquet`'s; ids = val ∪ test;
   metadata keys; `dl/` (outside pilot) has no `predictions_*`; train_report has no "test" key;
   environment recorded. Timings are printed but **not used** (ADR §9.4). `evaluate_dl.py` is **never**
   run on the pilot (S11 refuses `pilot` dirs). Expected RAM: `load_folds` on 2.04M rows + inference
   batches; not measured — report the observed peak if available.

**P-gpu — BLOCKED** until (a) the owner reads the Kaggle GPU accelerator, weekly quota and session length
in the interface (H-D2), (b) the owner says "yes" to each Kaggle write (package dataset version, kernel
create/version) and to each output download, (c) the input path for full_v0 parquets is decided (G14),
(d) S14 merged and its dry run green. Content (ADR §9.4): session check prints/records torch, CUDA, GPU
(aborts on P100/sm_60 or missing arch; this fixes the H-D7 pin); `ftt_full_s0, mlp_full_s0, ftt_noind_s0,
mlp_noind_s0`, 1 epoch on the full train fold, val prediction; measure sec/epoch, peak GPU memory,
inference time; projection `sec/epoch x max_epochs x 5 seeds x 4 systems` (upper bound) next to the quota.
Pilot outputs go to `dl/pilot/`; val numbers are not results.

## 4. Owner approvals required before anything runs
- A1. Install torch (CPU) in the local `.venv` (version: see Q1). No coder test of S4-S6/S8/S10 can run
  before this.
- A2. Every Kaggle write (dataset version of the package, kernel create/version, GPU accelerator) and every
  output download (P-gpu and batch), each with its own "yes".
- A3. Commits of each slice (conventional commits; CLAUDE.md and the CONTRACT forbid committing without
  approval).
- A4. The P-gpu projection -> batch approval: systems, seeds, `max_epochs`, sessions (ADR §9.4).
- A5. `val_freeze.json` approval text (human approval line in the freeze, ADR §5.1).
- A6. Running `evaluate_dl.py` on test (single exposure; ADR §14 reversibility). (Amended 2026-11-09) Enforced on disk: S13 writes `dl/test_opened.json` after a successful report and S11 refuses if it exists; reopening needs the owner to delete it manually and a dated ADR note (the deletion clause is the architect's recommendation, pending owner confirmation).
- A7. Plan B (Colab) or plan C (CPU subsample, N) if Kaggle is short.

## 5. Open questions for the owner (recommended default in brackets)
- Q1. Local torch version before the GPU pin exists. [Install the newest win cp312 CPU wheel available
  (R10 saw 2.9.1+cpu, unverified), record it as "local, not the pin"; reinstall to the pin after P-gpu if a
  matching wheel exists.]
- Q2. Same `.venv` or a separate `.venv-dl`? [Same `.venv` (CLAUDE.md rule); torch is not in
  `requirements.txt`, so Phase 1 is unaffected.]
- Q3. P-local seeds: s0 only, or all 5 seeds so `build_freeze` is exercised end-to-end on real val in
  `dl/pilot/`? [s0 only; freeze covered by S9/S10 tests.]
- Q4. T2 in pilots (1 epoch may not beat the prior logloss). [Recorded only in pilots; exit 1 in batches.]
- Q5. P-gpu output zip: include pilot prediction parquets (test-row probabilities, no labels)? [No: report,
  environment.json and run log only.]
- Q6. Split `tests/test_dl_compare.py` (536 lines) now? [No; record as deferred; new tests go to new files.]
- Q7. G14 input path on Kaggle. [Owner checks whether the Phase 1 notebook output still exists; if not,
  reuse the seeds-mode rebuild + parity stages.]

## 6. ADR gaps (not resolved silently; each needs an ADR 0006 dated note or the owner's OK)
- G1. ADR §15 lists `build_freeze` in `dl_compare` and `train_dl.py`/`evaluate_dl.py` importing
  `dl_compare`; note iii moves val IO to `dl_val` and test IO to `dl_eval`. This spec follows note iii.
- G2. `tests/test_dl_compare.py` already exceeds the 200-line cap (536).
- G3. "Checkpoint whose config differs is refused": config undefined. Spec: compat key = arch config,
  feature set, seed, patience, preprocessor sha, train-id sha, device type, torch version; `max_epochs`
  excluded.
- G4. FT-Transformer prediction head, tokenizer bias and the exact Kaiming variant are not in the ADR
  (amendment A says only "Kaiming"). Recommended: head LayerNorm -> ReLU -> Linear (paper Sec. 3, not
  re-verified by T-RES-DL); tokenizer W, b and [CLS] `kaiming_uniform_(a=sqrt(5))`, `nn.Linear` defaults
  elsewhere. Needs a researcher check or an owner OK before S4.
- G5. Ensemble "ties to the larger w" has no tolerance; floating mixtures of identical inputs differ in the
  last ulp. Spec adds `TIE_ABS_TOL = 1e-12`.
- G6. A bootstrap replicate with a **contaminant** class absent: `reweight` raises on an absent class with
  positive prior; the ADR invalidates only replicates without class c. Spec: absent contaminant adds 0.
- G7. Fronts (3)-(4) (signal effect) have no defined `delta`; spec prints D and CI only, no win/worse.
- G8. ADR §7 "DataLoader generator": spec uses a seeded `torch.Generator` driving `randperm` (no DataLoader).
- G9. Plan C: `train_lgbm_seeds.py --plan-c-subsample` (ADR §15) is not implemented; `compute_delta.py` has no
  subsample dir option; `N_SEEDS = 5` is a constant while R1 allows cutting seeds to N >= 3 (would need an
  amendment). S15 covers the first two only if plan C is chosen.
- G10. `dl_train` DEPENDS adds `random`, `evaluate_split`, `SEED`, `MODEL_VERSION`; val purity in
  `train_report_dl.json` (§5.1) is computed by the script through `dl_val`, not by `dl_train`.
- G11. `ensemble.py` gets IO (`build_ensemble`) and imports `evaluation.dl_val`; ADR lists numpy + lgbm
  only and <= 90 lines (spec estimate ~170).
- G12. T2 does not say whether a failing model's predictions file is written; spec: not written.
- G13. ADR §16.2 needs tests run with `requirements-dl.txt`, whose pin exists only after P-gpu; local tests
  first run on Q1's version, rerun after S16.
- G14. ADR §9.2 assumes the Phase 1 output zip with full_v0 parquets; the seeds-mode zip excludes them
  (`kaggle-seed-run.md`). If the Phase 1 output is gone (R9), S14 must reuse the rebuild + parity stages.
- G15. `lgbm._manifest` hard-codes `release="dataset_100k_v0"`; DL manifests inherit the stale label
  (deferred; not changed here).
- G16. "Test never read before val_freeze": `train_one(folds)` receives the test fold by ADR signature.
  Spec hardening: label-stripped test fold + tests proving test rows never influence fitting.
- G17. NULL `astrometric_params_solved` is not specified; spec raises TrainError (test fold has none,
  `runner.py:270`).
- G18. `dl_eval` would exceed ~500 lines under note iii; spec proposes `evaluation/dl_report.py` (render
  only) as a fourth module (G19 in the slice text).
- G20. (Decided 2026-11-09, owner) Test-exposure marker: S13 writes `dl/test_opened.json` (no metrics) only after the final report is written; S11 refuses to open test if it exists. Recorded in ADR 0006 §17 and A6.
- Status 2026-11-09: G4 (implemented; divergences i-ii pending owner), G10 (list widened) and G15 (deferred) recorded as ADR 0006 §17 notes.

## 7. Not verified here
Test count "95" (needs `pytest --collect-only`, not runnable by the architect); torch import (file listing
only); sklearn capping of `n_quantiles`; CPU inference time of FT-Transformer on 611k rows; Kaggle quota,
GPU type, torch version; Colab anything; `runner.make_cell` percentile convention (S12 uses plain
percentiles unless the coder confirms reuse is equivalent).

## 8. Maintenance cost
~1,450 production lines across 7 modules + 3 scripts + 1 Kaggle runner, ~2,100 test lines in 12 files; re-pin
torch when the Kaggle image changes; deleting the axis = deleting `dl/`, `dl_sub*/`, the modules and tests
listed here (Phase 1 untouched). Reversal: every slice is reversible in < 1 day until `evaluate_dl.py` runs
on test (A6).
