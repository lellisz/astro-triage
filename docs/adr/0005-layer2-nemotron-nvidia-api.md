# ADR 0005: Layer 2 with Nemotron through the NVIDIA API (text only), quota control and evaluation

- Status: **Closed without test evaluation** (Amendment 2, section 18, 2026-11-02). It was never Accepted:
  it was **Proposed** (draft, 2026-10-26; amended 2026-10-27, Amendment 1, section 17) and is closed on the
  descriptive val pilot only; the D7.8 test-fold rule was **not executed** and no D7.8 verdict exists.
  On 2026-10-27 H1, H4 and H8 were answered (section 17.0); on 2026-11-02 H2, H3, H5, H6 and H7 were closed
  as not pursued (section 18.3).
- Date: 2026-10-26 (Amendment 1: 2026-10-27)
- Task: T-L2-01 (Amendment 1: T-L2-02)
- Supersedes (on acceptance, by human edit, not by this ADR): the rule "paid calls (Bedrock, paid Gemini)
  and any AWS command only with explicit human confirmation and a BudgetTracker check before each call
  (total cap US$ 20)" in `.claude/CONTRACT.md` §8, and the backlog items SEC-01 "BudgetTracker ... US$ 20
  cap; Bedrock via SSO only" and "re-add `boto3` when Phase 2 starts". **No accepted ADR (0001-0004)
  contains a dollar cap, AWS, Bedrock or BudgetTracker**, so no ADR is amended. Proposed replacement text
  for CONTRACT §8 is in section 10 (accepted by the human on 2026-10-27, H8; the orchestrator edits the
  CONTRACT, this ADR does not).
- Related: ADR 0001 (Parquet, single writer, temp+rename), ADR 0002 cut (c) "same information",
  ADR 0003 (split, bootstrap, floor, ECE), ADR 0004 D7 (missing astrometry is NULL, never imputed).
- Out of scope: images/VLM, AWS/Bedrock, Gemini, any deployment, REALISTIC-prior regime for Layer 2,
  few-shot prompting, fine-tuning, any commercial use of the API output.
- Precedence: where this text and the implemented code (`src/astro_triage/{llm,cost,abstention}`,
  `src/astro_triage/evaluation/layer2*.py`, `scripts/layer2_*.py`) diverged before 2026-10-27, the code
  prevailed and Amendment 1 records the code's behaviour; where Amendment 1 changes a rule, the amendment
  prevails over both the earlier text and the current code until the code is changed to match it.

## 1. Context

- Cascade: Layer 1 = LightGBM `lgbm_full`, temperature-calibrated on val (`p_cal_star`, `p_cal_qso`,
  `p_cal_galaxy`, mapped by **name**). Layer 2 = LLM on uncertain cases only, structured output with an
  "unknown" option. Layer 3 = abstention ("needs a spectrum").
- User decision 2026-10-26: no AWS. Backend = NVIDIA API, OpenAI-compatible,
  `https://integrate.api.nvidia.com/v1/chat/completions`. Text/table input only. The US$ 20 cap and the
  dollar BudgetTracker are revoked; "cost" is now latency, tokens and request quota.
- Measured on the user's account on 2026-10-26 (free tier, invented prompt, **no accuracy measured**):
  - `GET /v1/models` lists 85 models; listing does not grant access. Chat works for
    `nvidia/nemotron-3-ultra-550b-a55b`, `nvidia/nemotron-3.5-lightning-30b-a3b`,
    `nvidia/nemotron-3.5-bolt-e4b`. HTTP 404 "Function not found for account" for six other Nemotron ids.
  - Default reasoning: ultra ~10 s and ~564 output tokens; lightning spent 1,500 tokens reasoning and
    returned `finish_reason=length`, empty `content`, **HTTP 200** (silent failure).
  - With `"chat_template_kwargs": {"enable_thinking": false}`: ultra 0.9-1.7 s, 18-22 output tokens;
    lightning 0.8-3.2 s, 17-21 tokens; valid JSON 3/3 each. bolt-e4b answers directly: ~0.9 s,
    15 tokens, 3/3.
  - Answer in `content`; reasoning (ultra) in a separate `reasoning_content`. Usage in
    `usage.prompt_tokens`, `usage.completion_tokens`.
  - **No rate-limit headers** are returned. "~40 req/min" on the free tier comes from secondary sources and
    is **not confirmed**. The free tier is aimed at development; terms not read by the implementer (see
    H1, section 17.0: confirmed by the human, not independently verified).
- Volume (100k release, **test fold**, provisional): 6,642 test objects; `max p_cal < 0.9` leaves 756
  (11%) containing 235 of 294 LightGBM errors; LightGBM accuracy on the kept 89% = 0.990. Accuracy 0.9557,
  macro-F1 0.9506, 77% of errors are STAR<->QSO. These numbers come from **test** and are used here only
  to size the problem; **0.9 is not an admissible threshold** (section 4). full_v0 is ~40x larger: sizes
  are estimated as fractions, not counts (order of magnitude in section 17.9).
- Repository facts (2026-10-26): `src/astro_triage/{llm,cost,abstention}` **did not exist** (created by
  this plan; implemented by 2026-10-27). `requirements.txt` pins no HTTP client; `ingest/xmatch.py`
  imports `requests`, which is only a transitive dependency of astroquery (unpinned). `ingest/tap.py` has
  `_sanitize` (secret/Bearer redaction). Predictions live in
  `data/interim/<release>/predictions_lgbm_full_<v>.parquet` and cover val and test (schema P1). The
  dashboard reads `data/interim/<release>/` (`dashboard/data.py`, `load_release`).
- Physical limit: STAR vs QSO is not separable from compact-source photometry/astrometry alone in part of
  the colour space; an LLM reading the same numbers has no new information (cut (c) logic of ADR 0002).
  The prior expectation is therefore that Layer 2 adds little; a negative result is a valid outcome.

## 2. Options

### 2.1 Models

| # | Option | Real cost |
|---|---|---|
| M1 | One model fixed now (ultra) | Zero selection calls. Choice not backed by any accuracy number; ultra is the slowest with reasoning and the most exposed to quota. |
| M2 | **Val bake-off of the three reachable models, thinking off, then one model** | 3 x 100 val calls (pilot). Chooses on data, on val only. One extra decision point for the human. |
| M3 | Escalation ladder bolt -> lightning -> ultra on `UNKNOWN`/low confidence | Up to 3 calls per object; escalation rule depends on verbalized confidence, which is uncalibrated; three cache keys, three failure surfaces. Pays off only if the models' errors are complementary, which is unmeasured. |

### 2.2 Reasoning

| # | Option | Real cost |
|---|---|---|
| R1 | **Thinking off by default** (`enable_thinking: false`, `max_tokens = 64`) | ~20 output tokens, ~1-3 s. Possible accuracy loss, unmeasured. |
| R2 | Thinking on | ~564 (ultra) to >1,500 (lightning) output tokens, ~10 s, and silent truncation (`finish_reason=length`, empty `content`). 25-75x the output tokens of R1. |

### 2.3 Layer 1 probability in the prompt

| # | Option | Real cost |
|---|---|---|
| P1 | **Blind prompt (features only) as the primary arm** | LLM opinion is independent of Layer 1, so agreement/disagreement is informative. Cannot exploit Layer 1's knowledge. |
| P2 | Prompt includes `p_cal_*` | Anchoring: the LLM may echo Layer 1, adding cost and no information; gains cannot be attributed. |
| P3 | P1 primary + P2 as a paired ablation on the same objects | 2x calls on the ablation subset. Measures anchoring instead of guessing. |

### 2.4 HTTP client

| # | Option | Real cost |
|---|---|---|
| H1 | **stdlib `urllib.request`** | No new dependency; ~60 lines incl. timeout and error mapping; trivially mocked by injecting an opener. No connection pooling (irrelevant at <= 1 req/s). |
| H2 | `requests` | Already imported by `xmatch.py` but **unpinned** (transitive). Using it here deepens an unpinned dependency. |
| H3 | `httpx` / `openai` SDK | New pinned dependency; SDK hides retries and may log request details; breaks the "no new lib without justification" rule for no measured benefit. |

## 3. Decision: model and reasoning (D1)

1. **M2.** The pilot (section 7) runs the same 100 val objects through the three reachable models, thinking
   off, blind prompt. The model with the highest val macro-F1 on the routed objects is chosen, ties broken
   by lower failure rate, then lower median latency. If the paired val CI of the top two overlaps 0, the
   human picks (decision H2). **Human decision.**
2. **No ladder in v1** (YAGNI; one escalation criterion would have to rest on an uncalibrated score).
   Reversal trigger T3 reopens it.
3. **Thinking off by default** (R1): the measured cost of R2 is 25-75x output tokens and a silent-failure
   mode; no accuracy evidence justifies it. Thinking on is an **optional pilot arm** for the chosen model
   only: same 100 val objects, `max_tokens = 4096`, `finish_reason=length` counted as failure. It is adopted
   only if its val macro-F1 gain over thinking off has a paired CI excluding 0 **and** the human accepts the
   measured token multiplier (decision H3). Results of the two settings never share a file (section 17.2).
4. Request body (fixed per prompt version): `model`, `messages`, `temperature = 0`, `top_p = 1`,
   `max_tokens = 64` (thinking off) or `4096` (thinking arm), `stream = false`,
   `chat_template_kwargs = {"enable_thinking": <bool>}`. Whether the endpoint honours a `seed` field is
   unknown; it is **not** sent until verified (open question Q4). JSON-mode / `response_format` is not used
   until verified on these models (Q5); the parser enforces the schema instead.

## 4. Decision: routing threshold and Layer 3 (D2)

1. **Routing**: an object goes to Layer 2 iff `conf_l1 = max(p_cal_star, p_cal_qso, p_cal_galaxy) < tau`,
   columns read by name from the `lgbm_full` predictions.
2. **`tau` is chosen on val only**, before any test call, by this rule: on the grid 0.50..0.99 step 0.01,
   `tau` = the **smallest** value such that LightGBM accuracy on val objects with `conf_l1 >= tau` is
   `>= A_keep`; if the routed val fraction at that `tau` exceeds `R_max`, `tau` is lowered to the largest
   grid value with routed fraction `<= R_max`, and the conflict is recorded. `A_keep = 0.99`,
   `R_max = 0.15` (**human decision H4, confirmed 2026-10-27**; both reversible). The 0.9 seen on test is
   never used.
3. The result is frozen in `data/interim/<release>/layer2/routing_v1.json` (tau, A_keep, R_max, A_keep2,
   grid, val n, routed fraction, val accuracy kept, sha256 of the predictions file, created_utc). The
   runner and the evaluator refuse **either fold** if the manifest is missing or the sha256 differs
   (`routing.select_routed`). The manifest is never rewritten (`scripts/layer2_route.py` refuses an
   existing file); its `tau2` and `verbal_calibration` keys are written as `null` placeholders and are
   **not authoritative** (section 17.6). Changing `tau` after any test call requires a new manifest
   version and an amendment to this ADR stating that test was already seen.
4. **Combination with Layer 3** (decision per object; primary policy = `abstain_on_fail`):
   - `conf_l1 >= tau` -> Layer 1 label.
   - routed, Layer 2 status `OK` with label in {STAR, GALAXY, QSO} and calibrated score `s_l2 >= tau2` ->
     Layer 2 label.
   - routed, anything else (`UNKNOWN`, any failure status, `s_l2 < tau2`) -> **ABSTAIN** ("needs spectrum").
   - Secondary policy reported alongside: `fallback_l1` (same, but ABSTAIN is replaced by the Layer 1
     label; full coverage).
5. `s_l2` is the **verbalized confidence calibrated on val** (D4.5); `tau2` is chosen on val with the
   same smallest-grid-value rule as `tau` against `A_keep2 = 0.90` (**H4, confirmed 2026-10-27**) and
   **no routed-fraction cap** (`R_MAX_TAU2 = 1.0`, human decision 2026-10-27). It is fitted by the
   evaluator from the val Layer 2 results and recorded in the `fit` block of `eval_l2.json`, not in the
   routing manifest (section 17.6). Three outcomes, each recorded in `tau2_status` (section 17.1):
   fitted; infeasible (`tau2 = +inf`, no Layer 2 answer is trusted); not applied (uncalibrated).

## 5. Decision: prompt and output schema (D3)

1. **Inputs = exactly the 21 LightGBM feature columns** of `features` (schema §4 rows 2-22), read by name.
   Same information as Layer 1 (ADR 0002 cut (c)). **Never in the prompt**: `source_id`, `ra`, `dec`, `l`,
   `b`, any `sdss_*` column (class, z, zwarning, ids), `fold`, `classprob_dsc_*`,
   `in_qso_candidates`, `in_galaxy_candidates`, `ap_row_present`, `healpix_l5`, any label. Identifiers and
   positions are excluded so that a memorized catalogue cannot be looked up (risk K3).
2. Formatting: one line per feature, `name = value unit`, values with 4 significant digits; a NULL value is
   printed as `name = MISSING (not measured)`, and when `astrometric_params_solved = 3` a fixed sentence
   states "2-parameter astrometric solution: parallax, proper motion and RUWE were not measured". No value
   is imputed, no sentinel number is printed. NaN in memory is treated as NULL.
3. System prompt states: the task (STAR / GALAXY / QSO from Gaia DR3 photometry and astrometry), that
   `UNKNOWN` is correct when the data do not separate the classes (explicitly: stars and quasars can be
   indistinguishable in these quantities), and the output contract. No class-specific numeric rules are
   written by us (that would be a hand classifier). Zero-shot.
4. **Output schema** (the only accepted form, one JSON object, no extra keys):
   `{"label": "STAR" | "GALAXY" | "QSO" | "UNKNOWN", "confidence": <integer 0..100>}`. No free-text reason
   in v1: it costs tokens, is an injection/log surface, and is not measured by any metric. A code fence
   around the object is stripped; anything else is `INVALID_JSON` or `SCHEMA_VIOLATION`.
5. **Arms**: `blind` (primary, P1) and `with_l1` (P3 ablation: adds the three `p_cal_*` values by name,
   2 decimals). `with_l1` runs only on the val pilot set and, if the human approves (H5), on a test subset.
6. `PROMPT_VERSION` (string, e.g. `"p1"`) and `prompt_template_sha256` (hash of the template text, not of
   the filled prompt) are constants in code; any template change bumps the version. The cache key uses the
   hash of the **filled** prompt (D5.4), so a feature-value change also misses the cache.

## 6. Decision: failure states, retries, quota, cache (D4, D5)

### D4. Failure states (explicit, never silent)

1. Every routed object ends with exactly one `status`. Statuses marked *terminal* are final in the cache
   (D5.4) and are the only ones that can appear in a consolidated parquet; retryable and fatal statuses
   are recorded in the cache and ledger but are **not terminal**: the object is re-attempted on the next
   run, and the fold cannot be consolidated or evaluated until it has a terminal record (code:
   `cache.TERMINAL`, `consolidate.consolidate`; clarified by Amendment 1, section 17.4).

| status | Condition | Terminal | Retry | Run behaviour |
|---|---|---|---|---|
| `OK` | HTTP 200, `finish_reason = "stop"`, valid JSON, schema valid, label in 3 classes | yes | - | continue |
| `UNKNOWN` | as `OK` but label `UNKNOWN` (a valid answer) | yes | - | continue |
| `TRUNCATED` | HTTP 200, `finish_reason = "length"` (any `content`) | yes | no | continue |
| `EMPTY_CONTENT` | HTTP 200, `finish_reason = "stop"`, empty/whitespace `content` | yes | no | continue |
| `INVALID_JSON` | HTTP 200 and the response body or `content` is not parseable as one JSON object | yes | no | continue |
| `SCHEMA_VIOLATION` | JSON parsed but keys/types/enum/range wrong | yes | no | continue |
| `SCHEMA_VIOLATION` | HTTP 200 with `finish_reason` not in {`stop`, `length`} (e.g. `content_filter`, `tool_calls`, absent); raw value kept in `finish_reason` (bounded, section 17.8) | yes | no | continue |
| `RATE_LIMITED` | HTTP 429 after retries exhausted | no | yes | continue; 3 consecutive -> stop run (exit 4) |
| `SERVER_ERROR` | HTTP 5xx after retries exhausted | no | yes | as above |
| `TIMEOUT` | no response within 60 s (per socket operation; total deadline in section 17.8), after retries | no | yes | as above |
| `NETWORK_ERROR` | connection/DNS/TLS error, after retries | no | yes | as above |
| `MODEL_UNAVAILABLE` | HTTP 404 | no | no | **stop run** (fail-closed, exit 2) |
| `AUTH_ERROR` | HTTP 401/403 | no | no | **stop run** (exit 2) |
| `BAD_REQUEST` | other 4xx | no | no | **stop run** (our bug, exit 2) |
| `BAD_REQUEST` | HTTP 2xx other than 200 | no | no | **stop run** (exit 2): undocumented success shape, not parsed |
| `BAD_REQUEST` | HTTP 3xx: **redirects are never followed** (`client.open_no_redirect`; following one would replay the `Authorization` header to another host; security finding S1, fixed) | no | no | **stop run** (exit 2) |
| `QUOTA_STOP` | not called because a cap would be exceeded | no | - | **stop run** (exit 3) |

2. Retry: only for 429, 5xx, timeout, network. At most 3 retries (4 attempts), backoff
   `min(30, 2 * 2**k)` s with uniform jitter of +/-20%, honouring a `Retry-After` header if one ever appears.
   Every attempt counts against the call cap and is written to the ledger. Deterministic outcomes
   (TRUNCATED, INVALID_JSON, SCHEMA_VIOLATION, EMPTY_CONTENT) are not retried: at temperature 0 a retry
   spends quota for the same answer.
3. A response whose `model` field differs from the requested id is kept but flagged (`model_reported`
   column) and counted in the run report.
4. **In metrics**, every routed object stays in the denominator. Non-`OK` statuses are: an abstention under
   `abstain_on_fail`, a Layer 1 fallback under `fallback_l1`, and a wrong answer in the "forced" view of the
   routed-subset comparison (D7.2). The failure rate per status is reported. An object with no terminal
   record makes the run **not evaluable** (the evaluator refuses; no partial-subset metric).
5. **Verbalized confidence is not a probability.** `confidence/100` is a raw score. It is mapped to `s_l2`
   by histogram binning fitted on the val routed set (5 equal-mass bins; bin value = val accuracy of
   `OK` answers in that bin), recorded in the `fit` block of `eval_l2.json` (section 17.6). If fewer than
   100 `OK` val answers exist, `s_l2 = NULL` for all objects and `tau2` is not applied (reported as
   "uncalibrated"); this is **intentional** (section 17.1).

### D5. Quota control (replaces the dollar BudgetTracker)

1. **Pacing**: minimum interval between request starts `60 / rpm_cap` s, `rpm_cap` default **30**
   (75% of the unconfirmed 40 req/min). Single process, sequential requests, no concurrency.
2. **Per-run caps**, both mandatory CLI arguments with no default above the pilot size: `max_calls`
   (HTTP attempts, retries included) and `max_tokens_total` (sum of `prompt_tokens + completion_tokens`).
3. **Hard stop before each attempt**: if `calls_done + 1 > max_calls`, or
   `tokens_done + est_prompt_tokens + max_tokens_request > max_tokens_total`, the attempt is not made, the
   object gets no record, the run ends with `QUOTA_STOP` and exit code 3. `est_prompt_tokens` = max
   `prompt_tokens` seen in this run, or 2,000 before the first response (conservative).
4. **Cache** (append-only JSONL, one line per call result, flushed and `os.fsync`ed per line):
   `data/cache/layer2/<model_slug>/<prompt_version>.jsonl`, shared across releases. Key =
   `sha256(model_id | prompt_version | arm | enable_thinking | max_tokens | filled_prompt_sha256 | source_id)`.
   A key with a terminal status in {OK, UNKNOWN, TRUNCATED, EMPTY_CONTENT, INVALID_JSON, SCHEMA_VIOLATION}
   is never called again. Keys whose last status is RATE_LIMITED/SERVER_ERROR/TIMEOUT/NETWORK_ERROR (or a
   fatal status) are re-attempted on the next run. A truncated last line (crash mid-write) is ignored with
   a warning.
5. **Resume**: re-running the same command reads the cache, skips terminal keys and continues; no state
   other than the cache and the ledger exists. A lock file (`<cache>.lock`, created exclusively, removed on
   exit) prevents two writers (ADR 0001 single-writer).
6. **Ledger** (append-only JSONL, one line per HTTP attempt):
   `data/interim/<release>/layer2/calls_<run_id>.jsonl` with `ts_utc, run_id, model, prompt_version, arm,
   enable_thinking (added by Amendment 1, section 17.7), source_id, attempt, status, http_status,
   latency_ms, prompt_tokens, completion_tokens, finish_reason, reasoning_chars`. Never headers, never
   request body, never raw error body.
7. **Drift canary**: each run starts by re-calling (bypassing the cache, without writing to it) 20 fixed
   val objects already answered (`OK`/`UNKNOWN`) in the cache; the agreement rule, including the
   treatment of transport failures, is in section 17.5. Costs 20 calls per run; skipped when fewer than 20
   such cached val answers exist (always on the very first run); disabled by `--no-canary`.

## 7. Decision: pilot and batch gating (D6)

1. **Pilot G-P (val only, never test)**: 100 objects drawn with `default_rng(20261025)` from the val
   routed set (after `routing_v1.json` exists). Calls: 3 models x 100 (blind, thinking off) = 300; optional
   +100 thinking arm, +100 `with_l1` arm, +50 memorization probe (section 9, K3). Caps for the pilot run:
   `max_calls = 700`, `max_tokens_total = 1,500,000`. Pilot runs use `--sample` and write **no parquet**.
   First evidence (ultra only, full_v0): section 17.11.
2. Pilot report (to the human, not a decision by any agent): produced by
   `scripts/layer2_pilot_report.py` from the cache and the ledger (specified in section 17.7): per model
   and arm, status counts, median and p95 latency, mean and p95 prompt/completion tokens, val
   accuracy/macro-F1 on the 100 (point + Wilson interval; the pixel floor is not reachable on 100 objects
   and is stated so), and a **projection** for the batch: `N_routed(val) + N_routed(test)` (fractions
   applied to the target release) x mean tokens and x `60 / rpm_cap` s, as a range from p50 to p95.
3. **No batch run without an explicit human "yes"** to a stated `max_calls`, `max_tokens_total`, release,
   folds and (on full_v0) pixel subset (section 17.9). The approval text and date go into the run manifest.

## 8. Decision: evaluation (D7)

1. **Population**: the test fold of the same release, same split (ADR 0003), same `labels`, Layer 1 =
   `lgbm_full`. Regime BALANCED (P0) only; Layer 2 outputs no class-probability vector, so prior shift and
   the REALISTIC regime are **not reported for Layer 2** (limitation). On full_v0 the population is the
   chosen test-pixel subset (section 17.9).
2. **Routed-subset comparison** (objects with `conf_l1 < tau`): Layer 1 vs Layer 2 on the same objects.
   Two views: *forced* (any non-`OK` Layer 2 status counts as wrong) and *selective* (only `OK` answers,
   with their coverage).
3. **System comparison** (all test objects), four systems:
   S0 Layer 1 alone (coverage 1); S1 Layer 1 + abstain on routed (no LLM); S2 Layer 1 + Layer 2 with
   `fallback_l1` (coverage 1); S3 Layer 1 + Layer 2 + abstention (`abstain_on_fail`, primary).
   Headline pairs: **S2 vs S0** (same coverage) and **S3 vs S1\***, where S1\* is Layer 1 with a stricter
   threshold `tau*` chosen **on val**: `tau*` is the value, among the unique Layer 1 confidences of the val
   fold, whose val coverage is closest to S3's val coverage, ties going to the smaller value
   (`layer2_data.choose_tau_star`; human decision 2026-10-27). The risk-coverage curve of Layer 1 is
   plotted with S1, S1\*, S3 as points.
4. **Metrics**: accuracy, macro-F1, per-class precision/recall/F1 (by name STAR/GALAXY/QSO), coverage,
   `UNKNOWN` rate, failure rate per status, top-label ECE of answered objects (15 equal-mass bins; system
   confidence = `p_cal` max for Layer 1 answers, `s_l2` for Layer 2 answers), prompt and completion tokens
   per routed object, calls per routed object, latency p50/p95.
5. **Uncertainty**: cluster bootstrap over test level-5 pixels within latitude bands, B = 1000,
   `default_rng(20261025)`, the same replicates for every system, paired-difference CI (ADR 0003,
   `eval.metrics.cluster_bootstrap`). **Reporting floor**: per cell, >= 100 objects in >= 20 pixels; else
   `n/a (n=..., pixels=...)`. On the 100k release the routed subset will mostly fail the floor; its numbers
   carry the PROVISIONAL header.
6. **Ablations**: `with_l1` vs `blind` (anchoring: also report the rate at which the LLM label equals the
   Layer 1 argmax in each arm); thinking on vs off (if run). Paired on the same objects.
7. **Non-determinism record**: temperature 0, requested and reported model ids, run date, prompt version
   and template hash in every output's metadata; canary agreement in the run report.
8. **Negative result rule** (pre-registered; metric and pairs fixed in code as `HEADLINE_METRIC =
   "macro_f1"` and `HEADLINE_PAIRS = (S2-S0, S3-S1*)` and approved by the human on 2026-10-27 **before any
   test number was seen**): a headline pair shows a gain iff its paired macro-F1 cell passes the floor
   **and** the lower bound of its paired CI is > 0. If neither pair shows a gain, the report states "Layer 2
   (text, Nemotron <model>, prompt <v>) adds no measurable value over Layer 1 at <tokens>/object", with the
   CIs. The same statement is emitted unconditionally when `tau2` is infeasible on val (section 17.1). No
   prompt or threshold is changed after seeing test to rescue it; any new attempt is a new prompt version
   evaluated first on val. A negative text result means the VLM follow-up is **not** started (user's own
   condition).

## 9. Risks

| id | Risk | Mitigation / measurement |
|---|---|---|
| K1 | Free-tier terms forbid this use or publication of results | **H1 answered 2026-10-27**: the human confirmed the terms allow this use and a public report; confirmed by the human, not independently verified by the implementer. Residual risk: terms change; re-check before any publication (T6). |
| K2 | Real rate limit unknown (no headers); a daily cap may exist | `rpm_cap = 30`, 429 handling, 3 consecutive 429 -> stop; pilot logs the 429 count. |
| K3 | Memorization: Gaia/SDSS are public and may be in the training data | No ids/positions in the prompt. **Probe**: 50 val objects asked "Gaia DR3 source_id X: SDSS class?" (id only); if accuracy exceeds the val class-prior baseline with a Wilson interval excluding it, leakage is reported as a finding and the human decides (H6). |
| K4 | Reasoning tokens explode or truncate silently | Thinking off; `TRUNCATED`/`EMPTY_CONTENT` are statuses; `reasoning_chars` logged. |
| K5 | Model behind the id changes (non-reproducible) | Reported `model` field stored; drift canary (D5.7, section 17.5); results dated. |
| K6 | Selection bias: routed objects are the hard, SDSS-selected tail | Routed-subset class mix, G-magnitude and `astrometric_params_solved` histograms reported next to every Layer 2 table; no claim outside that subset. |
| K7 | Key leak via logs, exceptions, cache or dashboard | Section 10; tests inject a fake key and assert it appears nowhere. Redirects never followed (S1). |
| K8 | Anchoring if `p_cal` in prompt | Blind primary arm; `with_l1` only as ablation. |
| K9 | Volume on full_v0 (~40x): val+test routed may need tens of thousands of calls (~hours at 30 rpm) | Pilot projection (section 17.7); the full_v0 batch **requires** a seeded random subset of test pixels with explicit caps (section 17.9); the size is H7. |
| K10 | Zero-shot LLM never saw training data, LightGBM did | Stated as a design property, not a flaw: the question is whether a general model adds anything on the hard cases. Few-shot is a non-goal (trigger T4). |
| K11 | An object that keeps failing on transport never gets a terminal record, so the fold can never be consolidated or evaluated | Exit 4 makes it visible (section 17.3); reruns resume; if the same objects stay non-terminal over 3 runs on different days, the human decides (T8). No partial-subset evaluation is introduced. |
| K12 | Quota accounting bypassed by a hostile/buggy `usage` (negative token counts lower `tokens_done`) | Backlog SEC-B1 (section 17.8). Not observed; not measured. |
| K13 | High `UNKNOWN` rate leaves too few `OK` val answers to calibrate (first pilot: 90/100 `UNKNOWN`, section 17.11) | `tau2` "not applied" branch (17.1.3) and T7; the negative-result rule covers the outcome. |

## 10. Security (D8)

1. The key is read only from `os.environ["NVIDIA_API_KEY"]`, else from the line `NVIDIA_API_KEY=...` of
   `<repo>/.env` (stdlib parser, that key only; `.env` is git-ignored). Missing key -> `MissingKeyError`
   before any call, run exit 1 (code; the earlier text said `AUTH_ERROR`). The key is never a function
   default, CLI argument, cache field or metadata value.
2. Every string that can reach a log, exception, ledger or the dashboard passes through `redact(text, key)`:
   replaces the literal key with `***`, applies the `tap._sanitize` patterns (Bearer, `key=`), truncates to
   200 chars. HTTP error bodies are **not** logged; only the status code and the status enum. `from exc`
   chains are not used in the client (raw server text would survive).
3. Only the 21 public Gaia DR3 feature values (and, in `with_l1`, three model probabilities) leave the
   machine. A test asserts that the filled prompt contains no forbidden column name and not the decimal
   `source_id`.
4. Model output is data: validated by the schema parser, stored as enum + int; it is never put in a path,
   SQL, ADQL or rendered as HTML/markdown (Streamlit `st.text`/dataframe only). Server-supplied free
   strings (`model_reported`, `finish_reason`) are bounded and filtered (section 17.8, SEC-B3).
5. Replacement for CONTRACT §8 (**accepted by the human on 2026-10-27, H8**; the orchestrator edits the
   CONTRACT):
   *"LLM calls go only to the NVIDIA API (ADR 0005), with `NVIDIA_API_KEY` from the environment or `.env`.
   Any run beyond the 100-object val pilot needs explicit human approval of `max_calls` and
   `max_tokens_total`. The quota guard is checked before every call. No AWS, Bedrock or paid provider
   without a new ADR."*
6. Never follow an HTTP redirect (section 6, D4 table; finding S1, fixed in `client.open_no_redirect`).

## 11. Decisions for the human

- H1. Free-tier terms of the NVIDIA API: is this research use (and a later public report) allowed?
  **Resolved 2026-10-27: yes, confirmed by the human, not independently verified by the implementer.**
- H2. Model choice after the pilot when the top two are statistically tied (D1.1). Open.
- H3. Whether to run the thinking-on arm, and to adopt it if it wins (token multiplier). Open.
- H4. `A_keep`, `R_max`, `A_keep2`. **Resolved 2026-10-27: 0.99, 0.15, 0.90.** Also resolved the same
  day: `tau2` has no routed-fraction cap (`R_MAX_TAU2 = 1.0`); `tau*` rule of D7.3.
- H5. Whether the `with_l1` ablation runs on test too (doubles test calls on its subset). Open.
- H6. Action if the memorization probe is positive. Open.
- H7. Batch scope on full_v0: number of test (and val) pixels in the seeded subset, `max_calls`,
  `max_tokens_total` (section 17.9). **Open; decided after the pilot report.**
- H8. Accept the CONTRACT §8 replacement text and closing of backlog SEC-01 dollar items. **Resolved
  2026-10-27: accepted.**
- Also resolved 2026-10-27: the negative-result metric and pairs of D7.8, approved before any test number.

## 12. Acceptance criteria (to leave Proposed)

1. H1, H4 and H8 answered (**done 2026-10-27**); `routing_v1.json` exists for the target release, built
   from val only (**full_v0: done 2026-10-27**, section 17.11).
2. Unit tests (no network) pass: prompt leak guard, NULL formatting, parser for every status, client error
   mapping for 200/2xx-not-200/3xx/400/401/404/429/5xx/timeout/`finish_reason=length`/other
   `finish_reason`, quota hard stop, cache resume after a truncated line, key redaction in every output
   channel, and the Amendment 1 acceptance items (section 17).
3. Pilot G-P completed on val with 0 `AUTH_ERROR`/`BAD_REQUEST`, failure rate (all non-`OK`, non-`UNKNOWN`
   statuses) <= 5% for the chosen model, and the pilot report (section 17.7) delivered to the human.
4. Reviewer and security verdicts with no BLOCKER/MAJOR open (R1-001 is closed by section 17.1 once
   implemented).

## 13. Reversal triggers

- T1. Pilot failure rate > 5% for every model, or `MODEL_UNAVAILABLE` for the chosen model -> stop; revisit
  backend (new ADR).
- T2. Measured 429s at `rpm_cap = 30` -> lower `rpm_cap` (reversible parameter); if a batch becomes > 24 h
  wall time, H7 subsetting becomes mandatory (on full_v0 it is mandatory anyway, section 17.9).
- T3. On val, the errors of two models are complementary (one is right on >= 30% of the other's wrong
  answers, CI excluding 0) -> reopen the ladder (M3).
- T4. Blind zero-shot val macro-F1 on routed objects is below Layer 1's on the same objects with a CI
  excluding 0 -> the expected outcome of a negative result; one new prompt version (few-shot from **train**)
  may be tried on val before closing the question.
- T5. Drift canary fails -> results of that run are not comparable with earlier runs; new run id and date.
- T6. The free-tier terms forbid the use (now or after a change) -> stop all calls; Layer 2 is reported as
  not run; no public output.
- T7. The val fold of the target release has < 100 `OK` answers (uncalibrated) -> the "not applied" branch
  of section 17.1 is used for that evaluation; if it happens on full_v0, the human decides whether to
  enlarge the val pixel subset (val only) before any test evaluation.
- T8. `CANARY_INCONCLUSIVE` on 3 consecutive runs, or the same routed objects non-terminal after 3 runs
  on different days -> stop batch runs; the human decides (backend health, not a model change).

## 14. Non-goals

Images/VLM; AWS/Bedrock/Gemini; streaming; concurrency; REALISTIC prior regime for Layer 2; few-shot or
fine-tuning in v1; free-text reasons; using the API from the dashboard (the dashboard only reads files);
changing ADR 0003 metrics, split or floor; evaluating a partial subset of routed objects; any mechanism that
lets a test number influence `tau`, `tau2`, `tau*`, `A_keep2` or the verbal calibration.

## 15. Consequences

- Good: Layer 2 is measured against Layer 1 on the same pixels with paired CIs; every call is accounted for;
  no silent failure; reruns cost nothing (cache); the negative result is pre-defined, including the case
  where Layer 2 is never trustworthy on val.
- Bad: zero-shot on 21 numbers may be weak by construction; REALISTIC regime unavailable for Layer 2; the
  pilot alone costs ~300-700 calls; full_v0 needs hours of wall time or pixel subsetting (now mandatory);
  results depend on a hosted model that can change; verbalized-confidence calibration needs >= 100 `OK`
  val answers; a persistently failing object blocks evaluation (K11).
- Maintenance cost: one stdlib HTTP client (~230 lines as built), prompt module with a version constant,
  quota guard, JSONL cache/ledger, one routing manifest per release, one evaluator module, one pilot-report
  module, one dashboard page; re-run the canary per run; re-check free-tier terms if they change.
- Reversibility: everything is reversible in < 1 day **except** test-fold exposure: once a prompt version
  and `tau` have been evaluated on test, they cannot be tuned and re-evaluated on the same test without
  declaring it.

## 16. Implementation plan (per file)

```
FILE: src/astro_triage/llm/__init__.py                      [create]
INTENT: package marker, no imports with side effects
EXPORTS: none
DEPENDS: none
INVARIANTS: importing it makes no network call and reads no file
ACCEPTANCE: `import astro_triage.llm` succeeds with NVIDIA_API_KEY unset
RISK: none
```

```
FILE: src/astro_triage/llm/secrets.py                       [create]
INTENT: load the API key and redact it from any text
EXPORTS:
  class MissingKeyError(RuntimeError)
  def load_api_key(env: Mapping[str, str] | None = None, dotenv_path: Path | None = None) -> str
  def redact(text: str, key: str | None, limit: int = 200) -> str
DEPENDS: os, pathlib, re, collections.abc, astro_triage.ingest.tap._sanitize (closed list)
INVARIANTS: reads only the NVIDIA_API_KEY line of .env; never prints/logs; redact output never contains key
ACCEPTANCE: env wins over .env; missing in both -> MissingKeyError whose message has no path content beyond
  the file name; redact("x <key> y", key) contains "***" and not key; quoted value in .env is unquoted
RISK: .env with BOM or CRLF; key containing regex metacharacters (use str.replace, not regex)
```

```
FILE: src/astro_triage/llm/prompt.py                        [create]
INTENT: build the messages from one feature row and parse the model answer
EXPORTS:
  PROMPT_VERSION: Final[str]; TEMPLATE_SHA256: Final[str]
  FEATURE_COLUMNS: Final[tuple[str, ...]]   # the 21 features, schema §4 rows 2-22
  FORBIDDEN_SUBSTRINGS: Final[tuple[str, ...]]
  LABELS: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO", "UNKNOWN")
  class Answer(NamedTuple): label: str; confidence: int
  class ParseError(ValueError): status: str   # "EMPTY_CONTENT" | "INVALID_JSON" | "SCHEMA_VIOLATION"
  def build_messages(row: Mapping[str, float | None], arm: Literal["blind", "with_l1"],
                     l1: Mapping[str, float] | None = None) -> list[dict[str, str]]
  def parse_answer(content: str | None) -> Answer
DEPENDS: json, math, hashlib, typing, collections.abc (closed list)
INVARIANTS: row read by column name; NULL/NaN -> "MISSING (not measured)"; no forbidden substring and no
  source_id in output; with_l1 reads p_cal_star/p_cal_qso/p_cal_galaxy by name; blind ignores l1
ACCEPTANCE: leak test passes; Solved=3 row prints the 2-parameter sentence and MISSING for 9 fields;
  parse_answer accepts a fenced object, rejects extra keys, confidence 101, label "star", "" , "{"
RISK: LLM adds prose around JSON (counts as INVALID_JSON by design); float formatting of -0.0/1e-5
```

```
FILE: src/astro_triage/llm/client.py                        [create]
INTENT: one chat completion over urllib, mapped to an explicit status, with retry/backoff
EXPORTS:
  BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
  class CallResult(NamedTuple): status: str; http_status: int | None; content: str | None;
      finish_reason: str | None; model_reported: str | None; prompt_tokens: int | None;
      completion_tokens: int | None; latency_ms: int; reasoning_chars: int; attempts: int
  class FatalCallError(RuntimeError): status: str   # MODEL_UNAVAILABLE | AUTH_ERROR | BAD_REQUEST
  def open_no_redirect(request: urllib.request.Request, timeout: float) -> Any   # added (S1)
  def chat(messages: list[dict[str, str]], model: str, *, key: str, enable_thinking: bool,
           max_tokens: int, timeout_s: float = 60.0, max_retries: int = 3,
           before_attempt: Callable[[], None], after_attempt: Callable[[CallResult], None],
           opener: Callable[..., Any] = open_no_redirect,
           sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None) -> CallResult
DEPENDS: json, time, random, http.client, urllib.request, urllib.error, socket, typing, collections.abc
  (closed list)
INVARIANTS: body has temperature 0, top_p 1, stream false, chat_template_kwargs.enable_thinking;
  before_attempt is called before every attempt (quota hard stop raises there); no header, body or
  key in any exception or return value; error bodies discarded; retries only 429/5xx/timeout/network;
  3xx never followed; 2xx != 200 -> BAD_REQUEST
ACCEPTANCE: fake opener tests for every row of the D4 table; finish_reason=length -> TRUNCATED even with
  HTTP 200; empty content -> status left to parser; 4 attempts max; backoff values <= 30 s
RISK: urllib raises HTTPError for 4xx/5xx and URLError wrapping socket.timeout; both must be mapped
```

```
FILE: src/astro_triage/cost/__init__.py                     [create]
FILE: src/astro_triage/cost/quota.py                        [create]
INTENT: pacing and per-run hard caps checked before each HTTP attempt; JSONL ledger
EXPORTS:
  class QuotaExceeded(RuntimeError)
  class QuotaGuard:
      def __init__(self, *, rpm_cap: float, max_calls: int, max_tokens_total: int,
                   clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep)
      def before_attempt(self, max_tokens_request: int) -> None   # paces, then raises QuotaExceeded
      def record(self, prompt_tokens: int | None, completion_tokens: int | None) -> None
      calls_done: int; tokens_done: int
  class Ledger:
      def __init__(self, path: Path); def append(self, entry: Mapping[str, object]) -> None
DEPENDS: json, os, time, pathlib, collections.abc (closed list)
INVARIANTS: no attempt after a cap would be crossed; est prompt tokens = max seen, 2000 before first;
  ledger lines flushed+fsynced; ledger rejects keys outside an allowlist (no headers/body fields)
ACCEPTANCE: fake clock shows >= 60/rpm_cap s between starts; max_calls=2 -> third before_attempt raises;
  token cap raises before the call that would cross it
RISK: None usage fields on error responses (count 0 tokens but 1 call)
```

```
FILE: src/astro_triage/llm/cache.py                         [create]
INTENT: append-only JSONL result cache with resume and single-writer lock
EXPORTS:
  TERMINAL: Final[frozenset[str]]
  def cache_key(model: str, prompt_version: str, arm: str, enable_thinking: bool, max_tokens: int,
                filled_prompt_sha256: str, source_id: int) -> str
  class ResultCache:
      def __init__(self, path: Path)            # acquires <path>.lock exclusively
      def get(self, key: str) -> dict | None    # last record for key
      def is_terminal(self, key: str) -> bool
      def put(self, key: str, record: Mapping[str, object]) -> None
      def close(self) -> None; __enter__/__exit__
DEPENDS: json, os, hashlib, logging, pathlib, collections.abc (closed list)
INVARIANTS: never rewrites existing lines; truncated last line ignored with a warning; second instance on
  the same path raises; source_id stored as int
ACCEPTANCE: write 3, truncate file mid-line, reopen -> 2 records, warning logged; retryable status not
  terminal; lock released on exception
RISK: stale lock after a hard kill (message tells the user to delete it; no auto-steal)
```

```
FILE: src/astro_triage/abstention/__init__.py               [create]
FILE: src/astro_triage/abstention/policy.py                 [create]
INTENT: choose tau/tau2 on val, route, calibrate verbal confidence, combine layers
EXPORTS:
  def l1_confidence(pred: pd.DataFrame) -> pd.Series          # max of p_cal_* by name
  def choose_threshold(conf: np.ndarray, correct: np.ndarray, a_keep: float, r_max: float,
                       grid: np.ndarray) -> dict[str, float | bool]
  def fit_verbal_calibration(raw: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> dict | None
  def apply_verbal_calibration(raw: np.ndarray, calib: dict | None) -> np.ndarray
  def combine(l1_label: np.ndarray, l1_conf: np.ndarray, l2_status: np.ndarray, l2_label: np.ndarray,
              s_l2: np.ndarray, tau: float, tau2: float | None,
              policy: Literal["abstain_on_fail", "fallback_l1"]) -> np.ndarray   # labels incl. "ABSTAIN"
  def write_routing_manifest(path: Path, payload: Mapping[str, object]) -> None   # temp+rename
DEPENDS: json, os, hashlib, pathlib, numpy, pandas, typing (closed list)
INVARIANTS: never receives test rows for fitting (caller passes val); class columns by name;
  fallback_l1 never outputs ABSTAIN; tau2=None means "not applied", tau2=+inf means "trust none"
ACCEPTANCE: toy cases for the r_max conflict, < 100 OK answers -> None, every combine branch,
  tau2=+inf -> no routed object gets a Layer 2 label
RISK: ties on the grid; bins with identical raw scores (use np.quantile with unique edges)
```

```
FILE: src/astro_triage/llm/runner.py                        [create]
INTENT: run Layer 2 for one release/fold/model/arm; consolidate cache into a predictions parquet
EXPORTS:
  def select_routed(release_dir: Path, fold: Literal["val", "test"], manifest: Path,
                    sample: int | None, seed: int = 20261025) -> pd.DataFrame
  def run(release_dir: Path, fold: str, model: str, arm: str, enable_thinking: bool,
          guard: QuotaGuard, cache: ResultCache, ledger: Ledger, key: str,
          sample: int | None = None, canary: bool = True) -> dict   # run report
  def consolidate(release_dir: Path, fold: str, model: str, arm: str, cache: ResultCache) -> Path
DEPENDS: astro_triage.llm.{prompt,client,cache,secrets}, astro_triage.cost.quota,
  astro_triage.abstention.policy, pandas, pyarrow, pyarrow.parquet, hashlib, json, logging, os, pathlib
INVARIANTS: both folds refused unless routing manifest sha matches predictions file; QuotaExceeded and
  FatalCallError stop the run after recording; consolidate refuses if any routed object lacks a
  terminal record; parquet written via temp+rename, zstd, sorted by source_id
ACCEPTANCE: end-to-end with fake opener on a 20-row synthetic release: resume after QUOTA_STOP produces
  the same parquet as one uninterrupted run; zero calls on a full-cache rerun **with `--no-canary`**
  (with the canary on and >= 20 cached val answers: exactly CANARY_N calls; section 17.5)
RISK: module size; keep it under 300 lines (as built, split into runner.py, routing.py, consolidate.py)
```

Output table, name as amended in section 17.2:
`data/interim/<release>/layer2/predictions_l2_<model_slug>_<prompt_version>_<arm>_think-<on|off>_<fold>.parquet`
(metadata: `schema_version`, `release`, `fold`, `model_requested`, `prompt_version`, `template_sha256`,
`arm`, `enable_thinking`, `max_tokens`, `temperature`, `endpoint`, `routing_manifest_sha256`,
`created_utc`, `git_sha`):

| Column | Arrow type | Nullable | Rule |
|---|---|---|---|
| `source_id` | int64 | no | strictly increasing |
| `l1_conf` | float32 | no | max of `p_cal_*` by name |
| `status` | dictionary<int8,string> | no | D4 enum, terminal statuses only |
| `label_l2` | dictionary<int8,string> | yes | non-null iff status in {OK, UNKNOWN} |
| `confidence_raw` | int8 | yes | 0..100; non-null iff `label_l2` non-null |
| `model_reported` | string | yes | `model` field of the response (bounded, section 17.8) |
| `prompt_tokens` | int32 | yes | from `usage`, summed over attempts |
| `completion_tokens` | int32 | yes | as above |
| `latency_ms` | int32 | no | sum over attempts |
| `attempts` | int8 | no | 1..4 |
| `finish_reason` | string | yes | last attempt, raw value (bounded, section 17.8) |
| `called_utc` | timestamp[ms, UTC] | no | last attempt |

`s_l2` and system labels are computed at evaluation time, not stored (as `predictions` §6).

```
FILE: scripts/layer2_route.py                               [create]
INTENT: CLI: build routing_v1.json from val (tau; tau2/verbal_calibration written as null placeholders)
FILE: scripts/layer2_run.py                                 [create]
INTENT: CLI: --release --fold --model --arm --thinking --sample --max-calls --max-tokens-total --rpm-cap
  --no-canary (max-calls/max-tokens-total required); exit codes per section 17.3
FILE: scripts/layer2_evaluate.py                            [create]
INTENT: CLI: fit verbal calibration/tau2/tau* on val results, evaluate S0-S3 on test, write
  data/interim/<release>/layer2/eval_l2.json and eval_l2.md (amended in section 17.6)
FILE: src/astro_triage/evaluation/layer2.py                 [create]
INTENT: S0-S3, routed-subset views, paired pixel bootstrap via eval.metrics, floor, top-label ECE
DEPENDS: eval.metrics, astro_triage.abstention.policy, numpy, pandas, json
INVARIANTS: refuses if any routed test object lacks a terminal record; per-class by name; same bootstrap
  replicates for all systems
FILE: src/astro_triage/dashboard/data.py                    [modify]
INTENT: add load_layer2(rel) -> pd.DataFrame | None and layer2_summary(...) reading eval_l2.json and the
  layer2 parquet; returns None when the directory is absent (no change to existing functions)
FILE: src/astro_triage/dashboard/app.py                     [modify]
INTENT: one "Camada 2" section shown only when load_layer2 is not None; text/dataframes only
```

## 17. Amendment 1 (2026-10-27)

Task T-L2-02. Scope: closes reviewer finding R1-001 (MAJOR), records the human decisions of 2026-10-27,
and fixes nine rules that the first implementation showed to be missing or ambiguous. Status stays
**Proposed**. Every item has: normative text, reason, verifiable acceptance. Section 17.10 is the file plan;
section 17.11 records the first pilot evidence. Nothing in this amendment uses, or was informed by, any
test-fold Layer 2 number: when it was written no `eval_l2.json` existed in the repository (the full_v0
`layer2/` directory held only `routing_v1.json` and two val ledgers).

### 17.0 Human decisions (2026-10-27)

| id | Decision | Recorded as |
|---|---|---|
| H1 | The NVIDIA free-tier terms allow this research use and a later public report | **Confirmed by the human, not independently verified by the implementer.** No agent read the terms. |
| H4 | `A_keep = 0.99`, `R_max = 0.15`, `A_keep2 = 0.90` | Defaults of `layer2_route.py` and of the manifest key `a_keep2` |
| H4b | `tau2` has no routed-fraction cap | `layer2_data.R_MAX_TAU2 = 1.0` |
| H4c | `tau*` = value among the unique val Layer 1 confidences whose val coverage is closest to S3's val coverage; ties to the smaller value | `layer2_data.choose_tau_star` |
| H8 | CONTRACT §8 replacement text of section 10.5 accepted | Orchestrator edits `.claude/CONTRACT.md`; this ADR does not |
| D7.8 | Headline metric `macro_f1` on pairs S2-S0 and S3-S1\*; gain iff the cell passes the floor **and** the paired-CI lower bound > 0 | `layer2.HEADLINE_METRIC`, `layer2.HEADLINE_PAIRS`, `layer2.headline_verdict`; approved **before** any test number |

Still open: H2, H3, H5, H6, H7.

### 17.1 `tau2` infeasible and uncalibrated (D2.5, D4.5; closes R1-001)

Normative:
1. `tau2` is fitted only when the verbal calibration exists (>= 100 `OK` val answers). It is the smallest
   grid value (0.50..0.99, step 0.01) whose kept val accuracy among `OK` answers with `s_l2 >= tau2` is
   `>= A_keep2`, with `R_MAX_TAU2 = 1.0`.
2. **Infeasible**: if no grid value reaches `A_keep2`, the evaluation is **not refused**. `tau2 = +inf`
   (no Layer 2 answer is trusted), so S3 = S1 and S2 = S0 by construction; `tau*` is still fitted from
   S3's val coverage. `fit.tau2_status = "infeasible: no grid value reaches A_keep2"`. The verdict is
   forced: `verdict.outcome = "no_gain"` with `verdict.forced_reason = "tau2 infeasible on val: Layer 2
   answers are never trusted, S3 = S1 and S2 = S0 by construction"`, regardless of the bootstrap cells
   (S3 vs S1\* can differ only through `tau*` vs `tau`, which is not a Layer 2 effect). The markdown
   emits the D7.8 negative-result sentence plus the forced reason. The bootstrap and all tables are still
   computed and written.
3. **Uncalibrated** (< 100 `OK` val answers): **intentional and kept**. `s_l2 = NaN` everywhere, `tau2` is
   not applied (`tau2 = None`): every `OK` Layer 2 answer in {STAR, GALAXY, QSO} is trusted.
   `fit.tau2_status = "not_applied: uncalibrated (fewer than 100 OK val answers)"`. The markdown states in
   the Result section that S2/S3 trust every `OK` answer and that their ECE is n/a (no score). Reason: with
   < 100 answers the binning is noise; trusting none would make the evaluation degenerate into 17.1.2
   without evidence, while the forced/selective routed views already expose raw Layer 2 quality. T7
   governs the full_v0 case.
4. Serialization: `eval_l2.json` cannot hold `inf` (`allow_nan=False`; `to_jsonable` turns non-finite
   numbers into `null`). Therefore `fit.tau2` is a finite number only when `tau2_status == "fitted"` and is
   `null` otherwise; **readers must branch on `tau2_status`, never on `tau2` being null** (null means both
   "trust none" and "trust all").
5. Unchanged and still fail-closed (`EvalError`, exit 1): routing manifest missing/unreadable or its
   predictions sha differs; a routed object without a terminal record (either fold); parquet metadata
   mismatch (`routing_manifest_sha256`, `model_requested`, `arm`, `fold`, `prompt_version`,
   `enable_thinking`); val/test mismatch in `template_sha256` or `enable_thinking`; pixel map disagreement.

Reason: R1-001: the code raised `EvalError` when `choose_threshold` found no feasible `tau2`, so the most
likely negative outcome (Layer 2 never reaches 0.90 on val) produced no report at all, contradicting D7.8.

Acceptance:
- Synthetic release whose val `OK` answers (>= 100) are all wrong: `layer2_evaluate.py` exits 0;
  `fit.tau2_status` starts with `"infeasible"`; `fit.tau2` is null; `verdict.outcome == "no_gain"`;
  `verdict.forced_reason` present; S2 labels equal S0 labels and S3 labels equal S1 labels on every test
  object; `eval_l2.md` contains "adds no measurable value".
- Synthetic release with 99 `OK` val answers: exit 0; `tau2_status` starts with `"not_applied"`; S3 gives a
  Layer 2 label to every routed `OK` test answer.
- Each fail-closed case of item 5 still exits 1 with "layer2 evaluation refused".

### 17.2 Parquet name carries the reasoning setting (section 16)

Normative: the consolidated parquet is
`predictions_l2_<model_slug>_<prompt_version>_<arm>_think-<on|off>_<fold>.parquet`.
`consolidate.output_path` takes `enable_thinking: bool` (keyword, required). `layer2_evaluate.py` gains
`--thinking` (store_true, same meaning as in `layer2_run.py`) and passes it to `load_fold`, which reads the
file of that setting and also checks `meta["enable_thinking"] == str(thinking).lower()`. A file with the
pre-amendment name (no `think-` segment) is never read; it is regenerated from the cache by
re-consolidating (zero API calls).

Reason: thinking on and off share model, prompt version, arm and fold, so the old name made one run
overwrite the other; the cache key already separated them.

Acceptance: consolidating the same fold with thinking off and on yields two files, both present; the
evaluator with `--thinking` reads only the `think-on` file and refuses (exit 1) if only `think-off`
exists; a legacy-named file in the directory is ignored.

### 17.3 Exit codes (`scripts/layer2_*.py`)

Normative, `scripts/layer2_run.py`:

| code | Meaning | Causes | Operator action |
|---|---|---|---|
| 0 | Complete | `stop_reason` null and `pending == 0`: full run consolidated, or `--sample` run with every sampled object terminal (no parquet by design) | none |
| 1 | Local error, nothing about the API | `MissingKeyError`, `CacheLockedError`, `ManifestMismatchError`/`ValueError` (manifest, predictions coverage, consolidation rule), `OSError`, `KeyError`, `pyarrow.ArrowException` | fix the local cause |
| 2 | Fatal stop, needs a human or a code change | `MODEL_UNAVAILABLE`, `AUTH_ERROR`, `BAD_REQUEST` (incl. 2xx != 200 and 3xx), `DRIFT` | do not rerun blindly; T1/T5 |
| 3 | `QUOTA_STOP` | `max_calls` or `max_tokens_total` would be exceeded | rerun with new approved caps; resumes from cache |
| 4 | **Incomplete, transient** (new) | `CONSECUTIVE_RETRYABLE`; `CANARY_INCONCLUSIVE` (17.5); loop finished with `pending > 0` (retryable records left) | rerun the same command later; resumes from cache; T8 |

`CONSECUTIVE_RETRYABLE` moves from 2 to 4 (it is transient, not a fatal API answer); `DRIFT` stays 2.
`pending > 0` moves from 0 to 4. The JSON run report is printed for every code except 1. `layer2_route.py`,
`layer2_evaluate.py` and `layer2_pilot_report.py`: 0 written, 1 refused or failed.

Reason: exit 0 with `pending > 0` writes no parquet, so any chained step (`layer2_evaluate.py`) would fail
later or read a stale file; a transient outage is not the same class as a revoked key.

Options weighed: keep 0 and rely on the printed "not consolidated" line (cost: silent in automation,
rejected); reuse 3 (cost: conflates quota with outage; caps would be raised for nothing, rejected).
Reversible in minutes.

Acceptance: fake-opener tests: 3 consecutive 503 -> exit 4; one object ending `TIMEOUT` among terminal
others -> exit 4 and no parquet; 404 -> 2; canary 17/20 conclusive agreements with 0 transport failures ->
2; cap reached -> 3; full success -> 0; missing key -> 1 and no ledger line written.

### 17.4 D4 table additions (applied in section 6)

Normative (already in the code; now in the table): HTTP 2xx other than 200 -> `BAD_REQUEST`, fatal; HTTP
3xx -> `BAD_REQUEST`, fatal, redirects never followed (`open_no_redirect`; finding S1, fixed);
`finish_reason` outside {`stop`, `length`} -> `SCHEMA_VIOLATION`, terminal, raw value kept in the
`finish_reason` column subject to 17.8 SEC-B3; malformed response body at HTTP 200 -> `INVALID_JSON`.
Clarification: retryable and fatal statuses are not terminal in the cache and never appear in a
consolidated parquet.

Reason: these behaviours existed in code without a normative row, so a later change could drop them.

Acceptance: `tests/test_llm_client.py` / `tests/test_llm_redirect.py` cover 201, 204, 301, 302, 307 and
`finish_reason` in {`content_filter`, `tool_calls`, null}; no request is sent to the redirect target.

### 17.5 Canary and the "zero calls on rerun" criterion (D5.7, section 16)

Normative:
1. "Zero calls on a full-cache rerun" holds **with `--no-canary`**. With the canary on and >= 20 cached
   val answers, a full-cache rerun makes exactly `CANARY_N = 20` attempts plus their retries.
2. Each canary result is one of: *agree* (status `OK`/`UNKNOWN` and same label as cached); *disagree*
   (any other terminal status, including `TRUNCATED`, `EMPTY_CONTENT`, `INVALID_JSON`,
   `SCHEMA_VIOLATION`, or a different label: these are model behaviour); *inconclusive* (final status
   `RATE_LIMITED`, `SERVER_ERROR`, `TIMEOUT`, `NETWORK_ERROR` after retries: transport, not model).
3. Decision, in order: `DRIFT` (exit 2) iff `disagree > CANARY_N - CANARY_MIN_AGREE` (i.e. >= 3);
   else `CANARY_INCONCLUSIVE` (exit 4, no object is called) iff `agree + disagree < CANARY_MIN_AGREE`
   (i.e. > 2 inconclusive); else pass. No replacement objects are drawn (the 20 stay fixed). Fatal statuses
   and `QuotaExceeded` during the canary keep their own exit codes (2, 3).
4. Run report: `canary = {"status": "pass" | "drift" | "inconclusive" | "skipped" | "disabled",
   "agree", "disagree", "inconclusive", "n"}`.

Reason: the code counted a 429/5xx/timeout as a disagreement, so a backend outage produced a false `DRIFT`
(exit 2) and the result "not comparable" (T5) for a non-model reason.

Acceptance: fake opener: 20 agreements -> pass; 3 label changes -> drift, exit 2; 3 final timeouts and 17
agreements -> inconclusive, exit 4, zero non-canary calls; 2 timeouts, 18 agreements -> pass; full-cache
rerun with `--no-canary` -> `calls_made == 0`.

### 17.6 Who freezes `tau2` and the verbal calibration (D2.3, D2.5, D4.5)

Normative:
1. `routing_v1.json` is immutable (its sha256 is in every parquet's metadata). Its `tau2: null` and
   `verbal_calibration: null` keys are placeholders, **not authoritative**, never filled, never read for a
   decision. Its `a_keep2` (0.90, H4) is the only source of `A_keep2`.
2. The **official record** of `tau2`, `tau2_status`, the verbal calibration (`edges`, `values`), `tau*`,
   `R_MAX_TAU2`, `A_keep2` and the val counts is the `fit` block of `eval_l2.json`. It is a deterministic
   function of (val parquet, routing manifest) only. The block also stores `val_parquet_sha256` and
   `routing_manifest_sha256`, so it can be recomputed and audited.
3. **None of these values uses the test fold**: `fit_on_val` receives only the val `FoldData`; test is
   loaded for evaluation after the fit. The `--a-keep2` override of `layer2_evaluate.py` is **removed**
   (it allowed changing `A_keep2` after test numbers were printed). A different `A_keep2` requires a new
   routing manifest version and an amendment stating whether test was seen.
3b. **Frozen reference (2026-10-29, recorded by the orchestrator before any test-fold call):**
   `data/interim/full_v0/layer2/routing_v1.json` has sha256
   `5cbb24c296973fcafde622fb8f42a598c890454c9fb943b3691d07ccd877af64` (tau = 0.86, val routed fraction
   0.0896, kept accuracy 0.9904, `a_keep2` = 0.90). Any later edit to the manifest changes this value and is
   therefore detectable; the evaluator's `fit.routing_manifest_sha256` must equal it.
4. A `routing_v2.json` written after the val Layer 2 run was rejected: it adds a second frozen artifact
   whose sha is bound to no parquet, must be produced in an extra step between val and test, and records
   nothing that the deterministic `fit` block does not.

Acceptance: running the evaluator twice yields byte-identical `fit` blocks; altering test labels or the
test parquet (synthetic) leaves the `fit` block unchanged; `layer2_evaluate.py --a-keep2 0.8` is rejected by
argparse (exit 2 from argparse is acceptable here); `fit.val_parquet_sha256` equals the sha256 of the val
parquet on disk.

### 17.7 Pilot report (D6.2)

Normative: new module `src/astro_triage/evaluation/layer2_pilot.py` and CLI
`scripts/layer2_pilot_report.py --release --models M [M ...] --arms {blind,with_l1}... [--thinking-models M
...] --sample 100 --target-release R [--rpm-cap 30]`. It makes **no API call** and writes
`data/interim/<release>/layer2/pilot_report.json` and `pilot_report.md` (temp+rename).
1. Population: `select_routed(release, "val", manifest, sample=100, seed=20261025)`, i.e. exactly the pilot
   objects; per (model, arm, thinking) the record of each object is read from the cache by recomputing
   its key with `request_for`. The cache is read **without taking the writer lock** (new read-only loader,
   same truncated-line tolerance). Objects with no record are reported as `pending`, never dropped.
2. Per (model, arm, thinking): status counts and rates; latency p50/p95 (ms, summed over attempts); prompt,
   completion and total tokens mean/p95; attempts mean/p95; val accuracy **forced** (n = 100, non-`OK`
   wrong) and **selective** (`OK` only) with 95% Wilson intervals; macro-F1 point (forced view) and Layer
   1 macro-F1 on the same objects; `UNKNOWN` rate; Layer-2-equals-Layer-1-argmax rate; `model_reported`
   counts. From the ledger (`calls_*.jsonl` of the release, filtered by model, arm and `enable_thinking`):
   total HTTP attempts, HTTP 429 count, retries, canary attempts. Ledger lines written before
   `enable_thinking` was added are reported under `thinking: "unknown"`.
3. Model comparison for H2: paired val pixel bootstrap (same `eval.metrics.bootstrap_pixel_draw`, B =
   1000, seed 20261025) of the forced macro-F1 difference for each model pair, labelled descriptive.
4. Projection for `--target-release`: `N = round(f_val * (n_val + n_test))`, where `f_val` is
   `val_routed_fraction` from the target release's `routing_v1.json` (if absent: from the pilot release's
   manifest, labelled "estimate from another release"); `n_val`, `n_test` are row counts of the target
   release's `labels.parquet` per fold (counts only; no test label or Layer 1 test outcome is read). Calls
   = `N x attempts` (p50..p95) + `CANARY_N` per planned run; tokens = `N x total tokens` (p50..p95);
   hours = `N x max(60 / rpm_cap, latency_s)` (p50..p95) / 3600. Also the same projection for a pixel
   fraction `q` in {0.05, 0.10, 0.25, 0.5, 1.0} as input to H7 (section 17.9). The report also projects the
   expected number of `OK` val answers (`N_val_routed x OK rate`, with its Wilson range) against the
   100-answer calibration minimum (17.1.3, K13).
5. The report states verbatim: "The reporting floor (>= 100 objects in >= 20 pixels per class) is not
   reachable with 100 objects; all pilot numbers are descriptive." No ranking words (same
   `FORBIDDEN_WORDS` check as `layer2_report`).
6. `src/astro_triage/evaluation/` must not import `astro_triage.dashboard`; the Wilson interval (~6 lines)
   is duplicated from `dashboard/data.py:wilson`, not extracted (second use; rule of three).

Reason: pilot runs use `--sample` and write no parquet by design, so `layer2_evaluate.py` cannot report
them; D6.2 had no implementing file.

Acceptance: on a synthetic release with a pre-filled cache and ledger: zero network access (opener not
importable from the module); `pilot_report.json` has one entry per (model, arm, thinking) with the fields
above; an object missing from the cache shows in `pending`; Wilson bounds match a hand-computed case
(k = 90, n = 100); the floor sentence is present; with `rpm_cap = 30` and latency 1 s, hours =
`N x 2 / 3600`.

### 17.8 Security backlog (MINOR findings; not this cycle)

| id | Item | Acceptance when done |
|---|---|---|
| SEC-B1 | `client._opt_int` accepts negative `usage` values; `QuotaGuard.record` would then **lower** `tokens_done`. Accept only ints >= 0 (else `None`, counted as 0) | negative or bool usage -> `None`; `tokens_done` never decreases |
| SEC-B2 | Total deadline per call: `timeout_s` is per socket operation, so a slow-drip response can exceed 60 s. Enforce a wall-clock deadline per attempt (60 s) and per `chat` call (attempts + backoff, e.g. 300 s) -> `TIMEOUT` | fake opener that drips past the deadline -> `TIMEOUT`, retry rules unchanged |
| SEC-B3 | Bound server free strings: `model_reported` and `finish_reason` truncated to 64 chars and filtered to `[A-Za-z0-9._:/-]` (others replaced by `?`) before cache, ledger, parquet, report or dashboard | 10 kB `model` value -> stored length <= 64; control characters absent |
| SEC-B4 | `.gitignore` gains `*.zip` and `NUL` (Windows artefact present in the working tree) | `git check-ignore` matches both; **the orchestrator edits `.gitignore`**, not this ADR |

Not measured: whether the endpoint ever returns negative usage, a redirect, or oversized strings.

### 17.9 Scale on full_v0 (H7 open)

Normative:
1. Order of magnitude only (from the 100k test fold, **not** used to fix any threshold): ~11% of the test
   fold routed, i.e. ~32,000 objects on full_v0. `tau` for full_v0 comes from its own val fold
   (section 4.2) and nothing else (done: `tau = 0.86`, section 17.11; the test-fold routed count on
   full_v0 at that `tau` is not measured here and must not be used to revise it).
2. A full_v0 batch run **requires** a seeded random subset of **level-5 test pixels**: pixels drawn with
   `default_rng(20261025)` from the sorted list of test pixels (the committed pixel map), without looking
   at labels, Layer 1 outcomes or Layer 2 results; **all** routed objects in each chosen pixel are called;
   the evaluation population is **all** test objects in the chosen pixels (routed or not), so S0..S3 are
   compared on the same objects and the pixel bootstrap stays valid. The val fold, used only for
   calibration and `tau2`, may be subset the same way (independent draw, same seed derivation recorded).
3. The subset is frozen in `data/interim/<release>/layer2/pixel_subset_v1.json` (seed, number of pixels,
   sorted pixel list, sha256 of the routing manifest, created_utc), whose sha256 joins the parquet
   metadata; runner, consolidator and evaluator refuse a mismatch.
4. Explicit `max_calls` and `max_tokens_total` are mandatory as always (D5.2) and are sized from the pilot
   projection (17.7.4).
5. The number of pixels (or fraction `q`), `max_calls` and `max_tokens_total` are **human decision H7**,
   taken after the pilot report. The concrete file plan for 2-3 is written when H7 is answered (it is not
   in 17.10); until then no full_v0 test run is allowed.

Reason: at 30 rpm ~32,000 test calls alone are ~18 h of pacing (32,000 x 2 s), before val, retries and
canaries; the floor needs pixels, not objects, so subsetting pixels keeps the CI valid at lower cost.

Acceptance (when implemented): the same seed yields the same pixel list; the list is independent of
labels (test: shuffling labels leaves it unchanged); the evaluator's `n_test` equals the object count of
the chosen pixels.

### 17.10 File plan (Amendment 1)

```
FILE: src/astro_triage/evaluation/layer2_data.py            [modify]
INTENT: tau2 infeasible -> +inf with tau2_status; uncalibrated status; thinking-aware loading; fit audit shas
EXPORTS:
  class Fit: + tau2_status: str; val_parquet_sha256: str; routing_manifest_sha256: str
  def fit_on_val(val: FoldData, tau: float, a_keep2: float) -> Fit        # no EvalError on infeasible
  def load_fold(release_dir: Path, fold: str, manifest: Path, model: str, arm: str,
                enable_thinking: bool) -> FoldData
  def run_evaluation(release_dir: Path, model: str, arm: str, enable_thinking: bool,
                     n_boot: int = M.N_BOOT) -> dict[str, object]          # a_keep2 parameter removed
  TAU2_FITTED: Final = "fitted"
  TAU2_INFEASIBLE: Final = "infeasible: no grid value reaches A_keep2"
  TAU2_UNCALIBRATED: Final = "not_applied: uncalibrated (fewer than 100 OK val answers)"
DEPENDS: unchanged closed list
INVARIANTS: fit uses val only; a_keep2 read only from routing_v1.json; Fit.tau2 is float('inf') iff
  infeasible, None iff uncalibrated; every 17.1.5 refusal still raises EvalError
ACCEPTANCE: 17.1, 17.2, 17.6 acceptance items
RISK: inf leaking into JSON as null without tau2_status; readers branching on null
```

```
FILE: src/astro_triage/evaluation/layer2.py                 [modify]
INTENT: forced no_gain verdict when tau2 is infeasible
EXPORTS:
  def headline_verdict(diffs: dict[str, dict[str, dict[str, object]]],
                       forced_reason: str | None = None) -> dict[str, object]
  def evaluate(inp: Layer2Inputs, n_boot: int = M.N_BOOT, forced_reason: str | None = None) -> dict
DEPENDS: unchanged
INVARIANTS: HEADLINE_METRIC and HEADLINE_PAIRS unchanged; forced_reason never turns no_gain into gain
ACCEPTANCE: forced_reason set -> outcome "no_gain" even with a synthetic positive lo
RISK: none beyond signature change in callers
```

```
FILE: src/astro_triage/evaluation/layer2_report.py          [modify]
INTENT: print tau2_status and forced_reason; uncalibrated notice
EXPORTS: unchanged
DEPENDS: unchanged
INVARIANTS: no ranking words; never formats fit["tau2"] when it is null
ACCEPTANCE: markdown for each of the three tau2_status values renders without error and contains it
RISK: f-string formatting of None
```

```
FILE: scripts/layer2_evaluate.py                            [modify]
INTENT: add --thinking; remove --a-keep2
EXPORTS: main(argv) -> int (0 written, 1 refused/failed)
DEPENDS: unchanged
INVARIANTS: no CLI path changes A_keep2, tau, tau2 or tau*
ACCEPTANCE: 17.2, 17.6
RISK: none
```

```
FILE: src/astro_triage/llm/consolidate.py                   [modify]
INTENT: thinking segment in the parquet name
EXPORTS: def output_path(release_dir: Path, fold: str, model: str, arm: str, *, enable_thinking: bool) -> Path
DEPENDS: unchanged
INVARIANTS: name = predictions_l2_<slug>_<pv>_<arm>_think-<on|off>_<fold>.parquet
ACCEPTANCE: 17.2
RISK: callers left on the old signature (keyword required, so they fail loudly)
```

```
FILE: src/astro_triage/llm/runner.py                        [modify]
INTENT: canary classification agree/disagree/inconclusive; enable_thinking in ledger entries
EXPORTS: run(...) unchanged signature; report["canary"] per 17.5.4; stop_reason may be "CANARY_INCONCLUSIVE"
DEPENDS: unchanged
INVARIANTS: canary never writes the cache; no replacement canary objects
ACCEPTANCE: 17.5
RISK: an exception inside the canary skipping the report fields
```

```
FILE: src/astro_triage/cost/quota.py                        [modify]
INTENT: allow enable_thinking in the ledger
EXPORTS: LEDGER_KEYS gains "enable_thinking"
DEPENDS: unchanged
INVARIANTS: allowlist still rejects any other key
ACCEPTANCE: ledger line with enable_thinking accepted; unknown key still raises
RISK: none
```

```
FILE: scripts/layer2_run.py                                 [modify]
INTENT: exit-code table of 17.3
EXPORTS: EXIT_LOCAL = 1; EXIT_FATAL = 2; EXIT_QUOTA = 3; EXIT_INCOMPLETE = 4
DEPENDS: unchanged
INVARIANTS: DRIFT -> 2; CONSECUTIVE_RETRYABLE, CANARY_INCONCLUSIVE, pending > 0 -> 4; docstring lists codes
ACCEPTANCE: 17.3
RISK: none
```

```
FILE: src/astro_triage/llm/cache.py                         [modify]
INTENT: read-only loader for reports
EXPORTS: def load_records(path: Path) -> dict[str, dict[str, object]]   # last record per key, no lock
DEPENDS: unchanged
INVARIANTS: never creates the lock or writes; truncated/invalid lines skipped with a warning
ACCEPTANCE: works while a ResultCache holds the lock; same records as ResultCache.get
RISK: reading a line being appended (skipped as truncated)
```

```
FILE: src/astro_triage/evaluation/layer2_pilot.py           [create]
INTENT: pilot report numbers and projection from cache + ledger (17.7)
EXPORTS:
  def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]
  def pilot_report(release_dir: Path, models: Sequence[str], arms: Sequence[str],
                   thinking_models: Sequence[str], sample: int, target_release_dir: Path,
                   rpm_cap: float) -> dict[str, object]
  def render_pilot_markdown(rep: Mapping[str, object]) -> str
DEPENDS: json, math, pathlib, collections.abc, numpy, pandas, astro_triage.llm.{cache,routing},
  astro_triage.abstention.policy, astro_triage.split, eval.metrics (closed list; no llm.client, no dashboard)
INVARIANTS: no network; val only for accuracy; target test fold read for row counts only; every sampled
  object reported (pending if missing)
ACCEPTANCE: 17.7
RISK: ledger lines without enable_thinking; module size (keep < 300 lines)
FILE: scripts/layer2_pilot_report.py                        [create]
INTENT: CLI over pilot_report; writes pilot_report.json/.md via temp+rename; exit 0/1
```

Deferred to H7: the pixel-subset implementation of 17.9.2-3. Backlog: SEC-B1..B4 (17.8).

### 17.11 First pilot evidence (full_v0, val only; reported by the orchestrator 2026-10-27)

Facts as reported to the architect (not recomputed by it):
- `data/interim/full_v0/layer2/routing_v1.json` generated 2026-10-27 from val only: `tau = 0.86`, val
  routed fraction 8.96% (below `R_max = 0.15`), Layer 1 val accuracy on kept objects 0.9904
  (`>= A_keep = 0.99`).
- Pilot, model `nvidia/nemotron-3-ultra-550b-a55b` only, 100 val routed objects (blind arm; thinking off
  assumed from the default, not stated): **90 `UNKNOWN`, 10 `OK`**, 0 other statuses; 108 HTTP attempts;
  48,119 tokens total (~481 tokens per object, ~446 per attempt).
- Not measured / not reported: accuracy of the 10 `OK` answers; latency; the split of the 8 extra attempts
  between retries and canary; 429 count; lightning and bolt arms; thinking arm; `with_l1`; memorization
  probe. Two ledgers exist (`calls_20261027T135902Z.jsonl`, `calls_20261027T140424Z.jsonl`); which run
  produced which count is not stated.

Consequences (no rule changes; the rules above already cover these):
1. Calibration: at a 10% `OK` rate, >= 100 `OK` val answers require ~1,000 routed val objects called;
   whether the full_v0 val routed set (8.96% of val) is that large is not stated here. If not, the
   evaluation takes the "not applied" branch (17.1.3, T7, K13).
2. S3 coverage of routed objects is at most the `OK` rate (~10% on this sample), so S3 sits close to S1;
   the negative-result rule (D7.8) is the most likely outcome, and was fixed before this observation.
3. Tokens: ~481 tokens per object is above the ~20 output tokens measured on an invented prompt
   (section 1); the difference is prompt tokens plus attempts, and is what the projection (17.7.4) must
   use, not the section 1 figure.
4. Model choice (D1.1) is still open until the other two models run on the same 100 objects; one model is
   not a bake-off. Gate G-P is not passed.

## 18. Amendment 2 (2026-11-02): closure without test-fold evaluation

Task T-L2-03, branch A of human decision H-L2-03. Scope: closes Layer 2 on the descriptive val pilot,
records the human decisions of 2026-11-02, closes H2, H3, H5, H6 and H7, and defines how Layer 2 may be
reopened. **No rule of sections 1-17 is changed**; those sections stay as the record of what would govern a
test evaluation. Every item has normative text, reason and verifiable acceptance. Nothing in this amendment
uses, or was informed by, any test-fold Layer 2 number: none exists (18.4).

### 18.0 Human decisions (2026-11-02)

Recorded verbatim from `.claude/state/T-L2-03.json` (`human_decisions_2026_11_02`), then the answer to
H-L2-03:

| # | Decision (verbatim) | Recorded as |
|---|---|---|
| 1 | "close Layer 2 as negative per ADR 0005 D7.8 / 17.1" | Qualified by the H-L2-03 answer below: the closure is taken on the val pilot, so "negative" here is a descriptive judgement, **not** the D7.8 or 17.1 outcome (18.1) |
| 2 | "no extra bake-off, no prompt v2, no lightning run" | No further model, prompt or arm is run (18.3) |
| 3 | "VLM follow-up does not start" | Human decision, not the D7.8 consequence (18.1.4) |
| 4 | "no IC deadline" | Recorded; no rule of this ADR depends on it |
| 5 | "DL compute later on Kaggle GPU = ADR 0006, separate task, not started here" | Out of scope of this ADR |
| H-L2-03 | Branch A: "close on val pilot, test fold untouched" | 0 API calls, 0 API hours; test fold of full_v0 stays unexposed (18.4) |

### 18.1 What the closure is, and what it is not

Normative:
1. Layer 2 (text, Nemotron through the NVIDIA API) is **closed on descriptive evidence** from the 100-object
   val pilot of full_v0 (18.2). No batch run, no test-fold run and no run of `scripts/layer2_evaluate.py`
   follows from this ADR.
2. **The D7.8 negative-result rule was NOT executed.** No headline pair (S2-S0, S3-S1\*) was computed, no
   paired CI on test exists, no floor check on test was made, and no `verdict.outcome` exists. This closure
   must not be called, cited or reported as a pre-registered negative result, as a D7.8 verdict, or with
   the D7.8 sentence "adds no measurable value". Admissible wording: "Layer 2 closed without test
   evaluation, on a descriptive 100-object val pilot (ADR 0005, section 18)."
3. **17.1.2 (`tau2` infeasible, forced `no_gain`) was NOT triggered.** It requires the verbal calibration,
   which requires >= 100 `OK` val answers (17.1.1). The pilot `OK` counts on val were: ultra blind **10**,
   bolt blind **50** (with_l1: ultra 74, bolt 87), each below the 100 minimum. `tau2` was never fitted,
   `fit.tau2_status` was never computed, and no `fit` block exists. The "not applied" branch (17.1.3) was
   not evaluated either: it is an evaluator outcome and the evaluator was not run.
4. The VLM follow-up is not started by human decision 18.0 item 3. D7.8 makes it conditional on a negative
   text result established by D7.8, which does not exist; the reason recorded is the human decision.
5. Gate G-P (section 12) was never passed and the ADR was never Accepted: 12.3 requires a chosen model and
   none was chosen (H2 not pursued, 18.3).

Reason: the human chose branch A of H-L2-03 (0 calls, test fold untouched). D7.8 and 17.1.2 are defined on
a test-fold evaluation and on a fitted calibration; the pilot reaches neither, so the closure is a
documented deviation owned by the human, not an outcome of the pre-registered rule.

Acceptance: section 18 says "NOT executed" for D7.8 and "NOT triggered" for 17.1.2 with the counts 10 and
50 against 100; section 18 contains no test-fold metric, no test CI and no `verdict.outcome`; the Status
line reads "Closed without test evaluation" and names Amendment 2.

### 18.2 Descriptive evidence (val pilot, full_v0)

Source: `data/interim/full_v0/layer2/pilot_report.md`, copied, not recomputed. 100 val routed objects drawn
per D6.1, thinking off, `pending` 0 in every row. The report states verbatim: "The reporting floor (>= 100
objects in >= 20 pixels per class) is not reachable with 100 objects; all pilot numbers are descriptive."
**Every number below is descriptive and below the reporting floor; none is a headline metric; none is on
test.**

| model | arm | status counts | acc forced [95% CI] | acc selective [95% CI] | macro-F1 forced | macro-F1 L1 | UNKNOWN | L2 = L1 argmax |
|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b-a55b | blind | {"OK": 10, "UNKNOWN": 90} | 0.070 [0.034, 0.137] | 0.700 [0.397, 0.892] (n=10) | n/a | 0.721 | 0.900 | 0.600 |
| nvidia/nemotron-3-ultra-550b-a55b | with_l1 | {"OK": 74, "UNKNOWN": 26} | 0.530 [0.433, 0.625] | 0.716 [0.605, 0.806] (n=74) | 0.628 | 0.721 | 0.260 | 1.000 |
| nvidia/nemotron-3.5-bolt-e4b | blind | {"OK": 50, "UNKNOWN": 50} | 0.230 [0.158, 0.322] | 0.460 [0.330, 0.596] (n=50) | n/a | 0.721 | 0.500 | 0.460 |
| nvidia/nemotron-3.5-bolt-e4b | with_l1 | {"OK": 87, "UNKNOWN": 13} | 0.580 [0.482, 0.672] | 0.667 [0.562, 0.757] (n=87) | 0.620 | 0.721 | 0.130 | 0.920 |

Also from the same report (descriptive): paired val pixel bootstrap of macro_f1_forced, ultra minus bolt:
blind "n/a [n/a, n/a]"; with_l1 "0.008 [-0.034, 0.047]".

Read-off, without inference beyond the table:
- Only `OK` and `UNKNOWN` statuses occur in the four rows; no failure status was recorded.
- Where macro-F1 forced is reported (with_l1), its point value (0.628, 0.620) is below the Layer 1 macro-F1
  on the same objects (0.721). A CI on the Layer 2 minus Layer 1 difference was **not measured**. In the
  blind arm macro-F1 forced is "n/a" in the report.
- In with_l1 the Layer 2 label equals the Layer 1 argmax at 1.000 (ultra) and 0.920 (bolt); whether this
  is anchoring (K8) was **not measured** beyond this rate.

Not measured: any test-fold quantity; `tau2` and the verbal calibration; lightning (called on the 100 val pilot
objects, blind, 2026-10-27, but absent from `pilot_report.md`; decision 18.0 item 2 only rules out further
lightning runs); the thinking-on arm (started, ultra blind, 66 attempts over 2 runs with many TIMEOUTs, never
finished and absent from `pilot_report.md`); the memorization probe (K3); T3 error complementarity; T4's blind
macro-F1 CI against Layer 1.

Acceptance: every number in the table and in the bootstrap line equals the corresponding cell of
`pilot_report.md`; the floor sentence is quoted verbatim.

### 18.3 Open human decisions closed as not pursued (2026-11-02)

| id | Question (section 11) | Closed 2026-11-02 as | Reason |
|---|---|---|---|
| H2 | Model choice when the top two are tied | Not pursued (moot) | No batch or test evaluation follows, so no model is needed. The with_l1 paired difference 0.008 [-0.034, 0.047] (18.2) is recorded; no model is chosen. |
| H3 | Run and adopt the thinking-on arm | Not pursued (moot) | Human decision "no extra bake-off" (18.0 item 2); the arm was started but never finished (ultra blind, 66 attempts over 2 runs, many TIMEOUTs) and is not in `pilot_report.md`; its metrics are not measured. |
| H5 | `with_l1` on test | Not pursued (moot) | No test run (H-L2-03 branch A). |
| H6 | Action if the memorization probe is positive | Not pursued (moot) | The probe was never run; K3 is not measured. |
| H7 | full_v0 batch scope: pixels, `max_calls`, `max_tokens_total` | Not pursued (moot) | No batch run. The 17.9.2-3 pixel subset was not implemented; `pixel_subset_v1.json` does not exist. |

T4's allowed "one new prompt version (few-shot from train)" is not used (human decision "no prompt v2").

Acceptance: each of H2, H3, H5, H6, H7 appears in this table with "Not pursued" and the date 2026-11-02.

### 18.4 The test fold was never seen by Layer 2

Normative: the full_v0 test fold was never sent to the NVIDIA API, never consolidated and never evaluated
by Layer 2, and this amendment does not change that. Consequently the section 15 irreversibility (test-fold
exposure of a prompt version and `tau`) was **not incurred**: `routing_v1.json` (`tau = 0.86`, sha256 in
17.6.3b) and the pilot prompt version have no test-fold exposure for Layer 2.

Evidence (2026-11-02): `data/interim/full_v0/layer2/` holds `routing_v1.json`, `pilot_report.json`,
`pilot_report.md` and seven `calls_*.jsonl` ledgers; there is **no** `eval_l2.json`, **no**
`predictions_l2_*.parquet` and **no** `pixel_subset_v1.json`. The ledgers carry no fold field (D5.6), so
the absence of test-fold calls inside them rests on the pilot procedure (D6.1: `--sample` on val) and the
orchestrator's record; a join of ledger `source_id` values to folds was **not measured** by the architect.

Reason: an unexposed test fold keeps the pre-registered evaluation available to a future ADR without the
section 15 declaration.

Acceptance: a listing of `data/interim/full_v0/layer2/` shows none of `eval_l2.json`,
`predictions_l2_*.parquet`, `pixel_subset_v1.json`. Optional, read-only, zero API calls: joining every
ledger `source_id` to the full_v0 split shows only val objects.

### 18.5 Reopening

Normative: Layer 2 is reopened only by a **new ADR** that cites this section and:
1. defines a **new prompt version** (bumped `PROMPT_VERSION`, new template sha256) and its model and
   reasoning setting;
2. evaluates it **first on val** (pilot, then calibration with >= 100 `OK` val answers or the explicit
   17.1.3 branch) before any test-fold call;
3. obtains explicit human approval of `max_calls` and `max_tokens_total` (CONTRACT §8, D5.2) and, on
   full_v0, of the pixel subset (17.9);
4. if it intends a pre-registered verdict, fixes its metric and pairs before any test number and states
   that, up to 2026-11-02, the full_v0 test fold had never been seen by Layer 2 (18.4).

Until such an ADR exists, `scripts/layer2_run.py` is not run under this ADR. This amendment orders no
code change and no deletion of code, cache, ledgers or pilot report.

Reason: the human ruled out a prompt v2 and further runs now; reopening must not re-use val-tuned choices
on test without a new pre-registration.

Acceptance: section 18.5 names a new ADR, a new prompt version and val-first evaluation as the only
reopening path.

### 18.6 Consequences and reversal

- Good: 0 API calls and 0 API hours spent by the closure; the full_v0 test fold stays unexposed for Layer 2;
  no claim is made beyond the descriptive pilot.
- Bad: there is no pre-registered answer to whether Layer 2 adds value on full_v0; the D7.8 and 17.1
  machinery (evaluator, forced verdict) is implemented and tested on synthetic data but was never run on
  real data; the evidence is 100 val objects below the reporting floor; lightning, thinking-on and the
  memorization probe are not measured.
- Maintenance cost: the Layer 2 modules (`src/astro_triage/{llm,cost,abstention}`,
  `src/astro_triage/evaluation/layer2*.py`, `scripts/layer2_*.py`, the dashboard Layer 2 section) and their
  tests stay in the repository unexercised; their behaviour when `eval_l2.json` is absent on the dashboard
  is **not measured** here; SEC-B1..B4 (17.8) stay in the backlog with no further calls to protect.
- Reversal trigger: a new ADR per 18.5.
