# CLAUDE.md

## Project
astro-triage classifies Gaia DR3 sources as STAR/GALAXY/QSO using SDSS spectroscopic labels (LightGBM + calibration, Gaia DSC as baseline).
Authority: `docs/adr/` and `.claude/CONTRACT.md` (agent protocol). The CONTRACT wins in a conflict.

## Hard rules (owner's decisions)
- No AWS, Bedrock or any paid provider without a new ADR.
- Kaggle: work only in private resources named `astro-triage-*`, code only, no raw data. Never read, modify, version or delete anything else on the account. Every write to Kaggle or download of its output needs the owner's explicit yes in the conversation.
- Git: no commit, push or branch change without the owner's explicit approval in the conversation. Use conventional commits. Commits carry the Co-Authored-By trailer (owner's decision of 2026-11-16, confirmed 2026-11-17).
- LLM/API calls: only the NVIDIA API (ADR 0005, which is closed: Layer 2 was closed without test evaluation, Amendment 2 section 18). The key comes only from the environment or `.env`; never open, read or print it. Any run beyond the 100-object val pilot needs the owner's explicit approval of `max_calls` and `max_tokens_total`. A reopening needs a new ADR.
- Raw data and outputs never enter git (`data/`, `logs/`, `*.parquet`, `*.duckdb`, `graphify-out/`).

## Status
Every metric stays PROVISIONAL because: (1) several cells are below the reporting floor, (2) ADR 0006 (deep-learning comparison protocol) is still Proposed: the validation freeze exists and the test was opened once (2026-11-16), but acceptance is open. (The stale "PROVISIONAL: 100k sample" header of the `full_v0` evaluation text was fixed on 2026-11-03.) The population has no magnitude cut (ADR 0001 amendment). Do not present numbers as final.

## Environment
Windows/PowerShell, Python 3.12. Always use `.venv\Scripts\python.exe`. Run tests with pytest.

## Navigation
If `graphify-out/` exists, run `graphify query "<topic>"` BEFORE any Read/Grep/Glob in `src/` or `docs/` (a query costs ~2k tokens; reading a whole ADR ~16k). Read `GRAPH_REPORT.md` only to orient, not instead of querying. If `graph.json` is older than the last commit, run `/graphify . --update` first. The graph is not committed.
