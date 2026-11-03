# Running phase 1 on Kaggle

Full pipeline on the 4,283,507 SDSS candidates (CDS XMatch, dataset `full_v0`, LightGBM, evaluation).
CPU only. Needs about 7 GB in `/kaggle/temp` (SDSS file, not part of the output).

1. Build the package on your machine:
   `.venv\Scripts\python.exe -I scripts\make_kaggle_package.py --outdir <some folder>`
   It writes `astro_triage_pkg.zip` (under 5 MB) and prints its SHA256.
2. Kaggle > Datasets > New Dataset, **private**, upload `astro_triage_pkg.zip`.
3. Kaggle > Code > New Notebook, add that dataset as input. Session options: **Accelerator: None**,
   **Internet: On**.
4. Put this in one cell:
   ```python
   import glob

   exec(open(glob.glob("/kaggle/input/**/run_kaggle.py", recursive=True)[0]).read())
   ```
5. **Save Version > Save & Run All (Commit)**. It keeps running with your PC off. Follow the markers in
   the log: `[pacote]`, `[pip]`, `[specobj]`, `[candidatos]`, `[xmatch lote i/N catalogo]`, `[dataset]`,
   `[treino]`, `[avaliação]`, `[zip]`.

## Resuming

If the session stops (time limit, XMatch failure), open the notebook, **Add Input > Notebook output**,
pick the earlier version of the same notebook, and save a new version. Finished XMatch chunks
(`xmatch_full/` or inside `astro_triage_outputs.zip`) are copied back and skipped. The run refuses to
mix chunks made with a different chunk size, radius or candidates file.

## What to download

From the finished version's Output tab: `astro_triage_outputs.zip` (reports, predictions, models,
`labels.parquet` and `features.parquet` of `full_v0`, XMatch chunks, manifests, environment versions,
run log) and `run_log.txt`.
