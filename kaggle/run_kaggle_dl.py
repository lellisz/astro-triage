"""Kaggle runner for the deep-learning axis (ADR 0006, spec S14): GPU notebook, stdlib only.

Notebook cell:
    import glob; exec(open(glob.glob('/kaggle/input/**/run_kaggle_dl.py', recursive=True)[0]).read())

Stages: pacote, pip, entradas, sessao_gpu, retomada, treino, ensemble, zip. It never freezes, never
evaluates and never uploads: `--freeze` is the owner's act. The batch is described by the package file
kaggle/kaggle_dl_mode.json (every key required, so a default run cannot start a 40-fit batch by accident).
Dry run (any machine, runs nothing, writes nothing):
    python kaggle/run_kaggle_dl.py --dry-run [--config PATH]
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime

INPUT = "/kaggle/input"
WORKING = "/kaggle/working"
REPO = f"{WORKING}/repo"
LOG_PATH = f"{WORKING}/run_log.txt"
ENV_PATH = f"{WORKING}/environment.json"
OUT_ZIP = f"{WORKING}/astro_triage_dl_outputs.zip"
PACKAGE_ZIP = "astro_triage_pkg.zip"
DATASET_DIR = f"{REPO}/data/interim/full_v0"
DL_DIR = f"{DATASET_DIR}/dl"
MODE_FILE = "kaggle/kaggle_dl_mode.json"
REPORT_NAME = "train_report_dl.json"
SEEDS_DIRNAME = "lgbm_seeds"
PREDICTION_VERSION = "v0"
OUT_ZIP_PATTERN = "astro_triage_dl_outputs*.zip"
N_SEEDS = 5
MIN_FREE_BYTES = 3 * 1024**3
# Order of ADR 0006 section 9.4 / spec P-gpu.
SYSTEMS = (("ftt", "full"), ("mlp", "full"), ("ftt", "noind"), ("mlp", "noind"))
MODE_KEYS = {"seeds", "max_epochs", "patience", "train_subset", "ensemble"}
PIP_SKIP = {"streamlit", "plotly", "pytest", "ruff", "duckdb", "astroquery", "torch"}
BANNED_NAMES = ("labels.parquet", "features.parquet")
ARTEFACT_RE = re.compile(
    r"^(?:predictions_(.+)_v0\.parquet(?:\.manifest\.json)?|models/(?:model_)?(.+?)"
    r"(?:\.pt|\.preproc\.pkl|\.manifest\.json))$"
)
SESSION_SNIPPET = (
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from astro_triage.models.dl_session import verify_gpu\n"
    "from astro_triage.models.lgbm import TrainError\n"
    "try:\n"
    "    verify_gpu(Path(sys.argv[2]), 'kaggle')\n"
    "except TrainError as exc:\n"
    "    raise SystemExit(f'session check failed: {exc}')\n"
)

T0 = time.time()
DRY = False


def log(message):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime(time.time() - T0))}] {message}"
    print(line, flush=True)
    if not DRY:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def fail(message):
    log(f"ERROR: {message}")
    raise SystemExit(f"astro-triage dl runner stopped: {message}")


def check_disk(stage):
    free = shutil.disk_usage(WORKING).free
    if free < MIN_FREE_BYTES:
        fail(f"{stage}: only {free / 1024**3:.1f} GB free in {WORKING} (< 3 GB)")


def run_stream(label, cmd, cwd=None):
    """Run a command, echo its output line by line into the log, stop on a non-zero exit."""
    log(f"{label}: {' '.join(cmd)}")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        fail(f"{label} exited with code {code}")


def timed(name, func):
    log(f"[{name}] start")
    started = time.time()
    try:
        return func()
    finally:
        log(f"[{name}] done in {time.time() - started:.1f}s")


def read_mode(path):
    try:
        with open(path, encoding="utf-8") as handle:
            mode = json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"[modo] cannot read {path}: {exc}")
    if not isinstance(mode, dict) or set(mode) != MODE_KEYS:
        fail(f"[modo] {path}: keys must be exactly {sorted(MODE_KEYS)}")
    seeds = mode["seeds"]
    ok = (
        isinstance(seeds, list)
        and seeds
        and len(set(seeds)) == len(seeds)
        and all(type(k) is int and 0 <= k < N_SEEDS for k in seeds)
    )
    if not ok:
        fail(f"[modo] 'seeds' must be distinct integers in 0..{N_SEEDS - 1}")
    for key in ("max_epochs", "patience"):
        if type(mode[key]) is not int or mode[key] < 1:
            fail(f"[modo] {key!r} must be an integer >= 1")
    subset = mode["train_subset"]
    if subset is not None and (type(subset) is not int or subset < 1):
        fail("[modo] 'train_subset' must be null or an integer >= 1")
    if type(mode["ensemble"]) is not bool:
        fail("[modo] 'ensemble' must be true or false")
    if subset is not None and mode["ensemble"]:
        fail("[modo] 'ensemble' is not allowed with 'train_subset' (pilot)")
    return mode


def finished_ids(dl_dir):
    """Model ids whose report entry has a prediction hash and whose parquet is on disk."""
    path = f"{dl_dir}/{REPORT_NAME}"
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as handle:
            fits = json.load(handle).get("fits", {})
    except (OSError, ValueError, AttributeError) as exc:
        fail(f"[treino] cannot read {path}: {exc}")
    return {
        mid
        for mid, entry in fits.items()
        if isinstance(entry, dict)
        and entry.get("prediction_sha256")
        and os.path.isfile(f"{dl_dir}/predictions_{mid}_{PREDICTION_VERSION}.parquet")
    }


def train_units(mode, done):
    """(arch, features, seeds still to fit) per system; finished ids are skipped."""
    units = []
    for arch, features in SYSTEMS:
        todo = [k for k in mode["seeds"] if f"{arch}_{features}_s{k}" not in done]
        if todo:
            units.append((arch, features, todo))
    return units


def train_cmd(arch, features, seeds, mode):
    cmd = [
        sys.executable,
        f"{REPO}/scripts/train_dl.py",
        "--outdir",
        DATASET_DIR,
        "--arch",
        arch,
        "--features",
        features,
        "--seeds",
        ",".join(map(str, seeds)),
        "--max-epochs",
        str(mode["max_epochs"]),
        "--patience",
        str(mode["patience"]),
        "--device",
        "cuda",
        "--platform",
        "kaggle",
    ]
    if mode["train_subset"] is not None:
        cmd += ["--train-subset", str(mode["train_subset"])]
    return cmd


def ensemble_cmd(arch, mode):
    return [
        sys.executable,
        f"{REPO}/scripts/train_dl.py",
        "--outdir",
        DATASET_DIR,
        "--ensemble",
        "--arch",
        arch,
        "--seeds",
        ",".join(map(str, mode["seeds"])),
        "--device",
        "cpu",
        "--platform",
        "kaggle",
    ]


def needed_inputs(mode):
    names = ["labels.parquet", "features.parquet"]
    for features in ("full", "noind"):
        for k in mode["seeds"]:
            names.append(
                f"{SEEDS_DIRNAME}/predictions_lgbm_{features}_s{k}_{PREDICTION_VERSION}.parquet"
            )
    return names


def find_package():
    zips = sorted(glob.glob(f"{INPUT}/**/{PACKAGE_ZIP}", recursive=True))
    marker = sorted(glob.glob(f"{INPUT}/**/scripts/train_dl.py", recursive=True))
    if os.path.isdir(REPO):
        shutil.rmtree(REPO)
    if zips:
        log(f"[pacote] extracting {zips[-1]}")
        with zipfile.ZipFile(zips[-1]) as archive:
            archive.extractall(REPO)
    elif marker:
        root = os.path.dirname(os.path.dirname(marker[-1]))
        log(f"[pacote] copying the extracted package from {root}")
        shutil.copytree(root, REPO)
    else:
        fail(f"[pacote] neither {PACKAGE_ZIP} nor scripts/train_dl.py under {INPUT}")
    for needed in ("scripts/train_dl.py", "requirements.txt", MODE_FILE):
        if not os.path.isfile(f"{REPO}/{needed}"):
            fail(f"[pacote] {needed} missing from the package")


def install_dependencies():
    packages = []
    with open(f"{REPO}/requirements.txt", encoding="utf-8") as handle:
        for raw in handle:
            spec = raw.split("#", 1)[0].strip()
            if spec and spec.split("==", 1)[0].strip().lower() not in PIP_SKIP:
                packages.append(spec)
    log(f"[pip] installing {packages}; torch comes from the Kaggle image, never pip")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
    ]
    result = subprocess.run(cmd + packages, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail(f"[pip] install failed (exit {result.returncode}):\n{result.stderr}")
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started_utc": datetime.fromtimestamp(T0, UTC).isoformat(timespec="seconds"),
        "pip_freeze": frozen,
    }
    with open(ENV_PATH, "w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)


def locate_inputs(names):
    """Map each needed relative name to ('file', path) or ('zip', archive, member); None if absent."""
    archives = {}
    for path in sorted(glob.glob(f"{INPUT}/**/*.zip", recursive=True)):
        try:
            with zipfile.ZipFile(path) as archive:
                archives[path] = set(archive.namelist())
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"[entradas] cannot read {path}: {exc}")
    found = {}
    for rel in names:
        hits = sorted(glob.glob(f"{INPUT}/**/full_v0/{rel}", recursive=True))
        if rel.startswith(f"{SEEDS_DIRNAME}/"):
            hits += sorted(glob.glob(f"{INPUT}/**/{rel}", recursive=True))
        if hits:
            found[rel] = ("file", hits[-1])
            continue
        for path, members in archives.items():
            for member in (f"full_v0/{rel}", rel):
                if member in members and (member.startswith("full_v0/") or "/" in rel):
                    found[rel] = ("zip", path, member)
        found.setdefault(rel, None)
    return found


def stage_inputs(mode):
    found = locate_inputs(needed_inputs(mode))
    missing = sorted(rel for rel, src in found.items() if src is None)
    if missing:
        fail(
            f"[entradas] not found under {INPUT}: {missing}. Add the Phase 1 notebook output (labels, "
            "features) and the seeds-mode output (lgbm_seeds predictions) as inputs. Rebuilding "
            "full_v0 from SDSS/XMatch is not implemented in this runner (spec gap G14)."
        )
    check_disk("[entradas]")
    for rel, src in found.items():
        target = f"{DATASET_DIR}/{rel}"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if src[0] == "file":
            shutil.copy(src[1], target)
        else:
            with (
                zipfile.ZipFile(src[1]) as archive,
                archive.open(src[2]) as s,
                open(target, "wb") as d,
            ):
                shutil.copyfileobj(s, d)
        log(f"[entradas] {rel} <- {src[1]}")


def check_session():
    code = subprocess.run(
        [
            sys.executable,
            "-c",
            SESSION_SNIPPET,
            f"{REPO}/src",
            f"{DL_DIR}/environment.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for row in (code.stdout + code.stderr).splitlines():
        log(f"[sessao_gpu] {row}")
    if code.returncode != 0:
        fail("[sessao_gpu] GPU/torch check failed; nothing was trained")


def merge_report(fits, envs, raw, origin):
    try:
        doc = json.loads(raw)
        found, found_envs = doc["fits"], doc.get("environment", {})
        if not isinstance(found, dict) or not isinstance(found_envs, dict):
            raise TypeError("fits/environment is not an object")
    except (ValueError, KeyError, TypeError) as exc:
        log(f"[retomada] ignoring unreadable report in {origin}: {exc}")
        return
    for key, entry in found.items():
        fits.setdefault(key, entry)
    for key, entry in found_envs.items():
        envs.setdefault(key, entry)


def restore_outputs():
    """Copy dl/ (predictions, models, ckpt, reports) from earlier outputs; never overwrite."""
    path = f"{DL_DIR}/{REPORT_NAME}"
    fits, envs = {}, {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            merge_report(fits, envs, handle.read(), path)
    copied = 0
    sources = []
    for src in sorted(glob.glob(f"{INPUT}/**/dl", recursive=True)):
        if os.path.isdir(src) and any(
            os.path.exists(f"{src}/{n}")
            for n in (REPORT_NAME, "ckpt", "environment.json")
        ):
            sources.append(("dir", src))
    for src in sorted(glob.glob(f"{INPUT}/**/{OUT_ZIP_PATTERN}", recursive=True)):
        sources.append(("zip", src))
    for kind, src in sources:
        try:
            if kind == "dir":
                items = [
                    (os.path.join(r, n), os.path.relpath(os.path.join(r, n), src))
                    for r, _d, files in os.walk(src)
                    for n in files
                ]
            else:
                with zipfile.ZipFile(src) as archive:
                    items = [
                        (m, m[3:])
                        for m in archive.namelist()
                        if m.startswith("dl/") and not m.endswith("/")
                    ]
            for origin, rel in items:
                parts = rel.replace(os.sep, "/").split("/")
                if (
                    "" in parts
                    or ".." in parts
                    or parts[0] == "pilot"
                    or parts[-1].startswith(".")
                    or parts[-1].endswith(".tmp")
                ):
                    continue
                if parts == [REPORT_NAME]:
                    if kind == "dir":
                        with open(origin, encoding="utf-8", errors="replace") as handle:
                            merge_report(fits, envs, handle.read(), origin)
                    else:
                        with zipfile.ZipFile(src) as archive:
                            merge_report(
                                fits,
                                envs,
                                archive.read(origin).decode("utf-8", "replace"),
                                src,
                            )
                    continue
                target = os.path.join(DL_DIR, *parts)
                if os.path.exists(target):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if kind == "dir":
                    shutil.copy(origin, target)
                else:
                    with (
                        zipfile.ZipFile(src) as archive,
                        archive.open(origin) as s,
                        open(target, "wb") as d,
                    ):
                        shutil.copyfileobj(s, d)
                copied += 1
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"[retomada] cannot read {src}: {exc}")
    if fits or envs:
        os.makedirs(DL_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=DL_DIR, prefix=".report.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"environment": envs, "fits": fits}, handle, indent=2)
        os.replace(tmp, path)
    orphans = set()
    for root, _dirs, files in os.walk(DL_DIR):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), DL_DIR).replace(os.sep, "/")
            match = ARTEFACT_RE.match(rel)
            if match and not rel.startswith("pilot/"):
                mid = match.group(1) or match.group(2)
                if mid not in fits:
                    orphans.add(mid)
    if orphans:
        fail(
            "[retomada] artefacts without a report entry in any earlier source (train_dl would "
            f"refuse to overwrite them): {sorted(orphans)}"
        )
    log(f"[retomada] {copied} files restored from earlier outputs")


def train(mode):
    for arch, features, seeds in train_units(mode, finished_ids(DL_DIR)):
        check_disk("[treino]")
        run_stream(
            f"[treino] {arch}_{features}", train_cmd(arch, features, seeds, mode)
        )


def build_ensembles(mode):
    done = finished_ids(DL_DIR)
    for arch in ("mlp", "ftt"):
        if any(f"{arch}_full_s{k}" not in done for k in mode["seeds"]):
            log(f"[ensemble] {arch}: not every {arch}_full seed is finished, skipped")
            continue
        run_stream(f"[ensemble] {arch}", ensemble_cmd(arch, mode))


def zip_members(mode):
    """(path, arc) pairs: dl/ tree minus pilot and finished checkpoints; pilot batches ship the report only."""
    members = []
    if mode["train_subset"] is not None:
        report = f"{DL_DIR}/pilot/{REPORT_NAME}"
        if os.path.isfile(report):
            members.append((report, f"dl/pilot/{REPORT_NAME}"))
    else:
        done = finished_ids(DL_DIR)
        for root, _dirs, files in os.walk(DL_DIR):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), DL_DIR).replace(
                    os.sep, "/"
                )
                parts = rel.split("/")
                if name.startswith(".") or name.endswith(".tmp") or parts[0] == "pilot":
                    continue
                if parts[0] == "ckpt" and len(parts) > 1 and parts[1] in done:
                    continue
                members.append((os.path.join(root, name), f"dl/{rel}"))
    for path, arc in ((ENV_PATH, "environment.json"), (LOG_PATH, "run_log.txt")):
        if os.path.isfile(path):
            members.append((path, arc))
    for _path, arc in members:
        if arc.rsplit("/", 1)[-1] in BANNED_NAMES or SEEDS_DIRNAME in arc.replace("\\", "/").split("/"):
            fail(f"[zip] refusing to package {arc}")
    return members


def package_outputs(mode):
    members = zip_members(mode)
    check_disk("[zip]")
    tmp = OUT_ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
        for path, arc in members:
            stored = path.endswith((".parquet", ".pt"))
            archive.write(
                path,
                arc,
                compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
            )
    os.replace(tmp, OUT_ZIP)
    keep = (os.path.basename(OUT_ZIP), os.path.basename(LOG_PATH))
    for name in os.listdir(WORKING):
        if name not in keep:
            path = f"{WORKING}/{name}"
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    log(f"[zip] {OUT_ZIP}: {os.path.getsize(OUT_ZIP)} bytes, {len(members)} files")


def print_plan(mode, config_path):
    log(
        f"[plano] dry run: nothing is executed or written; mode file {config_path}: {mode}"
    )
    log(
        "[plano] stages: pacote, pip, entradas, sessao_gpu, retomada, treino, ensemble, zip"
    )
    log(f"[plano] pip: requirements.txt minus {sorted(PIP_SKIP)}")
    log(f"[plano] entradas required under {DATASET_DIR}: {needed_inputs(mode)}")
    log(
        "[plano] sessao_gpu: verify_gpu (aborts on P100/sm_60, CPU build, missing arch)"
    )
    done = finished_ids(DL_DIR)
    for arch, features, seeds in train_units(mode, done):
        log("[plano] " + " ".join(train_cmd(arch, features, seeds, mode)))
    if mode["ensemble"]:
        for arch in ("mlp", "ftt"):
            log("[plano] (if all seeds finished) " + " ".join(ensemble_cmd(arch, mode)))
    log(
        f"[plano] zip: {OUT_ZIP} (dl/ tree, unfinished ckpt, environment.json, run_log.txt)"
    )
    log("[plano] never run: --freeze, evaluate_dl, any Kaggle upload")


def main(argv=None):
    global DRY
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config", default=None, help="mode file (default: package copy)"
    )
    args, _unknown = parser.parse_known_args(argv)
    DRY = args.dry_run
    if DRY:
        path = args.config or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODE_FILE
        )
        print_plan(read_mode(path), path)
        return
    if not os.path.isdir(INPUT) or not os.path.isdir(WORKING):
        raise SystemExit("this runner only works inside a Kaggle notebook")
    check_disk("start")
    timed("pacote", find_package)
    mode = read_mode(f"{REPO}/{MODE_FILE}")
    timed("pip", install_dependencies)
    timed("entradas", lambda: stage_inputs(mode))
    timed("sessao_gpu", check_session)
    try:
        timed("retomada", restore_outputs)
        timed("treino", lambda: train(mode))
        if mode["ensemble"]:
            timed("ensemble", lambda: build_ensembles(mode))
    finally:
        timed("zip", lambda: package_outputs(mode))


if __name__ == "__main__":
    main()
