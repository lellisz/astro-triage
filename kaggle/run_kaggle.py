"""Kaggle runner for the astro-triage phase 1 pipeline (CPU, internet on, no arguments).

Notebook cell:
    import glob; exec(open(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0]).read())

Stages: package, pip, SDSS specObj download, candidates, XMatch (resumable), dataset, training,
evaluation (if scripts/evaluate.py exists), one output zip. Large files live in /kaggle/temp
(not part of the output). /kaggle/working ends with only the output zip and the run log.

Seeds mode (only when kaggle/kaggle_mode.json is in the package): same stages up to the dataset,
then a parity check against kaggle/full_v0_reference.json, then the LightGBM seed fits
(docs/specs/kaggle-seed-run.md). Without that file the phase 1 path above runs unchanged.
"""

import glob
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime

INPUT = "/kaggle/input"
WORKING = "/kaggle/working"
TEMP = "/kaggle/temp"
REPO = f"{WORKING}/repo"
XMATCH_DIR = f"{WORKING}/xmatch_full"
XMATCH_WORK = f"{TEMP}/xmatch_work"
LOG_PATH = f"{WORKING}/run_log.txt"
ENV_PATH = f"{WORKING}/environment.json"
OUT_ZIP = f"{WORKING}/astro_triage_outputs.zip"
PACKAGE_ZIP = "astro_triage_pkg.zip"
DATASET_NAME = "full_v0"
DATASET_DIR = f"{REPO}/data/interim/{DATASET_NAME}"
MODE_FILE = "kaggle/kaggle_mode.json"
REFERENCE_FILE = "kaggle/full_v0_reference.json"
PARITY_REPORT = f"{WORKING}/parity_report.json"
SEEDS_RUN_PATH = f"{WORKING}/seeds_run.json"
SEEDS_DIRNAME = "lgbm_seeds"
SEEDS_REPORT_NAME = "train_report_lgbm_seeds.json"
MEASURE_ONE_ID = "lgbm_full_s0"
N_SEEDS = 5

FITS_URL = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits"
FITS_BYTES = 6741023040
FITS_PATH = f"{TEMP}/specObj-dr17.fits"
CANDIDATES_PATH = f"{TEMP}/specobj_candidates_v1.parquet"
EXPECTED_COUNTS = {"rows": 4283507, "STAR": 890744, "GALAXY": 2639535, "QSO": 753228}

MIN_FREE_BYTES = 3 * 1024**3
# Declared in requirements.txt for the local app and tooling; the pipeline does not import them.
PIP_SKIP = {"pytest", "ruff", "duckdb", "astroquery"}
PIP_SHOW = {
    "numpy",
    "pandas",
    "pyarrow",
    "lightgbm",
    "scikit-learn",
    "astropy",
    "scipy",
}

T0 = time.time()


def log(message):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime(time.time() - T0))}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def fail(message):
    log(f"ERROR: {message}")
    raise SystemExit(f"astro-triage runner stopped: {message}")


def check_disk(stage):
    free = shutil.disk_usage(WORKING).free
    if free < MIN_FREE_BYTES:
        fail(f"{stage}: only {free / 1024**3:.1f} GB free in {WORKING} (< 3 GB)")


def run_stream_code(label, cmd, cwd=None):
    """Run a command, echo its output line by line into the log; return (exit code, lines)."""
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
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        log(lines[-1])
    return proc.wait(), lines


def run_stream(label, cmd, cwd=None):
    """Run a command, echo its output line by line into the log, stop on a non-zero exit."""
    code, _lines = run_stream_code(label, cmd, cwd)
    if code != 0:
        fail(f"{label} exited with code {code}")


def find_package():
    zips = sorted(glob.glob(f"{INPUT}/**/{PACKAGE_ZIP}", recursive=True))
    marker = sorted(glob.glob(f"{INPUT}/**/scripts/xmatch_full.py", recursive=True))
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
        fail(
            f"[pacote] neither {PACKAGE_ZIP} nor scripts/xmatch_full.py found under {INPUT}"
        )
    for needed in (
        "scripts/xmatch_full.py",
        "scripts/build_dataset.py",
        "requirements.txt",
    ):
        if not os.path.isfile(f"{REPO}/{needed}"):
            fail(f"[pacote] {needed} missing from the package")
    log(f"[pacote] ready in {REPO}: {sorted(os.listdir(REPO))}")


def install_dependencies():
    packages, skipped = [], []
    with open(f"{REPO}/requirements.txt", encoding="utf-8") as handle:
        for raw in handle:
            spec = raw.split("#", 1)[0].strip()
            if not spec:
                continue
            name = spec.split("==", 1)[0].strip().lower()
            (skipped if name in PIP_SKIP else packages).append(spec)
    log(
        f"[pip] installing {packages}; not installed (unused by the pipeline): {skipped}"
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
    ]
    result = None
    for attempt in (1, 2):
        result = subprocess.run(
            cmd + packages, capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            break
        log(
            f"[pip] attempt {attempt} failed (exit {result.returncode}):\n{result.stderr}"
        )
        if (
            "No matching distribution" in result.stderr
            or "Requires-Python" in result.stderr
        ):
            break
    if result is None or result.returncode != 0:
        fail(
            "[pip] a pinned version is not available for Python "
            f"{platform.python_version()}; the exact pip error is above. Update "
            "requirements.txt and rebuild the package."
        )
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    shown = [row for row in frozen if row.split("==", 1)[0].lower() in PIP_SHOW]
    log(f"[pip] python {platform.python_version()}; versions: {shown}")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started_utc": datetime.fromtimestamp(T0, UTC).isoformat(timespec="seconds"),
        "pip_freeze": frozen,
    }
    with open(ENV_PATH, "w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)


def remote_size():
    try:
        request = urllib.request.Request(FITS_URL, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers["Content-Length"])
    except (urllib.error.URLError, KeyError, ValueError, OSError):
        request = urllib.request.Request(FITS_URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def download_specobj():
    if os.path.isfile(FITS_PATH) and os.path.getsize(FITS_PATH) == FITS_BYTES:
        log("[specobj] file already present with the expected size")
        return
    if shutil.disk_usage(TEMP).free < FITS_BYTES + 2 * 1024**3:
        fail(f"[specobj] not enough free space in {TEMP} for {FITS_BYTES} bytes")
    try:
        size = remote_size()
    except (urllib.error.URLError, KeyError, ValueError, OSError) as exc:
        fail(f"[specobj] cannot read the remote size: {exc}")
    if size != FITS_BYTES:
        fail(f"[specobj] Content-Length {size} != expected {FITS_BYTES}")
    part = FITS_PATH + ".part"
    failures = 0
    for attempt in range(1, 31):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == FITS_BYTES:
            break
        if have > FITS_BYTES:
            os.remove(part)
            have = 0
        log(f"[specobj] attempt {attempt}: resuming at byte {have} of {FITS_BYTES}")
        request = urllib.request.Request(FITS_URL, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    log("[specobj] server ignored Range; restarting from byte 0")
                    have = 0
                mode = "ab" if have else "wb"
                next_mark = have + 500 * 1024**2
                with open(part, mode) as handle:
                    while True:
                        block = response.read(8 * 1024**2)
                        if not block:
                            break
                        handle.write(block)
                        have += len(block)
                        if have >= next_mark:
                            log(f"[specobj] {have / 1024**3:.2f} GB")
                            next_mark += 500 * 1024**2
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            failures += 1
            log(f"[specobj] transfer error ({failures}): {exc}")
            time.sleep(min(60, 5 * failures))
    final = os.path.getsize(part) if os.path.exists(part) else 0
    if final != FITS_BYTES:
        fail(f"[specobj] final size {final} != expected {FITS_BYTES}; rerun to resume")
    os.replace(part, FITS_PATH)
    log(f"[specobj] downloaded {FITS_BYTES} bytes to {FITS_PATH}")


COUNT_SNIPPET = (
    "import json, sys, pyarrow.compute as pc, pyarrow.parquet as pq\n"
    "t = pq.read_table(sys.argv[1], columns=['class'])\n"
    "out = {'rows': t.num_rows}\n"
    "for item in pc.value_counts(t['class']).to_pylist():\n"
    "    out[item['values']] = item['counts']\n"
    "print(json.dumps(out))\n"
)


def candidate_counts():
    done = subprocess.run(
        [sys.executable, "-c", COUNT_SNIPPET, CANDIDATES_PATH],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        return None
    return json.loads(done.stdout.strip().splitlines()[-1])


def build_candidates():
    counts = candidate_counts() if os.path.isfile(CANDIDATES_PATH) else None
    if counts != EXPECTED_COUNTS:
        run_stream(
            "[candidatos] extract_specobj",
            [
                sys.executable,
                f"{REPO}/scripts/extract_specobj.py",
                "--fits",
                FITS_PATH,
                "--out",
                CANDIDATES_PATH,
            ],
        )
        counts = candidate_counts()
    if counts != EXPECTED_COUNTS:
        fail(f"[candidatos] counts {counts} differ from the expected {EXPECTED_COUNTS}")
    log(f"[candidatos] verified {counts}")


def restore_previous_outputs():
    os.makedirs(XMATCH_DIR, exist_ok=True)
    copied = 0
    for source in sorted(glob.glob(f"{INPUT}/**/xmatch_full", recursive=True)):
        if not os.path.isdir(source):
            continue
        for name in os.listdir(source):
            target = f"{XMATCH_DIR}/{name}"
            path = f"{source}/{name}"
            if (
                os.path.isfile(path)
                and not name.startswith(".")
                and not os.path.exists(target)
            ):
                shutil.copy(path, target)
                copied += 1
    for archive_path in sorted(
        glob.glob(f"{INPUT}/**/astro_triage_outputs*.zip", recursive=True)
    ):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    parts = member.split("/")
                    if len(parts) != 2 or parts[0] != "xmatch_full" or not parts[1]:
                        continue
                    target = f"{XMATCH_DIR}/{parts[1]}"
                    if not os.path.exists(target):
                        with archive.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        copied += 1
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"[retomada] cannot read {archive_path}: {exc}")
    log(f"[retomada] {copied} XMatch files restored from earlier versions")


def package_outputs():
    log("[zip] writing the output zip")
    members = []
    for base, arc in ((DATASET_DIR, DATASET_NAME), (XMATCH_DIR, "xmatch_full")):
        for root, _dirs, files in os.walk(base):
            for name in files:
                if name.endswith(".tmp"):
                    continue
                path = os.path.join(root, name)
                members.append(
                    (path, f"{arc}/{os.path.relpath(path, base)}".replace(os.sep, "/"))
                )
    for extra, arc in (
        (f"{REPO}/manifests", "manifests"),
        (f"{REPO}/reports", "reports"),
    ):
        for root, _dirs, files in os.walk(extra):
            for name in files:
                path = os.path.join(root, name)
                members.append(
                    (path, f"{arc}/{os.path.relpath(path, extra)}".replace(os.sep, "/"))
                )
    members.append((ENV_PATH, "environment.json"))
    members.append((LOG_PATH, "run_log.txt"))
    check_disk("[zip]")
    tmp = OUT_ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
        for path, arc in members:
            kind = (
                zipfile.ZIP_STORED
                if path.endswith(".parquet")
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, arc, compress_type=kind)
    with zipfile.ZipFile(tmp) as archive:
        names = set(archive.namelist())
    for required in (
        f"{DATASET_NAME}/labels.parquet",
        f"{DATASET_NAME}/features.parquet",
    ):
        if required not in names:
            os.remove(tmp)
            fail(f"[zip] {required} missing from the output zip")
    os.replace(tmp, OUT_ZIP)
    for name in os.listdir(WORKING):
        if name in (os.path.basename(OUT_ZIP), os.path.basename(LOG_PATH)):
            continue
        path = f"{WORKING}/{name}"
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    log(f"[zip] {OUT_ZIP}: {os.path.getsize(OUT_ZIP)} bytes, {len(names)} files")


PHASE1_STAGES = [
    "pacote",
    "pip",
    "retomada",
    "specobj",
    "candidatos",
    "xmatch",
    "dataset",
    "treino",
    "avaliação",
    "zip",
]
SEEDS_STATE = {}


def read_mode():
    path = f"{REPO}/{MODE_FILE}"
    if not os.path.isfile(path):
        return {"mode": "phase1"}
    try:
        with open(path, encoding="utf-8") as handle:
            mode = json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"[modo] cannot read {MODE_FILE}: {exc}")
    if not isinstance(mode, dict) or mode.get("mode") not in ("phase1", "seeds"):
        fail(f"[modo] {MODE_FILE}: 'mode' must be 'phase1' or 'seeds'")
    expected = {"mode"} if mode["mode"] == "phase1" else {"mode", "after_measure_one"}
    if set(mode) != expected:
        fail(f"[modo] {MODE_FILE}: keys {sorted(mode)} differ from {sorted(expected)}")
    if mode["mode"] == "seeds" and mode["after_measure_one"] not in (
        "stop",
        "continue",
    ):
        fail(f"[modo] {MODE_FILE}: 'after_measure_one' must be 'stop' or 'continue'")
    return mode


def plan_stages(mode):
    if mode.get("mode", "phase1") == "phase1":
        return list(PHASE1_STAGES)
    tail = ["paridade", "retomada_seeds", "seeds_measure_one"]
    if mode["after_measure_one"] == "continue":
        tail.append("seeds_rest")
    return PHASE1_STAGES[:7] + tail + ["zip_seeds"]


def remaining_fits(report):
    done = report.get("fits", {})
    return {
        "full": [k for k in range(1, N_SEEDS) if f"lgbm_full_s{k}" not in done],
        "noind": [k for k in range(N_SEEDS) if f"lgbm_noind_s{k}" not in done],
    }


def timed(name, func):
    log(f"[{name}] start")
    started = time.time()
    try:
        return func()
    finally:
        seconds = time.time() - started
        SEEDS_STATE.setdefault("stage_seconds", {})[name] = round(seconds, 1)
        log(f"[{name}] done in {seconds:.1f}s")


def read_seed_report():
    path = f"{DATASET_DIR}/{SEEDS_DIRNAME}/{SEEDS_REPORT_NAME}"
    if not os.path.isfile(path):
        return {"fits": {}}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"[seeds] cannot read {path}: {exc}")


def check_parity():
    code, lines = run_stream_code(
        "[paridade] full_v0_parity",
        [
            sys.executable,
            f"{REPO}/scripts/full_v0_parity.py",
            "check",
            "--reference",
            f"{REPO}/{REFERENCE_FILE}",
            "--outdir",
            DATASET_DIR,
            "--report",
            PARITY_REPORT,
        ],
    )
    if code == 0:
        log("[paridade] match")
        return True
    for row in lines:
        if row.startswith("MISMATCH: "):
            log(f"[paridade] {row}")
    if code != 3:
        log(f"[paridade] ERROR: parity script exited with code {code}")
    if not os.path.isfile(PARITY_REPORT):
        with open(PARITY_REPORT, "w", encoding="utf-8") as handle:
            json.dump(
                {"match": False, "error": f"parity script exit code {code}"}, handle
            )
    return False


SEED_ARTEFACT_RE = re.compile(
    r"^(?:models/(?:model_)?(lgbm_(?:full|noind)_s\d+)(?:\.txt|\.manifest\.json)"
    r"|predictions_(lgbm_(?:full|noind)_s\d+)_.+)$"
)


def merge_source_report(fits, raw, origin):
    try:
        found = json.loads(raw)["fits"]
        if not isinstance(found, dict):
            raise TypeError("fits is not an object")
    except (ValueError, KeyError, TypeError) as exc:
        log(f"[retomada_seeds] ignoring unreadable report in {origin}: {exc}")
        return
    for model_id, entry in found.items():
        fits.setdefault(model_id, entry)


def write_merged_report(target_root, fits):
    os.makedirs(target_root, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_root, prefix=".report.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"fits": fits}, handle, indent=2)
        os.replace(tmp, f"{target_root}/{SEEDS_REPORT_NAME}")
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def restore_seed_outputs():
    target_root = f"{DATASET_DIR}/{SEEDS_DIRNAME}"
    copied = 0
    fits = dict(read_seed_report()["fits"])
    for source in sorted(glob.glob(f"{INPUT}/**/{SEEDS_DIRNAME}", recursive=True)):
        if not os.path.isdir(source):
            continue
        for root, _dirs, files in os.walk(source):
            for name in files:
                if name.startswith(".") or name.endswith(".tmp"):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, source)
                if rel == SEEDS_REPORT_NAME:
                    with open(path, encoding="utf-8", errors="replace") as handle:
                        merge_source_report(fits, handle.read(), path)
                    continue
                target = os.path.join(target_root, rel)
                if not os.path.exists(target):
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy(path, target)
                    copied += 1
    prefix = f"{SEEDS_DIRNAME}/"
    for archive_path in sorted(
        glob.glob(f"{INPUT}/**/astro_triage_outputs*.zip", recursive=True)
    ):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    parts = member.split("/")
                    if (
                        not member.startswith(prefix)
                        or "" in parts
                        or ".." in parts
                        or parts[-1].startswith(".")
                    ):
                        continue
                    if parts[1:] == [SEEDS_REPORT_NAME]:
                        merge_source_report(
                            fits,
                            archive.read(member).decode("utf-8", "replace"),
                            f"{archive_path}:{member}",
                        )
                        continue
                    target = os.path.join(target_root, *parts[1:])
                    if not os.path.exists(target):
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with archive.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        copied += 1
        except (zipfile.BadZipFile, OSError) as exc:
            log(f"[retomada_seeds] cannot read {archive_path}: {exc}")
    if fits or os.path.isdir(target_root):
        write_merged_report(target_root, fits)
    orphans = set()
    for root, _dirs, files in os.walk(target_root):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), target_root)
            match = SEED_ARTEFACT_RE.match(rel.replace(os.sep, "/"))
            if match:
                model_id = match.group(1) or match.group(2)
                if model_id not in fits:
                    orphans.add(model_id)
    if orphans:
        fail(
            "[retomada_seeds] artefacts without a report entry in any earlier "
            f"source (refit would be refused): {sorted(orphans)}"
        )
    SEEDS_STATE["fits_skipped_as_restored"] = sorted(fits)
    log(f"[retomada_seeds] {copied} seed files restored from earlier versions")


def seeds_script_cmd(*args):
    return [
        sys.executable,
        f"{REPO}/scripts/train_lgbm_seeds.py",
        "--outdir",
        DATASET_DIR,
        *args,
    ]


def seeds_measure_one():
    fits = read_seed_report()["fits"]
    if MEASURE_ONE_ID in fits:
        log(
            f"[seeds_measure_one] {MEASURE_ONE_ID} already done, skipped; stored "
            f"wall_seconds={fits[MEASURE_ONE_ID]['wall_seconds']}"
        )
    else:
        check_disk("[seeds_measure_one]")
        run_stream(
            "[seeds_measure_one] train_lgbm_seeds", seeds_script_cmd("--measure-one")
        )
    wall = read_seed_report()["fits"][MEASURE_ONE_ID]["wall_seconds"]
    SEEDS_STATE["measure_one_wall_seconds"] = wall
    log(f"[seeds_measure_one] {MEASURE_ONE_ID} wall_seconds={wall}")


def seeds_rest():
    for name, ks in remaining_fits(read_seed_report()).items():
        if not ks:
            log(f"[seeds_rest] --features {name}: nothing left to fit")
            continue
        check_disk("[seeds_rest]")
        run_stream(
            f"[seeds_rest] train_lgbm_seeds {name}",
            seeds_script_cmd("--features", name, "--seeds", ",".join(map(str, ks))),
        )


def package_seeds():
    if not os.path.isfile(PARITY_REPORT):
        fail("[zip_seeds] parity_report.json missing; refusing to package")
    SEEDS_STATE["fits_done"] = sorted(read_seed_report()["fits"])
    with open(SEEDS_RUN_PATH, "w", encoding="utf-8") as handle:
        json.dump(SEEDS_STATE, handle, indent=2)
    members = []
    base = f"{DATASET_DIR}/{SEEDS_DIRNAME}"
    for root, _dirs, files in os.walk(base):
        for name in files:
            if name.endswith(".tmp") or name.startswith("."):
                continue
            path = os.path.join(root, name)
            arc = f"{SEEDS_DIRNAME}/{os.path.relpath(path, base)}".replace(os.sep, "/")
            members.append((path, arc))
    members += [
        (PARITY_REPORT, "parity_report.json"),
        (SEEDS_RUN_PATH, "seeds_run.json"),
        (ENV_PATH, "environment.json"),
        (LOG_PATH, "run_log.txt"),
    ]
    top_level = {
        "parity_report.json",
        "seeds_run.json",
        "environment.json",
        "run_log.txt",
    }
    for _path, arc in members:
        banned = arc.rsplit("/", 1)[-1] in ("labels.parquet", "features.parquet")
        if banned or not (arc in top_level or arc.startswith(f"{SEEDS_DIRNAME}/")):
            fail(f"[zip_seeds] refusing to package {arc}")
    check_disk("[zip_seeds]")
    tmp = OUT_ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
        for path, arc in members:
            kind = (
                zipfile.ZIP_STORED
                if path.endswith(".parquet")
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, arc, compress_type=kind)
    os.replace(tmp, OUT_ZIP)
    keep = (os.path.basename(OUT_ZIP), os.path.basename(LOG_PATH), "xmatch_full")
    for name in os.listdir(WORKING):
        if name in keep:
            continue
        path = f"{WORKING}/{name}"
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    log(
        f"[zip_seeds] {OUT_ZIP}: {os.path.getsize(OUT_ZIP)} bytes, {len(members)} files"
    )


def run_seeds_tail(mode):
    if not timed("paridade", check_parity):
        timed("zip_seeds", package_seeds)
        fail("[paridade] rebuild differs from the reference; no training was run")
    try:
        timed("retomada_seeds", restore_seed_outputs)
        timed("seeds_measure_one", seeds_measure_one)
        if mode["after_measure_one"] == "continue":
            timed("seeds_rest", seeds_rest)
        else:
            log("[seeds] after_measure_one=stop: the remaining nine fits were not run")
    finally:
        timed("zip_seeds", package_seeds)


def main():
    if not os.path.isdir(INPUT) or not os.path.isdir(WORKING):
        raise SystemExit("this runner only works inside a Kaggle notebook")
    os.makedirs(TEMP, exist_ok=True)
    os.makedirs(XMATCH_WORK, exist_ok=True)
    check_disk("start")
    timed("pacote", find_package)
    mode = read_mode()
    SEEDS_STATE.update(
        {"mode": mode["mode"], "after_measure_one": mode.get("after_measure_one")}
    )
    timed("pip", install_dependencies)
    timed("retomada", restore_previous_outputs)
    timed("specobj", download_specobj)
    timed("candidatos", build_candidates)
    timed("xmatch", build_xmatch)
    timed("dataset", build_dataset)
    if mode["mode"] == "seeds":
        run_seeds_tail(mode)
    else:
        run_phase1_tail()


def build_xmatch():
    check_disk("[xmatch]")
    run_stream(
        "[xmatch] xmatch_full",
        [
            sys.executable,
            f"{REPO}/scripts/xmatch_full.py",
            "--candidates",
            CANDIDATES_PATH,
            "--outdir",
            XMATCH_DIR,
            "--workdir",
            XMATCH_WORK,
            "--chunk-size",
            "1000000",
            "--radius",
            "2.0",
            "--sleep",
            "20",
        ],
    )


def build_dataset():
    check_disk("[dataset]")
    run_stream(
        "[dataset] build_dataset",
        [
            sys.executable,
            f"{REPO}/scripts/build_dataset.py",
            "--dataset-name",
            DATASET_NAME,
            "--xmatch-dir",
            XMATCH_DIR,
            "--candidates",
            CANDIDATES_PATH,
        ],
    )


def run_phase1_tail():
    run_stream(
        "[treino] train_lgbm",
        [sys.executable, f"{REPO}/scripts/train_lgbm.py", "--outdir", DATASET_DIR],
    )
    if os.path.isfile(f"{REPO}/scripts/evaluate.py"):
        run_stream(
            "[avaliação] evaluate",
            [sys.executable, f"{REPO}/scripts/evaluate.py", "--outdir", DATASET_DIR],
        )
    else:
        log("[avaliação] evaluation step skipped: scripts/evaluate.py not in package")
    package_outputs()


if __name__ == "__main__":
    main()
