"""GPU session check, platform environment pin (ADR 0006 H-D7) and test-key-safe JSON I/O for the DL runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.dataset.build import CLASSES
from astro_triage.evaluation.dl_val import (
    FRACTION_TOL,
    build_freeze,
    sha256_file,
    val_purity_summary,
)
from astro_triage.evaluation.runner import EvalError
from astro_triage.models.ensemble import build_ensemble
from astro_triage.models.lgbm import MODEL_VERSION, TrainError, _atomic

ENV_NAME = "environment.json"


def has_test_key(node: object) -> bool:
    if isinstance(node, dict):
        return any("test" in str(k).lower() or has_test_key(v) for k, v in node.items())
    if isinstance(node, list):
        return any(has_test_key(v) for v in node)
    return False


def write_json(path: Path, doc: dict[str, object]) -> None:
    if has_test_key(doc):
        raise TrainError(
            f"{path.name} would contain a test-derived key; refusing to write"
        )
    _atomic(
        path,
        lambda tmp: Path(tmp).write_text(json.dumps(doc, indent=2), encoding="utf-8"),
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_gpu_session(
    version: str,
    cuda: str | None,
    name: str,
    capability: tuple[int, int],
    arch_list: list[str],
    pinned: str | None,
) -> None:
    """Refuse a session whose GPU/torch build cannot run the pinned, deterministic setup."""
    if cuda is None:
        raise TrainError(
            f"torch {version} is a CPU build; --device cuda needs a CUDA build"
        )
    if "p100" in name.lower() or capability == (6, 0):
        raise TrainError(
            f"GPU {name!r} (sm_60) is not supported by this torch build; request a T4 or newer"
        )
    arch = f"sm_{capability[0]}{capability[1]}"
    if arch not in arch_list:
        raise TrainError(
            f"torch {version} lacks {arch} for GPU {name!r} (arch list {arch_list})"
        )
    if pinned is not None and pinned != version:
        raise TrainError(
            f"torch {version} differs from the version {pinned} recorded for this platform; "
            "restore the recorded environment or start a new platform record deliberately"
        )


def gpu_facts() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise TrainError("--device cuda but torch.cuda.is_available() is False")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": tuple(torch.cuda.get_device_capability(0)),
        "arch_list": list(torch.cuda.get_arch_list()),
    }


def verify_gpu(env_path: Path, platform: str) -> dict[str, object]:
    """Probe the GPU, print it, and check it against the version pinned for this platform."""
    facts = gpu_facts()
    print(
        f"torch {facts['torch']}, CUDA {facts['cuda']}, GPU {facts['gpu']}",
        file=sys.stderr,
    )
    recorded = read_json(env_path).get(platform)
    pinned = recorded.get("torch") if isinstance(recorded, dict) else None
    capability = facts["capability"]
    arch_list = facts["arch_list"]
    assert isinstance(capability, tuple) and isinstance(arch_list, list)
    check_gpu_session(
        str(facts["torch"]),
        None if facts["cuda"] is None else str(facts["cuda"]),
        str(facts["gpu"]),
        (int(capability[0]), int(capability[1])),
        [str(a) for a in arch_list],
        None if pinned is None else str(pinned),
    )
    return facts


def record_environment(
    env_path: Path, platform: str, facts: dict[str, object] | None
) -> dict[str, object]:
    """Return the platform's environment entry, writing it on first sight."""
    env = read_json(env_path)
    if platform not in env:
        import torch

        env[platform] = {
            "torch": facts["torch"] if facts else str(torch.__version__),
            "cuda": facts["cuda"] if facts else None,
            "gpu": facts["gpu"] if facts else "cpu",
            "first_seen": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_json(env_path, env)
    entry = env[platform]
    assert isinstance(entry, dict)
    return entry


def unlink_all(paths: list[Path]) -> list[Path]:
    """Delete files; on Windows retry a locked file (PermissionError) up to 5 times, 50 to 400 ms. Returns the survivors."""
    stuck: list[Path] = []
    for path in paths:
        for delay in (0.05, 0.1, 0.2, 0.4, None):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                if sys.platform != "win32" or delay is None:
                    stuck.append(path)
                    break
                time.sleep(delay)
    return stuck


def fits_on_disk(report_path: Path) -> dict[str, object]:
    fits = read_json(report_path).get("fits", {})
    assert isinstance(fits, dict)
    return fits


def save_fit(
    report_path: Path,
    platform: str,
    environment: dict[str, object],
    model_id: str,
    entry: dict[str, object],
) -> None:
    """Re-read the report right before writing so concurrent runs in one dl/ keep each other's entries."""
    fits = fits_on_disk(report_path)
    fits[model_id] = {
        **entry,
        "platform": platform,
        "torch_version": environment["torch"],
    }
    envs = read_json(report_path).get("environment", {})
    assert isinstance(envs, dict)
    write_json(
        report_path,
        {"environment": {**envs, platform: environment}, "fits": fits},
    )


def write_once(path: Path, doc: dict[str, object]) -> None:
    """Write a JSON document that must never be replaced: temp file + os.link fails if the target exists."""
    text = json.dumps(doc, indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise TrainError(f"refusing to overwrite {path}") from exc
    finally:
        Path(tmp).unlink(missing_ok=True)


def check_pi_src(
    fractions: object, lgbm_pred: Path, model_id: str, discard: list[Path]
) -> None:
    """The DL train fractions must equal the LightGBM seed's (same train rows); spec S6 invariant."""
    try:
        meta = pq.read_schema(lgbm_pred).metadata or {}
        ref = json.loads(meta[b"train_class_fractions"])
        assert isinstance(fractions, dict)
        gap = max(abs(float(fractions[c]) - float(ref[c])) for c in CLASSES)
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        AssertionError,
        pa.ArrowException,
    ) as exc:
        unlink_all(discard)
        raise TrainError(
            f"cannot compare train_class_fractions of {model_id} with {lgbm_pred}: {exc}"
        ) from exc
    if gap > FRACTION_TOL:
        unlink_all(discard)
        raise TrainError(
            f"{model_id} train_class_fractions differ from {lgbm_pred.name} by {gap:.3g}"
        )


def purity_or_discard(
    outdir: Path, pred: Path, model_id: str, discard: list[Path]
) -> dict[str, object]:
    """Val purity of a written prediction file; on failure its outputs are deleted."""
    try:
        return val_purity_summary(outdir, pred)
    except EvalError as exc:
        stuck = unlink_all(discard)
        left = f"; could not delete: {', '.join(map(str, stuck))}" if stuck else ""
        raise TrainError(
            f"{model_id} val purity failed, outputs discarded: {exc}{left}"
        ) from exc


def run_freeze(outdir: Path, dl_dir: Path, lgbm_dir: Path, approval: str) -> Path:
    """Write <dl_dir>/val_freeze.json once, only with the owner's non-empty approval text."""
    if not approval.strip():
        raise TrainError("--freeze needs the owner's approval text in --approval-text")
    target = dl_dir / "val_freeze.json"
    if target.exists():
        raise TrainError(f"refusing to overwrite {target}")
    try:
        doc = build_freeze(outdir, dl_dir, lgbm_dir, approval)
    except EvalError as exc:
        raise TrainError(f"freeze refused: {exc}") from exc
    write_once(target, doc)
    return target


def run_ensemble(
    outdir: Path,
    dl_dir: Path,
    lgbm_dir: Path,
    nets: list[Literal["mlp", "ftt"]],
    ks: list[int],
    report_path: Path,
    platform: str,
    environment: dict[str, object],
) -> list[str]:
    """Build ens_<net>_full_s<k> and record each in the report with its val purity (val rows only)."""
    done: list[str] = []
    for net in nets:
        for k in ks:
            mid = f"ens_{net}_full_s{k}"
            pred = dl_dir / f"predictions_{mid}_{MODEL_VERSION}.parquet"
            prior = fits_on_disk(report_path).get(mid)
            if (
                isinstance(prior, dict)
                and pred.exists()
                and prior.get("prediction_sha256") == sha256_file(pred)
            ):
                print(f"{mid}: already built, skipped", file=sys.stderr)
                continue
            entry = build_ensemble(outdir, dl_dir, lgbm_dir, net, k)
            purity = purity_or_discard(
                outdir, pred, mid, [pred, pred.with_name(f"{pred.stem}.manifest.json")]
            )
            save_fit(
                report_path,
                platform,
                environment,
                mid,
                {
                    **entry,
                    "prediction_sha256": sha256_file(pred),
                    "val_purity": purity,
                },
            )
            done.append(mid)
            print(f"{mid}: w_lgbm {entry['weight']}", file=sys.stderr)
    return done
