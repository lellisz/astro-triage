"""Training loop of the deep-learning axis, part 1: seeding, early stopping, checkpoint, resume.

ADR 0006 section 3.7 and 7, spec T-DL-05 slice S5. `train_one` (part 2) builds on `fit_network`.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from astro_triage.dataset.build import CLASSES
from astro_triage.models import lgbm
from astro_triage.models.dl_data import Preprocessor, save_preprocessor, subsample_index
from astro_triage.models.dl_nets import (
    FTT_CONFIG,
    MLP_CONFIG,
    build_model,
    parameter_groups,
)
from astro_triage.models.lgbm import (
    CALIBRATION_NAME,
    MODEL_FULL,
    MODEL_NOIND,
    MODEL_VERSION,
    SEED,
    TEMPERATURE_BOUNDS,
    Fold,
    TrainError,
    _atomic,
    _git_sha,
    _manifest,
    evaluate_split,
    fit_temperature,
    predictions_table,
)

try:
    import resource
except ImportError:  # Windows has no resource module
    resource = None  # type: ignore[assignment]

INFERENCE_BATCH = 8192
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DLSpec:
    arch: Literal["mlp", "ftt"]
    feature_set: Literal["full", "noind"]
    seed_index: int

    @property
    def model_id(self) -> str:
        return f"{self.arch}_{self.feature_set}_s{self.seed_index}"

    @property
    def seed(self) -> int:
        return SEED + self.seed_index


class FitTensors(NamedTuple):
    """CPU tensors: x_cont float32 (n, d), x_cat int64 (n,) or None, y int64 (n,) in CLASSES order."""

    train_cont: Tensor
    train_cat: Tensor | None
    train_y: Tensor
    val_cont: Tensor
    val_cat: Tensor | None
    val_y: Tensor


def set_seeds(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.use_deterministic_algorithms(True, warn_only=True)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def compat_key(
    spec: DLSpec,
    config: Mapping[str, object],
    patience: int,
    preproc_sha: str,
    train_ids_sha: str,
    device_type: str,
    torch_version: str,
) -> dict[str, object]:
    """Everything that must match to resume; max_epochs is a cost bound and is excluded."""
    return {
        "arch": spec.arch,
        "feature_set": spec.feature_set,
        "seed": spec.seed,
        "config": dict(config),
        "patience": patience,
        "preproc_sha": preproc_sha,
        "train_ids_sha": train_ids_sha,
        "device_type": device_type,
        "torch_version": torch_version,
    }


def _on_epoch_end(epoch: int) -> None:
    """No-op hook called after last.pt is written; tests raise here to simulate a kill."""


def _batch_size(compat: Mapping[str, object]) -> int:
    config = compat["config"]
    size = config.get("batch_size") if isinstance(config, dict) else None
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise TrainError("compat['config']['batch_size'] must be a positive int")
    return size


def _val_logloss(model: nn.Module, tensors: FitTensors, device: str) -> float:
    model.eval()
    total = 0.0
    n = tensors.val_y.shape[0]
    with torch.no_grad():
        for start in range(0, n, INFERENCE_BATCH):
            sl = slice(start, start + INFERENCE_BATCH)
            cat = None if tensors.val_cat is None else tensors.val_cat[sl].to(device)
            logits = model(tensors.val_cont[sl].to(device), cat)
            total += float(F.cross_entropy(logits, tensors.val_y[sl].to(device), reduction="sum"))
    return total / n


def _check_compat(stored: object, now: dict[str, object], last: Path) -> None:
    if not isinstance(stored, dict) or stored != now:
        old = stored if isinstance(stored, dict) else {}
        diff = sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))
        raise TrainError(
            f"checkpoint {last} was written with a different configuration (keys: {diff}); "
            "delete it or rerun with the original settings"
        )


def fit_network(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    tensors: FitTensors,
    *,
    ckpt: Path,
    compat: dict[str, object],
    max_epochs: int,
    patience: int,
    device: str,
    gen: torch.Generator,
) -> dict[str, object]:
    """Train with early stopping on val logloss; `ckpt` is the last.pt path, best.pt sits beside it.

    The model must already be on `device`. Returns the best epoch and costs, best weights loaded.
    """
    last, best = ckpt, ckpt.with_name("best.pt")
    last.parent.mkdir(parents=True, exist_ok=True)
    if max_epochs < 1:
        raise TrainError(f"max_epochs must be >= 1, got {max_epochs}")
    if patience < 1:
        raise TrainError(f"patience must be >= 1, got {patience}")
    if compat.get("patience") != patience:
        raise TrainError(f"compat['patience']={compat.get('patience')!r} differs from patience={patience}")
    if compat.get("device_type") != torch.device(device).type:
        raise TrainError(
            f"compat['device_type']={compat.get('device_type')!r} differs from device={device!r}"
        )
    batch = _batch_size(compat)
    # Plain JSON types only (str subclasses such as TorchVersion are rejected by weights_only loads).
    compat = json.loads(json.dumps(compat))
    n = tensors.train_y.shape[0]
    epoch, best_loss, best_epoch, counter = 0, float("inf"), 0, 0
    seconds: list[float] = []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        _check_compat(state["compat"], compat, last)
        cuda_rng = state["rng"]["torch_cuda"]
        if cuda_rng is not None and (
            not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count()
        ):
            raise TrainError(
                f"checkpoint {last} holds {len(cuda_rng)} CUDA RNG state(s) but this machine has "
                f"{torch.cuda.device_count() if torch.cuda.is_available() else 0} usable CUDA device(s); "
                "resume on the original hardware or delete the checkpoint"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        epoch, best_loss = int(state["epoch"]), float(state["best_val_loss"])
        best_epoch, counter = int(state["best_epoch"]), int(state["counter"])
        seconds = [float(s) for s in state["epoch_seconds"]]
        torch.set_rng_state(state["rng"]["torch_cpu"])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        gen.set_state(state["rng"]["shuffle"])
    else:
        best.unlink(missing_ok=True)  # stale best.pt from an earlier, unrelated run
    while epoch < max_epochs and counter < patience:
        started = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            cat = None if tensors.train_cat is None else tensors.train_cat[idx].to(device)
            logits = model(tensors.train_cont[idx].to(device), cat)
            loss = F.cross_entropy(logits, tensors.train_y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_loss = _val_logloss(model, tensors, device)
        if not np.isfinite(val_loss):
            raise TrainError(f"non-finite validation loss at epoch {epoch + 1}; lower the learning rate")
        epoch += 1
        if val_loss < best_loss:
            best_loss, best_epoch, counter = val_loss, epoch, 0
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pack = {"model": weights, "best_epoch": best_epoch, "best_val_loss": best_loss}
            _atomic(best, lambda tmp, p=pack: torch.save(p, tmp))
        else:
            counter += 1
        seconds.append(time.perf_counter() - started)
        snapshot = {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
            "best_val_loss": best_loss, "best_epoch": best_epoch, "counter": counter,
            "rng": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "shuffle": gen.get_state(),
            },
            "epoch_seconds": seconds, "compat": compat,
        }
        _atomic(last, lambda tmp, s=snapshot: torch.save(s, tmp))
        _on_epoch_end(epoch)
    if not best.exists():
        raise TrainError(f"no epoch completed (max_epochs={max_epochs}); {best} was not written")
    pack = torch.load(best, map_location="cpu", weights_only=True)
    if pack["best_epoch"] != best_epoch or pack["best_val_loss"] != best_loss:
        raise TrainError(
            f"{best} (epoch {pack['best_epoch']}, loss {pack['best_val_loss']}) does not match "
            f"{last} (epoch {best_epoch}, loss {best_loss}); delete both and retrain"
        )
    model.load_state_dict(pack["model"])
    return {
        "best_epoch": best_epoch, "epochs_run": epoch,
        "best_val_logloss": best_loss, "epoch_seconds": seconds,
    }


def _predict(model: nn.Module, cont: Tensor, cat: Tensor | None, device: str) -> Floats:
    """Logits as float64 in CLASSES order, batched, eval mode, no gradients."""
    model.eval()
    parts: list[Floats] = []
    with torch.no_grad():
        for start in range(0, cont.shape[0], INFERENCE_BATCH):
            sl = slice(start, start + INFERENCE_BATCH)
            batch_cat = None if cat is None else cat[sl].to(device)
            parts.append(model(cont[sl].to(device), batch_cat).double().cpu().numpy())
    return np.concatenate(parts)


def _peak_rss_bytes() -> int | str:
    if resource is None or sys.platform != "linux":
        return "not measured"
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # KiB on Linux


def _opt(model: nn.Module, config: Mapping[str, object]) -> torch.optim.Optimizer:
    lr, wd = config["lr"], config["weight_decay"]
    if not isinstance(lr, float) or not isinstance(wd, float):
        raise TrainError("model config needs float lr and weight_decay")
    return torch.optim.AdamW(parameter_groups(model, wd), lr=lr)


def _opt_tensor(a: npt.NDArray[np.generic] | None) -> Tensor | None:
    return None if a is None else torch.from_numpy(a)


def train_one(
    spec: DLSpec,
    folds: dict[str, Fold],
    outdir: Path,
    ckpt_dir: Path,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    train_subset: int | None = None,
) -> dict[str, object]:
    """Fit one network on train, calibrate on val, write val+holdout predictions and the model.

    The third fold is read for features and source_id only (never labels), after every weight
    and the temperature are frozen.
    """
    model_id = spec.model_id
    pred_path = outdir / f"predictions_{model_id}_{MODEL_VERSION}.parquet"
    model_path = outdir / "models" / f"{model_id}.pt"
    preproc_path = outdir / "models" / f"{model_id}.preproc.pkl"
    clash = [p.name for p in (pred_path, model_path, preproc_path) if p.exists()]
    if clash:
        raise TrainError(f"refusing to overwrite {clash} in {outdir}; delete them to retrain")
    train, val, holdout = folds["train"], folds["val"], folds["test"]
    features = (MODEL_FULL if spec.feature_set == "full" else MODEL_NOIND).features
    train_frame, train_y, train_ids = train.frame, train.labels, train.source_id
    if train_subset is not None:
        keep = subsample_index(len(train_ids), train_subset)
        train_frame, train_y, train_ids = train_frame.iloc[keep], train_y[keep], train_ids[keep]
    counts = np.bincount(train_y, minlength=len(CLASSES)).astype(np.float64)
    if (counts == 0).any():
        raise TrainError(f"train rows used for {model_id} lack a class (counts {counts.tolist()})")
    pi_src = counts / counts.sum()

    prep = Preprocessor(features).fit(train_frame)
    x_tr, c_tr = prep.transform(train_frame)
    x_va, c_va = prep.transform(val.frame)
    tensors = FitTensors(
        torch.from_numpy(x_tr), _opt_tensor(c_tr), torch.from_numpy(train_y),
        torch.from_numpy(x_va), _opt_tensor(c_va), torch.from_numpy(val.labels),
    )
    config = MLP_CONFIG if spec.arch == "mlp" else FTT_CONFIG
    gen = set_seeds(spec.seed)
    model = build_model(spec.arch, prep.n_continuous, prep.n_categories).to(device)
    opt = _opt(model, config)
    preproc_sha = prep.state_sha256()
    compat = compat_key(
        spec, config, patience, preproc_sha,
        hashlib.sha256(np.ascontiguousarray(train_ids, dtype="<i8").tobytes()).hexdigest(),
        torch.device(device).type, str(torch.__version__),
    )
    on_cuda = torch.device(device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    fit = fit_network(
        model, opt, tensors, ckpt=ckpt_dir / model_id / "last.pt", compat=compat,
        max_epochs=max_epochs, patience=patience, device=device, gen=gen,
    )
    seconds, best_loss = fit["epoch_seconds"], fit["best_val_logloss"]
    if not isinstance(seconds, list) or not isinstance(best_loss, float):
        raise TrainError("fit_network returned an unexpected report")

    started = time.perf_counter()
    z_val = _predict(model, tensors.val_cont, tensors.val_cat, device)
    val_seconds = time.perf_counter() - started
    temperature = fit_temperature(z_val, val.labels)  # weights and T are frozen from here on
    started = time.perf_counter()
    x_ho, c_ho = prep.transform(holdout.frame)
    z_ho = _predict(model, torch.from_numpy(x_ho), _opt_tensor(c_ho), device)
    predict_seconds = val_seconds + time.perf_counter() - started

    ids = np.concatenate([val.source_id, holdout.source_id])
    table, _ = predictions_table(ids, np.concatenate([z_val, z_ho]), temperature)
    fractions = {c: float(pi_src[i]) for i, c in enumerate(CLASSES)}
    inputs = ["labels.parquet", "features.parquet"]
    meta = {
        "schema_name": "predictions", "schema_version": "1",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(), "inputs": json.dumps(inputs), "model_id": model_id,
        "model_version": MODEL_VERSION, "features_file": "features.parquet",
        "labels_file": "labels.parquet",
        "split_file": "labels.parquet (column fold, astro_triage.split)",
        "train_class_fractions": json.dumps(fractions),
        "calibration": CALIBRATION_NAME, "calibration_fit_fold": "val",
        "family": "deep_learning", "arch": spec.arch, "feature_set": spec.feature_set,
        "arch_config": json.dumps(config, sort_keys=True), "seed": str(spec.seed),
        "best_epoch": str(fit["best_epoch"]), "epochs_run": str(fit["epochs_run"]),
        "temperature": repr(temperature), "preprocessor_sha256": preproc_sha,
        "torch_version": str(torch.__version__), "device": device,
        "train_subset": "none" if train_subset is None else str(train_subset),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _atomic(model_path, lambda tmp: torch.save(weights, tmp))
    save_preprocessor(prep, preproc_path)
    out_table = table.sort_by("source_id").replace_schema_metadata(meta)
    _atomic(pred_path, lambda tmp: pq.write_table(out_table, tmp, compression="zstd"))
    params = (
        f"torch {torch.__version__}; arch={spec.arch}; lr={config['lr']}; "  # d_token in config trips the manifest secret filter
        f"weight_decay={config['weight_decay']}; "
        f"seed={spec.seed}; best_epoch={fit['best_epoch']}; patience={patience}; "
        f"calibration={CALIBRATION_NAME} fit on val"
    )
    _manifest(pred_path, pred_path.stem, len(out_table), params, inputs)
    _manifest(model_path, f"model_{model_id}", 1, params, inputs)

    t_lo, t_hi = TEMPERATURE_BOUNDS
    gpu: dict[str, object] = (
        {"peak_gpu_bytes": torch.cuda.max_memory_allocated(),
         "device_name": torch.cuda.get_device_name(torch.device(device))}
        if on_cuda else {"peak_gpu_bytes": "not measured", "device_name": "cpu"}
    )
    costs = {
        "epoch_seconds_mean": float(np.mean(seconds)), "epoch_seconds_max": float(np.max(seconds)),
        "epoch_seconds_total": float(np.sum(seconds)), "epochs_run": fit["epochs_run"],
        "predict_seconds": predict_seconds,
        "predict_seconds_per_1m_rows": predict_seconds / len(ids) * 1e6,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": model_path.stat().st_size, "peak_rss_bytes": _peak_rss_bytes(),
        **gpu,
    }
    return {
        "model_id": model_id, "seed": spec.seed,
        "val_calibration_fitted_here": evaluate_split(val, z_val, temperature, pi_src, False),
        "temperature": temperature,
        "temperature_at_bound": bool(temperature <= t_lo * 1.001 or temperature >= t_hi * 0.999),
        "train_class_fractions": fractions,
        "beats_prior_logloss": best_loss < float(-np.log(pi_src)[val.labels].mean()),
        "best_epoch": fit["best_epoch"], "costs": costs,
        "prediction_file": pred_path.relative_to(lgbm.REPO_ROOT).as_posix(),
        "model_file": model_path.relative_to(lgbm.REPO_ROOT).as_posix(),
    }
