"""Consolidate extracted Kaggle DL batches (one seed per batch) into data/interim/full_v0/dl/.

Dry run by default; --apply writes. Never deletes or overwrites: a differing destination file is a conflict.
Exit 0 = ok, 2 = refused (nothing written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT = "train_report_dl.json"
ENV_FILE = "environment.json"
ENV_KEYS = ("torch", "cuda", "gpu")


class Refused(Exception):
    pass


@dataclass
class Plan:
    copy: list[tuple[Path, Path]] = field(default_factory=list)
    present: list[Path] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    done_models: list[str] = field(default_factory=list)
    fits: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    report_changed: bool = False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report(path: Path, what: str) -> dict[str, Any]:
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused(f"cannot read {what} {path}: {exc}") from exc
    except ValueError as exc:
        raise Refused(f"invalid JSON in {what} {path}: {exc}") from exc
    if not isinstance(rep, dict) or not isinstance(rep.get("fits"), dict):
        raise Refused(f"{what} {path} has no 'fits' object")
    env = rep.get("environment", {})
    if not isinstance(env, dict):
        raise Refused(f"{what} {path} has a non-object 'environment'")
    return rep


def _kaggle(rep: dict[str, Any], what: str) -> dict[str, Any] | None:
    kag = rep.get("environment", {}).get("kaggle")
    if kag is None:
        return None
    if not isinstance(kag, dict):
        raise Refused(f"{what}: environment.kaggle is not an object")
    return kag


def _plan(batches: list[Path], dest: Path) -> Plan:
    plan = Plan()
    existing: dict[str, Any] = {}
    existing_env: dict[str, Any] = {}
    rep_dst = dest / REPORT
    if rep_dst.exists():
        old = _load_report(rep_dst, "destination report")
        existing, existing_env = old["fits"], dict(old.get("environment", {}))

    kaggles: list[tuple[str, dict[str, Any]]] = []
    if (kag := _kaggle({"environment": existing_env}, "destination report")) is not None:
        kaggles.append(("destination report", kag))

    fits: dict[str, Any] = dict(existing)
    seen_batch: set[str] = set()
    planned: dict[Path, str] = {}
    for b in batches:
        if not b.is_dir():
            raise Refused(f"batch folder not found: {b}")
        if not (b / REPORT).is_file():
            raise Refused(f"batch {b} has no {REPORT}")
        rep = _load_report(b / REPORT, "batch report")
        kag = _kaggle(rep, f"batch {b}")
        if kag is None:
            raise Refused(f"batch {b}: report has no environment.kaggle block")
        kaggles.append((str(b), kag))

        for mid, entry in rep["fits"].items():
            pred = b / f"predictions_{mid}_v0.parquet"
            want = entry.get("prediction_sha256") if isinstance(entry, dict) else None
            if not pred.is_file():
                raise Refused(f"batch {b}: predictions file missing for {mid}: {pred.name}")
            if _sha(pred) != want:
                raise Refused(f"batch {b}: {pred.name} does not match prediction_sha256 of its own report")
            if mid in fits and fits[mid] != entry:
                where = "destination report" if mid in existing and mid not in seen_batch else "another batch"
                raise Refused(f"model {mid}: entry differs from the one in {where}")
            if mid in existing or mid in seen_batch:
                if mid not in plan.done_models:
                    plan.done_models.append(mid)
            else:
                plan.new_models.append(mid)
            seen_batch.add(mid)
            fits[mid] = entry

        for src in sorted(b.rglob("*")):
            if not src.is_file() or src.name in (REPORT, ENV_FILE):
                continue
            dst = dest / src.relative_to(b)
            digest = _sha(src)
            if dst in planned:
                if planned[dst] != digest:
                    raise Refused(f"batches disagree on {dst.relative_to(dest).as_posix()}")
            elif dst.exists():
                if _sha(dst) != digest:
                    raise Refused(f"conflict, destination differs and is not overwritten: {dst}")
                plan.present.append(dst)
            else:
                plan.copy.append((src, dst))
            planned[dst] = digest

    sigs = {tuple(k.get(key) for key in ENV_KEYS) for _, k in kaggles}
    if len(sigs) != 1:
        raise Refused(f"Kaggle environments differ ({', '.join(ENV_KEYS)}): {sorted(map(str, sigs))}")
    firsts = [str(k["first_seen"]) for _, k in kaggles if k.get("first_seen")]
    merged_kag = {**kaggles[0][1], **({"first_seen": min(firsts)} if firsts else {})}
    plan.environment = {**existing_env, "kaggle": merged_kag}
    plan.fits = dict(sorted(fits.items()))
    plan.report_changed = not rep_dst.exists() or json.loads(rep_dst.read_text(encoding="utf-8")) != {
        "environment": plan.environment,
        "fits": plan.fits,
    }
    return plan


def _write_report(path: Path, plan: Plan) -> None:
    text = json.dumps({"environment": plan.environment, "fits": plan.fits}, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _verify(dest: Path, fits: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for mid, entry in fits.items():
        pred = dest / f"predictions_{mid}_v0.parquet"
        if not pred.is_file() or _sha(pred) != entry["prediction_sha256"]:
            bad.append(mid)
    return bad


def _show(plan: Plan, dest: Path) -> None:
    def rel(p: Path) -> str:
        return p.relative_to(dest).as_posix()

    print(f"files to copy: {len(plan.copy)}")
    for _, dst in plan.copy:
        print(f"  {rel(dst)}")
    print(f"files already present (identical): {len(plan.present)}")
    print(f"new models: {len(plan.new_models)}" + (f" ({', '.join(sorted(plan.new_models))})" if plan.new_models else ""))
    print(f"models already consolidated: {len(plan.done_models)}")
    print(f"report changes: {plan.report_changed}")
    print("merged environment:", json.dumps(plan.environment, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, action="append", required=True, help="extracted dl/ folder of a batch")
    ap.add_argument("--dest", type=Path, default=None, help="default: data/interim/full_v0/dl under the repo")
    ap.add_argument("--apply", action="store_true", help="write; without it only the plan is printed")
    a = ap.parse_args(argv)
    dest = a.dest if a.dest is not None else REPO_ROOT / "data" / "interim" / "full_v0" / "dl"
    try:
        plan = _plan(a.batch, dest)
        _show(plan, dest)
        if not a.apply:
            print("(dry run: nothing was written)")
            return 0
        try:
            for src, dst in plan.copy:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            if plan.report_changed:
                dest.mkdir(parents=True, exist_ok=True)
                _write_report(dest / REPORT, plan)
        except OSError as exc:
            print(f"consolidation failed while writing: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        bad = _verify(dest, plan.fits)
        n = len(plan.fits)
        if bad:
            print(f"verification FAILED for {len(bad)} of {n}: {', '.join(sorted(bad))}", file=sys.stderr)
            return 2
        print(f"copied {len(plan.copy)} files; OK ({n} of {n})")
    except Refused as exc:
        print(f"consolidation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
