#!/usr/bin/env python
"""JDM ideal-fair reproduction ladder (two-machine goal mode, 2026-07-23).

ROOT-CAUSE FIX this encapsulates: the paper "ideal" (Fig. 8/13, infdB / no
random impairments == CSRD v1) must be measured by evaluating the best
FULL-DATA-trained detector/joint checkpoint on the clean v1 TEST split -- not
by retraining on v1-only (which underfits to mAP ~0.31).

The ladder is idempotent (skips a step when its output eval dir already has a
metrics JSON) and writes eval dirs that ``tools.goal_mode_helpers`` reads for
dual-protocol scoring (tightened 2026-07-24 to paper + generate.m):
    eval_ideal_v1_det_testonly              (ideal det, v1 test)
    eval_simulate_real_awgn_det_testonly    (simulate det, Real+Real_awgn)
    eval_ideal_v1_joint_testonly            (ideal joint, v1 test)
    eval_simulate_real_awgn_joint_testonly  (simulate joint, Real+Real_awgn)

Fig. 8/13 "simulate" is NOT the full 124-version mixture (that inflates the
bar by mixing ideal / pure AWGN / single-factor ablations).

Eval-only (default): discover the best existing det/AMC checkpoints and score
both protocols -- run this on a box that HAS the checkpoints + CSRD data (the
local 2-GPU box).

--train: first train det (full data, 30ep) and AMC if no checkpoint exists,
then evaluate -- run this on the H100 box after CSRD data is copied.

Usage:
    python tools/jdm/ideal_fair_ladder.py --gpu 0
    python tools/jdm/ideal_fair_ladder.py --gpu 2 --train
    python tools/jdm/ideal_fair_ladder.py --det-ckpt <p> --amc-ckpt <p> --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
PYTHON = os.environ.get("PYTHON", sys.executable)
RETUNE = REPO / "work_dirs" / "jdm" / "retune"

DET_TRAIN_CFG = REPO / "configs/jdm/jdm-det_fft-csrd.py"
DET_SIM_EVAL_CFG = (
    REPO / "configs/jdm/experiments/retune/eval_simulate_real_awgn_det_testonly.py"
)
DET_IDEAL_EVAL_CFG = REPO / "configs/jdm/experiments/retune/eval_ideal_v1_det_testonly.py"
JOINT_SIM_EVAL_CFG = (
    REPO / "configs/jdm/experiments/retune/eval_simulate_real_awgn_joint_testonly.py"
)
JOINT_IDEAL_EVAL_CFG = REPO / "configs/jdm/experiments/retune/eval_ideal_v1_joint_testonly.py"
AMC_TRAIN_CFG = REPO / "configs/jdm/experiments/retune/amc_wave3b_detprops_30ep.py"

# Where to look for the best FULL-data detector checkpoint (mixed-trained).
# Prefer longer full-data rungs; exclude det_ideal_v1_* (v1-only underfit).
DET_CKPT_GLOBS = [
    "det_full_120ep_lr1e3/best_detection_mAP_epoch_*.pth",
    "det_full_90ep_lr1e3/best_detection_mAP_epoch_*.pth",
    "det_full_90ep_lr5e4/best_detection_mAP_epoch_*.pth",
    "det_full_60ep_lr1e3/best_detection_mAP_epoch_*.pth",
    "det_full_30ep/best_detection_mAP_epoch_*.pth",
    "det_wave3b_5ep_lr1e3/best_detection_mAP_epoch_*.pth",
    "det_30ep_anchor096146_bw20/best_detection_mAP_epoch_*.pth",
    "det_wave*/best_detection_mAP_epoch_*.pth",
]
AMC_CKPT_GLOBS = [
    "amc_wave3b_detprops_30ep/best_accuracy_top1_epoch_*.pth",
    "amc_*detprops*/best_accuracy_top1_epoch_*.pth",
]


def log(msg: str) -> None:
    print(f"[ideal_fair_ladder] {msg}", flush=True)


def _newest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def discover_ckpt(globs: list[str]) -> Path | None:
    for pattern in globs:
        hit = _newest(list(RETUNE.glob(pattern)))
        if hit is not None:
            return hit
    return None


def _has_metrics(work_dir: Path) -> bool:
    if not work_dir.is_dir():
        return False
    for p in work_dir.rglob("*.json"):
        if p.name in ("snr_curve.json", "GOAL_STATUS.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and (
            "detection/mAP" in data or "accuracy/top1" in data
        ):
            return True
    return False


def run(cmd: list[str], gpu: str, log_path: Path) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log(f"gpu={gpu} $ {' '.join(str(c) for c in cmd)}  (log {log_path})")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def test_det(cfg: Path, ckpt: Path, work_dir: Path, gpu: str) -> None:
    if _has_metrics(work_dir):
        log(f"skip eval (metrics present): {work_dir.name}")
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    run(
        [PYTHON, str(REPO / "tools/test_det.py"), str(cfg), str(ckpt),
         "--work-dir", str(work_dir)],
        gpu, work_dir / "eval.log",
    )


def train(cfg: Path, work_dir: Path, gpu: str, done_ckpt_glob: str) -> Path | None:
    existing = _newest(list(work_dir.glob(done_ckpt_glob)))
    if existing is not None:
        log(f"skip train (ckpt present): {existing.relative_to(REPO)}")
        return existing
    work_dir.mkdir(parents=True, exist_ok=True)
    run(
        [PYTHON, str(REPO / "tools/train.py"), str(cfg), "--work-dir", str(work_dir)],
        gpu, work_dir / "train.log",
    )
    return _newest(list(work_dir.glob(done_ckpt_glob)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0", help="single physical GPU id")
    ap.add_argument("--train", action="store_true",
                    help="train det (full 30ep) + AMC if no checkpoint exists")
    ap.add_argument("--det-ckpt", default=None)
    ap.add_argument("--amc-ckpt", default=None)
    args = ap.parse_args()
    gpu = str(args.gpu)
    RETUNE.mkdir(parents=True, exist_ok=True)

    det_ckpt = Path(args.det_ckpt) if args.det_ckpt else None
    amc_ckpt = Path(args.amc_ckpt) if args.amc_ckpt else None

    if args.train:
        if det_ckpt is None or not det_ckpt.is_file():
            log("train: full-data detector (30ep, all versions)")
            det_ckpt = train(
                DET_TRAIN_CFG, RETUNE / "det_full_30ep", gpu,
                "best_detection_mAP_epoch_*.pth",
            )
        if amc_ckpt is None or not amc_ckpt.is_file():
            if AMC_TRAIN_CFG.is_file():
                log("train: AMC proposal-crop")
                amc_ckpt = train(
                    AMC_TRAIN_CFG, RETUNE / "amc_wave3b_detprops_30ep", gpu,
                    "best_accuracy_top1_epoch_*.pth",
                )

    if det_ckpt is None or not det_ckpt.is_file():
        det_ckpt = discover_ckpt(DET_CKPT_GLOBS)
    if amc_ckpt is None or not amc_ckpt.is_file():
        amc_ckpt = discover_ckpt(AMC_CKPT_GLOBS)

    if det_ckpt is None:
        log("ERROR: no detector checkpoint found; cannot evaluate ideal protocol")
        return 2
    log(f"detector ckpt: {det_ckpt}")
    log(f"amc ckpt: {amc_ckpt}")

    # --- Detector: ideal (v1) + simulate (Real+Real_awgn) ---
    test_det(DET_IDEAL_EVAL_CFG, det_ckpt,
             RETUNE / "eval_ideal_v1_det_testonly", gpu)
    test_det(DET_SIM_EVAL_CFG, det_ckpt,
             RETUNE / "eval_simulate_real_awgn_det_testonly", gpu)

    # --- Joint: merge det + AMC, eval ideal + simulate ---
    if amc_ckpt is not None and amc_ckpt.is_file():
        merged = RETUNE / "jdm_joint_ideal_fair_amc.pth"
        if not merged.is_file():
            log("merge det + AMC -> joint checkpoint")
            run(
                [PYTHON, str(REPO / "tools/merge_jdm_checkpoints.py"),
                 str(det_ckpt), str(amc_ckpt), str(merged)],
                gpu, RETUNE / "merge_ideal_fair.log",
            )
        if merged.is_file():
            test_det(JOINT_IDEAL_EVAL_CFG, merged,
                     RETUNE / "eval_ideal_v1_joint_testonly", gpu)
            test_det(JOINT_SIM_EVAL_CFG, merged,
                     RETUNE / "eval_simulate_real_awgn_joint_testonly", gpu)
    else:
        log("no AMC checkpoint; skipping joint eval")

    log("ideal-fair ladder complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
