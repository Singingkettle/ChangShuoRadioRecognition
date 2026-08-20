#!/usr/bin/env python
"""Build a count-matched predicted-box crop cache for one detector family.

Two steps, scripted so the cross-detector families are all built identically
(the FCOS/RTMDet matched caches were originally subsampled with an unrecorded
manual one-liner -- this makes it reproducible):

  1. bridge.py build-pred --split train  -> the FULL set of IoU-matched
     predicted-box crops (score-thr 0.1, min-iou 0.5): trainpred_<fam>_L1024.npz
  2. uniform-random subsample to the GT cache count (174136, fixed seed) ->
     trainpred_<fam>_matched_L1024.npz. Uniform (NOT stratified) so the
     detector's PREDICTED class/quality distribution is preserved; only the
     crop COUNT is matched to the GT recogniser's training set, so the
     predicted-box vs GT-box delta is not confounded by training-set size.

Writes summary.csv so the file-queue worker marks the cell done.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data/citybuster/iqdet_repro/ChangShuoRadioRecognition")
CACHE = REPO / "work_dirs" / "returniq_cache"
PY = os.environ.get("VENV", "/data/citybuster/iqdet_repro/venv") + "/bin/python"
GT_COUNT = 174136  # train_L1024.npz box count -- the count every family is matched to


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--fam", required=True, help="family tag, e.g. atss")
    ap.add_argument("--baseline-pred", required=True, help="<fam>_traindump/source_data/test_predictions.bbox.json")
    ap.add_argument("--n", type=int, default=GT_COUNT)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--score-thr", type=float, default=0.1)
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if not work.is_absolute():
        work = REPO / work
    work.mkdir(parents=True, exist_ok=True)

    full = f"trainpred_{args.fam}_L1024.npz"
    matched = f"trainpred_{args.fam}_matched_L1024.npz"
    full_path = CACHE / full
    matched_path = CACHE / matched

    # Step 1: full build-pred (skip if the full cache already exists).
    if not full_path.exists():
        cmd = [PY, "-u", "configs/detection_is_easy/bridge.py", "build-pred",
               "--split", "train", "--baseline-pred", args.baseline_pred,
               "--out", full, "--L", "1024",
               "--score-thr", str(args.score_thr), "--min-iou", str(args.min_iou)]
        print("[build-pred-matched] " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=REPO, env=os.environ).returncode
        if rc != 0:
            print(f"[build-pred-matched] build-pred failed rc={rc}", flush=True)
            return rc
    else:
        print(f"[build-pred-matched] reusing existing {full_path}", flush=True)

    # Step 2: uniform-random subsample to the GT count (fixed seed).
    d = np.load(full_path)
    X, y, cfo = d["X"], d["y"], d["cfo"]
    N = y.shape[0]
    n = min(args.n, N)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(N, size=n, replace=False))
    np.savez(matched_path, X=X[idx], y=y[idx], cfo=cfo[idx])
    print(f"[build-pred-matched] {full} N={N} -> {matched} n={n} (seed {args.seed})", flush=True)

    with (work / "summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("split,metric,value\n")
        fh.write("run,exit_code,0\n")
        fh.write(f"train,full_boxes,{N}\n")
        fh.write(f"train,matched_boxes,{n}\n")
    print(f"[build-pred-matched] wrote {work / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
