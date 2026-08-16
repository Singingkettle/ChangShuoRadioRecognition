#!/usr/bin/env python3
"""Ladder-only audit: how much a frozen hard model gains from each per-bin rung.

No training-time method needed. This answers "is there any per-bin slack left in this
frozen model at all", which is the question RadioML2018.01A raised: there the method,
all three rungs and the hard model landed on the same band accuracy. To know whether
that is a property of the dataset rather than of one backbone, we need the same
ladder on a second backbone, and for that a matched method is not required.

    python ladder_only.py --hard <dir with seed_*/predictions/{validation,test}.pkl> \\
        --out out.csv --tag "MCformer / RML2018.01A"
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
from scipy.stats import t as student_t

from ladder_lib import sm, acc, band_mask, per_bin, EPS


def load(p):
    d = pickle.load(open(p, "rb"))
    return (np.log(np.clip(np.asarray(d["pps"], float), EPS, 1)),
            np.asarray(d["gts"]).astype(int), np.asarray(d["snrs"]).astype(float))


def tci(xs):
    xs = np.asarray(xs, float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = float(student_t.ppf(0.975, n - 1)) * float(xs.std(ddof=1)) / np.sqrt(n)
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", default=["seed_2026", "seed_2027", "seed_2028"])
    a = ap.parse_args()

    hard_dir = Path(os.path.expanduser(a.hard))
    print(f"{a.tag}   {hard_dir}", flush=True)
    print(f"{'seed':6s} | {'K':>3s} {'bins':>5s} {'n_band':>8s} | {'hard':>6s} "
          f"{'shift':>6s} {'VS':>6s} {'aff':>6s} | {'aff-hard':>8s}", flush=True)
    rows, lifts = [], []
    for s in a.seeds:
        hv = hard_dir / s / "predictions" / "validation.pkl"
        ht = hard_dir / s / "predictions" / "test.pkl"
        if not (hv.exists() and ht.exists()):
            print(f"{s[-4:]:6s} | missing", flush=True)
            continue
        vZ, vY, vR = load(str(hv))
        tZ, tY, tR = load(str(ht))
        hp = sm(tZ)
        band = band_mask(hp, tY, tR)
        hard_b = acc(hp[band], tY[band])
        res = {}
        for rung in ("shift", "vs", "aff"):
            res[rung] = acc(per_bin(rung, vZ, vY, vR, tZ, tR)[band], tY[band])
        lift = res["aff"] - hard_b
        lifts.append(lift)
        print(f"{s[-4:]:6s} | {tZ.shape[1]:3d} {np.unique(tR).size:5d} "
              f"{int(band.sum()):8d} | {hard_b:6.2f} {res['shift']:6.2f} "
              f"{res['vs']:6.2f} {res['aff']:6.2f} | {lift:+8.2f}", flush=True)
        rows.append(dict(seed=s[-4:], K=int(tZ.shape[1]), n_bins=int(np.unique(tR).size),
                         n_band=int(band.sum()), hard=round(hard_b, 2),
                         F_shift=round(res["shift"], 2), F_VS=round(res["vs"], 2),
                         F_aff=round(res["aff"], 2), aff_minus_hard=round(lift, 2)))
    if not rows:
        print("no seeds", flush=True)
        return
    m, lo, hi = tci(lifts)
    print(f"\nreadout lift over the frozen model: {m:+.2f} pp  95% t-CI [{lo:+.2f},{hi:+.2f}]",
          flush=True)
    print("-> slack exists" if lo > 0 else "-> no per-bin slack to extract", flush=True)
    out = Path(os.path.expanduser(a.out))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
