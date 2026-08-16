#!/usr/bin/env python3
"""Full ladder audit of one matched (hard, method) pair, at the default iteration budget.

Reports both quantities the E-B2 pre-registration turns on:
  readout lift   = F_aff - hard   (is there any per-bin slack in the frozen model?)
  method - F_aff                  (does the trained method reach past the readout?)
plus method - hard, and every rung, per seed and as a t-CI at df=n-1.

    python pair_ladder.py --hard <dir> --method <dir> --out out.csv --tag "..."
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
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", default=["seed_2026", "seed_2027", "seed_2028"])
    a = ap.parse_args()

    H = Path(os.path.expanduser(a.hard))
    M = Path(os.path.expanduser(a.method))
    print(f"{a.tag}\n  hard   {H}\n  method {M}", flush=True)
    print(f"{'seed':6s} | {'K':>3s} {'bins':>5s} {'n_band':>8s} | {'hard':>6s} {'meth':>6s} "
          f"{'shift':>6s} {'VS':>6s} {'aff':>6s} | {'aff-hard':>8s} {'meth-aff':>8s} "
          f"{'meth-hard':>9s}", flush=True)
    rows, lift, gap, mh = [], [], [], []
    for s in a.seeds:
        hv, ht = H / s / "predictions" / "validation.pkl", H / s / "predictions" / "test.pkl"
        mt = M / s / "predictions" / "test.pkl"
        if not (hv.exists() and ht.exists() and mt.exists()):
            print(f"{s[-4:]:6s} | missing", flush=True)
            continue
        vZ, vY, vR = load(str(hv))
        tZ, tY, tR = load(str(ht))
        mP = sm(load(str(mt))[0])
        if mP.shape[0] != tY.size:
            print(f"{s[-4:]:6s} | size mismatch", flush=True)
            continue
        hp = sm(tZ)
        band = band_mask(hp, tY, tR)
        res = {r: acc(per_bin(r, vZ, vY, vR, tZ, tR)[band], tY[band])
               for r in ("shift", "vs", "aff")}
        hb = acc(hp[band], tY[band])
        mb = acc(mP[band], tY[band])
        lift.append(res["aff"] - hb)
        gap.append(mb - res["aff"])
        mh.append(mb - hb)
        print(f"{s[-4:]:6s} | {tZ.shape[1]:3d} {np.unique(tR).size:5d} {int(band.sum()):8d} | "
              f"{hb:6.2f} {mb:6.2f} {res['shift']:6.2f} {res['vs']:6.2f} {res['aff']:6.2f} | "
              f"{lift[-1]:+8.2f} {gap[-1]:+8.2f} {mh[-1]:+9.2f}", flush=True)
        rows.append(dict(seed=s[-4:], K=int(tZ.shape[1]), n_bins=int(np.unique(tR).size),
                         n_band=int(band.sum()), hard=round(hb, 2), method=round(mb, 2),
                         F_shift=round(res["shift"], 2), F_VS=round(res["vs"], 2),
                         F_aff=round(res["aff"], 2), aff_minus_hard=round(lift[-1], 2),
                         method_minus_aff=round(gap[-1], 2),
                         method_minus_hard=round(mh[-1], 2)))
    if not rows:
        print("no seeds", flush=True)
        return
    for name, xs in (("readout lift (F_aff - hard)", lift),
                     ("method - F_aff", gap),
                     ("method - hard", mh)):
        m, lo, hi = tci(xs)
        verdict = ("> 0" if lo > 0 else "< 0" if hi < 0 else "includes zero")
        print(f"{name:28s} {m:+7.2f} pp  95% t-CI [{lo:+.2f},{hi:+.2f}]   {verdict}",
              flush=True)
    out = Path(os.path.expanduser(a.out))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
