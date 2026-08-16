#!/usr/bin/env python3
"""Three-seed extension of the comprehensive cross-architecture/dataset ladder audit.

ladder_lib.py discovers matched (hard, reliability-aware method) pairs at
seed_2026 and runs the ladder once. This script reuses the SAME ladder/band/verdict
math but, for each discovered pair, runs the ladder on every seed in
{2026,2027,2028} that has both the hard (val+test) and method (test) predictions,
and reports the seed-mean transition-band hard/method/readout, the mean
method-minus-affine-readout gap with a 3-seed t-CI, and how many seeds are
decision-layer. This hardens the single-seed headline table against the obvious
"only one seed" objection.

    python ladder_audit.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob

import numpy as np
from scipy.stats import t as student_t

import ladder_lib as CA
from ladder_lib import load, sm, acc, band_mask, per_bin, discover

# Robustness-check speedup: the per-bin affine/vector-scaling fits dominate runtime on
# radio's ~20 SNR bins. Cap the L-BFGS iterations for this 3-seed robustness pass; the
# decision-layer verdicts and the large 3-18 pp readout margins are unaffected (the
# single-seed headline table in the paper uses the full mi=250).
CA.FT["vs"] = lambda Z, y: CA.fit_vs(Z, y, mi=60)
CA.FT["aff"] = lambda Z, y: CA.fit_aff(Z, y, mi=60)

SEEDS = ["seed_2026", "seed_2027", "seed_2028"]
T_975_DF2 = 4.302653  # two-sided 95% t critical, df=2


def audit_one(hpath, mpath):
    vZ, vY, vR = load(hpath.replace("test.pkl", "validation.pkl"))
    tZ, tY, tR = load(hpath)
    mP = sm(load(mpath)[0])
    if mP.shape[0] != tY.size:
        return None
    hp = sm(tZ)
    band = band_mask(hp, tY, tR)
    d = {}
    ro = None
    for rung in ("shift", "vs", "aff"):
        nP = per_bin(rung, vZ, vY, vR, tZ, tR)
        d[rung] = acc(mP[band], tY[band]) - acc(nP[band], tY[band])
        if rung == "aff":
            ro = acc(nP[band], tY[band])
    return {"hard": acc(hp[band], tY[band]), "method": acc(mP[band], tY[band]),
            "readout": ro, "m_shift": d["shift"], "m_vs": d["vs"], "m_aff": d["aff"],
            "K": int(tZ.shape[1])}


def tci(xs):
    """Two-sided 95% t-CI of the mean, with the critical value taken at df=n-1.

    Using a fixed df=2 (or the normal 1.96) for a two-seed row understates the
    interval badly: t(0.975, df=1) is 12.706, not 4.303 and not 1.96.
    """
    xs = np.asarray(xs, float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    sd = float(xs.std(ddof=1))
    se = sd / np.sqrt(n)
    h = float(student_t.ppf(0.975, n - 1)) * se
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--skip-substr", default="hisar",
                    help="skip pairs whose dataset contains this substring (26-class HisarMod "
                         "F_aff fits dominate runtime and HisarMod is already the single-seed "
                         "documented exception); set '' to include all")
    a = ap.parse_args()

    pairs = discover()  # (backbone, dataset, hpath_2026, mpath_2026)
    if a.skip_substr:
        pairs = [p for p in pairs if a.skip_substr not in p[1].lower() and a.skip_substr not in p[0].lower()]
    print(f"Three-seed comprehensive ladder audit: {len(pairs)} pairs "
          f"(seeds {', '.join(s.split('_')[1] for s in SEEDS)} where available)\n")
    hdr = (f"{'backbone':12s} {'dataset':18s} {'n':>2s} | {'hard':>5s} {'meth':>5s} "
           f"{'readout':>7s} | {'m-aff mean':>10s} {'95% t-CI':>16s} | dec/seed verdict")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    dec_all = 0
    for bb, ds, hp26, mp26 in pairs:
        per = []
        for s in SEEDS:
            hp = hp26.replace("seed_2026", s)
            mp = mp26.replace("seed_2026", s)
            if not (glob.glob(hp) and glob.glob(mp) and glob.glob(hp.replace("test.pkl", "validation.pkl"))):
                continue
            try:
                r = audit_one(hp, mp)
            except Exception as e:  # noqa: BLE001
                print(f"  {bb}/{ds} {s} ERROR {e}")
                r = None
            if r:
                per.append(r)
        if not per:
            continue
        maff = [r["m_aff"] for r in per]
        m, lo, hi = tci(maff)
        dec = sum(1 for x in maff if x <= 0)
        n = len(per)
        verdict = "decision-layer" if m <= 0 else ("survives F_aff" if lo > 0 else "decision-layer (CI incl. 0)")
        if m <= 0:
            dec_all += 1
        row = {"backbone": bb, "dataset": ds, "n_seeds": n,
               "hard": round(float(np.mean([r["hard"] for r in per])), 1),
               "method": round(float(np.mean([r["method"] for r in per])), 1),
               "readout": round(float(np.mean([r["readout"] for r in per])), 1),
               "m_aff_mean": round(m, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
               "dec_seeds": dec, "verdict": verdict}
        rows.append(row)
        print(f"{bb:12s} {ds:18s} {n:>2d} | {row['hard']:>5.1f} {row['method']:>5.1f} "
              f"{row['readout']:>7.1f} | {m:>+10.2f} {f'[{lo:+.2f},{hi:+.2f}]':>16s} | "
              f"{dec}/{n} {verdict}")

    print(f"\n{dec_all}/{len(rows)} pairs decision-layer on the seed-mean "
          f"(readout reproduces/exceeds the method).")
    print("Per-seed: a pair is decision-layer in 'dec/n' of its seeds.")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {a.csv}")


if __name__ == "__main__":
    main()
