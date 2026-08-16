#!/usr/bin/env python3
"""Feature-space (representation-level) audit, generic over the phi directory.

Same math as representation_probe.py, but the phi directory, seeds and output path
are arguments, so the D/R/CKA certificate can be run on any (hard, method) pair
whose penultimate features have been dumped by collect_features.py.

For each seed, in the transition band, per reliability bin:
  - frozen hard-phi linear probe: per-bin L2 logistic regression fit on the HARD
    model's validation phi, applied to its test phi (the d-dim upgrade of F_aff).
  - method band accuracy (from the method's test probs).
  - linear CKA between hard-phi and method-phi on the band test samples.

Metric (two-part decomposition of the band gain G = method - hard):
  D   = phiprobe(hard) - hard         (decision-layer part)
  R   = method - phiprobe(hard)       (representation residual; D + R = G)
  RSS = 1 - CKA(hard-phi, method-phi) (geometric representation shift)
R is reported with a t-CI at df=n-1. R ~ 0 (CI includes 0) => decision-layer.

    python representation_probe_generic.py --phi-dir work_dirs/repr_phi_2018A \\
        --out work_dirs/repr_probe_2018A.csv
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

from ladder_lib import acc, band_mask

MIN_BIN = 50


def load(phi_dir, model, seed, split):
    d = pickle.load(open(Path(phi_dir) / f"{model}_seed{seed}_{split}.pkl", "rb"))
    return (d["phi"].astype(np.float64), d["pps"].astype(np.float64),
            d["gts"].astype(int), d["snrs"].astype(float))


def linear_cka(X, Y):
    """Linear CKA between two (n, d) feature matrices (columns centered)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    nx = np.linalg.norm(X.T @ X, ord="fro")
    ny = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (nx * ny + 1e-12))


def per_bin_probe(vphi, vy, vr, tphi, tr, max_iter=3000):
    """Per-bin L2 logistic-regression probe fit on val phi, applied to test phi.

    The features are standardised (statistics taken on validation only) before the
    fit. Standardisation is an invertible affine map, so it does not change the
    hypothesis class the probe can express; it only lets L-BFGS converge. Without it,
    a 24-class 128-dim probe hits the iteration cap and reports an accuracy well below
    what the frozen features actually support, which would understate D and overstate
    the representation residual R.

    Returns (predictions, n_capped) where n_capped counts fits that still hit the cap.
    """
    from sklearn.linear_model import LogisticRegression

    mu = vphi.mean(0, keepdims=True)
    sd = vphi.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    vphi = (vphi - mu) / sd
    tphi = (tphi - mu) / sd

    capped = [0]

    def fit(Xtr, ytr):
        clf = LogisticRegression(max_iter=max_iter, C=1.0)
        clf.fit(Xtr, ytr)
        if np.any(np.asarray(clf.n_iter_) >= max_iter):
            capped[0] += 1
        return clf

    glob = fit(vphi, vy)
    pred = np.empty(tphi.shape[0], dtype=int)
    for b in np.unique(vr):
        vm = vr == b
        tm = tr == b
        if not tm.any():
            continue
        clf = (fit(vphi[vm], vy[vm])
               if (vm.sum() >= MIN_BIN and np.unique(vy[vm]).size > 1) else glob)
        pred[tm] = clf.predict(tphi[tm])
    un = ~np.isin(tr, np.unique(vr))
    if un.any():
        pred[un] = glob.predict(tphi[un])
    return pred, capped[0]


def tci(xs):
    """Two-sided 95% t-CI of the mean, critical value at df=n-1."""
    xs = np.asarray(xs, float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = float(student_t.ppf(0.975, n - 1)) * float(xs.std(ddof=1)) / np.sqrt(n)
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", default=["2026", "2027", "2028"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-iter", type=int, default=3000,
                    help="L-BFGS cap for each per-bin logistic probe")
    a = ap.parse_args()

    print(f"phi dir: {a.phi_dir}   {a.tag}", flush=True)
    rows = []
    print(f"{'seed':6s} | {'hard':>5s} {'phiprobe':>8s} {'method':>6s} | "
          f"{'D':>5s} {'R':>6s} {'CKA':>5s} {'RSS':>5s}", flush=True)
    print("-" * 60, flush=True)
    agg = {k: [] for k in ("hard", "phiprobe", "method", "D", "R", "cka")}
    for s in a.seeds:
        try:
            hphi_v, _, hy_v, hr_v = load(a.phi_dir, "hard", s, "validation")
            hphi_t, hpps_t, hy_t, hr_t = load(a.phi_dir, "hard", s, "test")
            mphi_t, mpps_t, my_t, mr_t = load(a.phi_dir, "method", s, "test")
        except FileNotFoundError as e:
            print(f"{s:6s} | missing: {e.filename}", flush=True)
            continue
        band = band_mask(hpps_t, hy_t, hr_t)
        hard_b = acc(hpps_t[band], hy_t[band])
        method_b = acc(mpps_t[band], my_t[band])
        pred, capped = per_bin_probe(hphi_v, hy_v, hr_v, hphi_t, hr_t,
                                     max_iter=a.max_iter)
        phiprobe_b = float((pred[band] == hy_t[band]).mean() * 100)
        cka = linear_cka(hphi_t[band], mphi_t[band])
        D = phiprobe_b - hard_b
        R = method_b - phiprobe_b
        for k, v in (("hard", hard_b), ("phiprobe", phiprobe_b), ("method", method_b),
                     ("D", D), ("R", R), ("cka", cka)):
            agg[k].append(v)
        print(f"{s:6s} | {hard_b:>5.1f} {phiprobe_b:>8.1f} {method_b:>6.1f} | "
              f"{D:>+5.1f} {R:>+6.2f} {cka:>5.3f} {1-cka:>5.3f}"
              f"   ({capped} fits hit the iteration cap)", flush=True)
        rows.append({"seed": s, "hard": round(hard_b, 2),
                     "phiprobe_hard": round(phiprobe_b, 2), "method": round(method_b, 2),
                     "D": round(D, 2), "R": round(R, 2),
                     "cka": round(cka, 4), "rss": round(1 - cka, 4),
                     "n_band": int(band.sum()), "fits_capped": capped})
    if not rows:
        print("no seeds produced results", flush=True)
        return
    print("-" * 60, flush=True)
    Rm, Rlo, Rhi = tci(agg["R"])
    ckam = float(np.mean(agg["cka"]))
    verdict = ("decision-layer" if not (Rlo > 0) else "representation-level")
    print(f"\n{len(rows)}-seed mean: hard {np.mean(agg['hard']):.1f}  phiprobe(hard) "
          f"{np.mean(agg['phiprobe']):.1f}  method {np.mean(agg['method']):.1f}", flush=True)
    print(f"D (decision part)      {np.mean(agg['D']):+.2f} pp", flush=True)
    print(f"R (repr residual)      {Rm:+.2f} pp  95% t-CI [{Rlo:+.2f}, {Rhi:+.2f}]  "
          f"-> {verdict}", flush=True)
    print(f"RSS = 1-CKA            {1-ckam:.3f}  (CKA {ckam:.3f})", flush=True)
    out = Path(os.path.expanduser(a.out))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
