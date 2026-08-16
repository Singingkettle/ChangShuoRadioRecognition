#!/usr/bin/env python3
"""Family E-4a: per-bin whitening ladder V0-V3 (ceiling campaign, pre-registered).

Question: can LABEL-FREE per-bin feature normalization (mean/std or full
Ledoit-Wolf whitening from validation statistics) + ONE global linear head
recover the decision deficit that the labelled per-bin probe (ceiling proxy)
recovers over a global probe?

  V0  global probe          : global (val) mean/std standardization -> one LR
  V1  per-bin probe         : one LR per SNR bin (>=50 val samples else global
                              fallback), raw phi, paper probe protocol
  V2  per-bin standardize   : per-bin val mean / per-dim std -> one global LR
  V3  per-bin whiten        : per-bin val Ledoit-Wolf covariance^-1/2 -> one
                              global LR
All fits on validation only; evaluation on test. Test-bin statistics always
come from the SAME bin's validation stats (deployment consistency); test bins
absent from validation fall back to global val stats. Band = hard-model 20-80%
accuracy band from test pps/gts/snrs (identical rule to
scripts/ladder/ladder_lib.band_mask). Shared classifier across rungs:
LogisticRegression(max_iter=3000, C=1.0). CPU only; data opened read-only.

Usage:
  python whitening_ladder.py --dir DIR --prefix hard --cell "petcgdnn_10b/10B" \
      --out summary.csv --bins-out bins.csv [--seeds 2026 2027 2028]
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

MIN_BIN = 50      # min val samples for a per-bin probe fit (V1), paper protocol
MAX_ITER = 3000
C_REG = 1.0
SD_FLOOR = 1e-8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (np.asarray(d["phi"], dtype=np.float64),
            np.asarray(d["pps"], dtype=np.float64),
            np.asarray(d["gts"], dtype=int),
            np.asarray(d["snrs"], dtype=float))


def band_from_hard(hp, y, r):
    """20-80% accuracy band of the hard model (ladder_lib.band_mask rule)."""
    bins = np.unique(r)
    a = np.array([float((hp[r == b].argmax(1) == y[r == b]).mean() * 100) for b in bins])
    s = a.max() - a.min()
    if s < 1e-9:
        return np.ones(y.size, bool), bins
    keep = bins[(a >= a.min() + 0.2 * s) & (a <= a.min() + 0.8 * s)]
    if keep.size == 0:
        keep = bins[np.argsort(a)[len(bins) // 2: len(bins) // 2 + 1]]
    return np.isin(r, keep), keep


def fit_lr(X, y):
    clf = LogisticRegression(max_iter=MAX_ITER, C=C_REG)
    clf.fit(X, y)
    return clf


def inv_sqrt(cov):
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, w.max() * 1e-10 + 1e-12)
    return (V * (w ** -0.5)) @ V.T


def v0_pred(vphi, vy, tphi):
    mu = vphi.mean(0)
    sd = np.maximum(vphi.std(0), SD_FLOOR)
    clf = fit_lr((vphi - mu) / sd, vy)
    return clf.predict((tphi - mu) / sd)


def v1_pred(vphi, vy, vr, tphi, tr):
    """Per-bin probe, paper protocol (raw phi; >=MIN_BIN & >=2 classes else global)."""
    glob = [None]

    def get_glob():
        if glob[0] is None:
            log("    V1: fitting global fallback (raw phi)")
            glob[0] = fit_lr(vphi, vy)
        return glob[0]

    pred = np.empty(tphi.shape[0], dtype=int)
    n_fb = 0
    for b in np.unique(vr):
        vm = vr == b
        tm = tr == b
        if not tm.any():
            continue
        if vm.sum() >= MIN_BIN and np.unique(vy[vm]).size > 1:
            clf = fit_lr(vphi[vm], vy[vm])
        else:
            clf = get_glob()
            n_fb += 1
        pred[tm] = clf.predict(tphi[tm])
    un = ~np.isin(tr, np.unique(vr))
    if un.any():
        pred[un] = get_glob().predict(tphi[un])
        n_fb += 1
    return pred, n_fb


def transform_perbin(vphi, vr, tphi, tr, mode):
    """Label-free per-bin transform. Stats from val only; same-bin stats applied
    to test; test bins absent from val -> global val stats."""
    vX = np.empty_like(vphi)
    tX = np.empty_like(tphi)
    for b in np.unique(vr):
        vm = vr == b
        mu = vphi[vm].mean(0)
        if mode == "std":
            sd = np.maximum(vphi[vm].std(0), SD_FLOOR)
            vX[vm] = (vphi[vm] - mu) / sd
        else:
            W = inv_sqrt(LedoitWolf().fit(vphi[vm]).covariance_)
            vX[vm] = (vphi[vm] - mu) @ W
        tm = tr == b
        if tm.any():
            if mode == "std":
                tX[tm] = (tphi[tm] - mu) / sd
            else:
                tX[tm] = (tphi[tm] - mu) @ W
    un = ~np.isin(tr, np.unique(vr))
    if un.any():
        gmu = vphi.mean(0)
        if mode == "std":
            gsd = np.maximum(vphi.std(0), SD_FLOOR)
            tX[un] = (tphi[un] - gmu) / gsd
        else:
            gW = inv_sqrt(LedoitWolf().fit(vphi).covariance_)
            tX[un] = (tphi[un] - gmu) @ gW
    return vX, tX


def per_bin_acc(pred, y, r):
    return {float(b): (float((pred[r == b] == y[r == b]).mean() * 100), int((r == b).sum()))
            for b in np.unique(r)}


def run_seed(cell, seed, phi_dir, prefix, wsum, wbins):
    t0 = time.time()
    vphi, _vpp, vy, vr = load(phi_dir / f"{prefix}_seed{seed}_validation.pkl")
    tphi, tpp, ty, tr = load(phi_dir / f"{prefix}_seed{seed}_test.pkl")
    band, keep = band_from_hard(tpp, ty, tr)
    log(f"  {cell} seed {seed}: val {vphi.shape} test {tphi.shape} "
        f"classes {np.unique(vy).size} bins {np.unique(vr).size} "
        f"band bins {list(keep)} (n_band {int(band.sum())}/{ty.size})")

    hard_pred = tpp.argmax(1)
    preds = {"hard": hard_pred}

    log("    V0: global standardized probe")
    preds["V0"] = v0_pred(vphi, vy, tphi)
    log(f"    V0 done ({time.time()-t0:.0f}s)")

    log("    V1: per-bin probe (ceiling proxy)")
    preds["V1"], n_fb = v1_pred(vphi, vy, vr, tphi, tr)
    log(f"    V1 done, fallback bins {n_fb} ({time.time()-t0:.0f}s)")

    log("    V2: per-bin standardization -> global probe")
    vX, tX = transform_perbin(vphi, vr, tphi, tr, "std")
    preds["V2"] = fit_lr(vX, vy).predict(tX)
    log(f"    V2 done ({time.time()-t0:.0f}s)")

    log("    V3: per-bin Ledoit-Wolf whitening -> global probe")
    vX, tX = transform_perbin(vphi, vr, tphi, tr, "lw")
    preds["V3"] = fit_lr(vX, vy).predict(tX)
    log(f"    V3 done ({time.time()-t0:.0f}s)")

    keepset = set(float(b) for b in keep)
    for name, pred in preds.items():
        b_acc = float((pred[band] == ty[band]).mean() * 100)
        a_acc = float((pred == ty).mean() * 100)
        wsum.writerow([cell, seed, name, f"{b_acc:.4f}", f"{a_acc:.4f}"])
        for b, (acc_b, n_b) in per_bin_acc(pred, ty, tr).items():
            wbins.writerow([cell, seed, name, b, n_b, f"{acc_b:.4f}",
                            int(b in keepset)])
        log(f"    {name}: band {b_acc:.2f}  all {a_acc:.2f}")
    log(f"  {cell} seed {seed} finished in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", default="hard")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", nargs="+", default=["2026", "2027", "2028"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins-out", required=True)
    args = ap.parse_args()

    phi_dir = Path(os.path.expanduser(args.dir))
    new_sum = not Path(args.out).exists()
    new_bins = not Path(args.bins_out).exists()
    with open(args.out, "a", newline="") as fs, open(args.bins_out, "a", newline="") as fb:
        wsum = csv.writer(fs)
        wbins = csv.writer(fb)
        if new_sum:
            wsum.writerow(["cell", "seed", "variant", "band_acc", "all_acc"])
        if new_bins:
            wbins.writerow(["cell", "seed", "variant", "snr", "n_test", "acc", "in_band"])
        for seed in args.seeds:
            run_seed(args.cell, seed, phi_dir, args.prefix, wsum, wbins)
            fs.flush()
            fb.flush()
    log(f"ALL DONE cell={args.cell}")


if __name__ == "__main__":
    main()
