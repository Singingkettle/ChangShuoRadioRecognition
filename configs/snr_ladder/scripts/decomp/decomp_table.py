#!/usr/bin/env python3
"""The decomposition table: distance-to-Bayes on the anchor benchmark.

Restricted 7-class digital problem (the label space of the exact ceiling): test
frames of the 7 digital classes; model posteriors masked to those classes and
renormalized. For each backbone/seed and SNR bin:

  Acc*(b)     exact/certified Bayes accuracy (ceiling_final.csv, computed from the
              generator only, never from any model)
  probe(b)    per-bin linear probe on the frozen features phi (val fit, test eval)
  readout(b)  per-bin affine readout on the frozen restricted logits (val fit)
  hard(b)     the frozen model's own restricted argmax

  T1+T2 = Acc* - probe   (representation deficit + readout-nonlinearity residual;
                          the split needs a stronger probe family and is reported
                          as a bracket elsewhere)
  T3_probe   = probe - hard    (decision deficit, feature scope)
  T3_readout = readout - hard  (decision deficit, logit scope; lower-variance)

Usage (from the repo root):
  python decomp_table.py --out decomp_synA.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ladder"))
import ladder_lib as CA  # noqa: E402  (load/sm/acc/per_bin/fit_aff)

W = "work_dirs"
PHI = os.path.join(W, "repr_phi_synA")
RUNS = os.path.join(W, "syn_awgn/amc/synthetic_awgn_v1")
DATA = "data/synthetic_awgn_amc_v1"
BACKBONES = ["petcgdnn", "mcformer", "cgdnet"]
SEEDS = ["2026", "2027", "2028"]
DIGITAL = ["8PSK", "BPSK", "CPFSK", "4PAM", "16QAM", "64QAM", "QPSK"]
EPS = 1e-12


def class_order():
    meta = json.load(open(os.path.join(DATA, "test.json")))["metainfo"]
    mods = meta["modulations"]
    dig_idx = [mods.index(m) for m in DIGITAL]
    return mods, dig_idx


def load_pred(path):
    d = pickle.load(open(path, "rb"))
    return (np.asarray(d["pps"], float), np.asarray(d["gts"]).astype(int),
            np.asarray(d["snrs"]).astype(float))


def restrict(pps, gts, snrs, dig_idx):
    """Mask posteriors to the digital classes and relabel gts to 0..6."""
    keep = np.isin(gts, dig_idx)
    p = pps[keep][:, dig_idx]
    p = np.clip(p, EPS, None)
    p /= p.sum(axis=1, keepdims=True)
    remap = {g: i for i, g in enumerate(dig_idx)}
    y = np.array([remap[g] for g in gts[keep]])
    return np.log(p), y, snrs[keep]


def load_phi(bb, seed, split, dig_idx):
    d = pickle.load(open(os.path.join(
        PHI, f"{bb}_hard_seed{seed}_{split}.pkl"), "rb"))
    phi = np.asarray(d["phi"], np.float64)
    gts = np.asarray(d["gts"]).astype(int)
    snrs = np.asarray(d["snrs"]).astype(float)
    keep = np.isin(gts, dig_idx)
    remap = {g: i for i, g in enumerate(dig_idx)}
    y = np.array([remap[g] for g in gts[keep]])
    return phi[keep], y, snrs[keep]


def per_bin_probe(vphi, vy, vr, tphi, tr, max_iter=3000):
    from sklearn.linear_model import LogisticRegression
    mu = vphi.mean(0, keepdims=True)
    sd = vphi.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    vphi = (vphi - mu) / sd; tphi = (tphi - mu) / sd

    def fit(X, y):
        clf = LogisticRegression(max_iter=max_iter, C=1.0)
        clf.fit(X, y); return clf

    glob = fit(vphi, vy)
    pred = np.empty(tphi.shape[0], dtype=int)
    for b in np.unique(vr):
        vm, tm = vr == b, tr == b
        if not tm.any():
            continue
        clf = fit(vphi[vm], vy[vm]) if (vm.sum() >= 50
                                        and np.unique(vy[vm]).size > 1) else glob
        pred[tm] = clf.predict(tphi[tm])
    un = ~np.isin(tr, np.unique(vr))
    if un.any():
        pred[un] = glob.predict(tphi[un])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="decomp_synA.csv")
    ap.add_argument("--ceiling", default="configs/snr_ladder/results/ceiling_final.csv")
    a = ap.parse_args()

    import csv as _csv
    ceil = {int(float(r["snr"])): float(r["bayes_acc"]) * 100
            for r in _csv.DictReader(open(a.ceiling))}
    mods, dig_idx = class_order()
    print(f"class order {mods}; digital idx {dig_idx}", flush=True)

    rows = []
    for bb in BACKBONES:
        for seed in SEEDS:
            run = os.path.join(RUNS, f"{bb}_hard-ce", f"seed_{seed}")
            vZ_full, vy_full, vr_full = load_pred(
                os.path.join(run, "predictions", "validation.pkl"))
            tZ_full, ty_full, tr_full = load_pred(
                os.path.join(run, "predictions", "test.pkl"))
            vZ, vy, vr = restrict(vZ_full, vy_full, vr_full, dig_idx)
            tZ, ty, tr = restrict(tZ_full, ty_full, tr_full, dig_idx)
            hp = CA.sm(tZ)
            ro = CA.per_bin("aff", vZ, vy, vr, tZ, tr)
            pv, yv, rv = load_phi(bb, seed, "validation", dig_idx)
            pt, yt, rt = load_phi(bb, seed, "test", dig_idx)
            assert len(yt) == len(ty), (len(yt), len(ty))
            pred = per_bin_probe(pv, yv, rv, pt, rt)
            for b in sorted(np.unique(tr)):
                m = tr == b
                mb = rt == b
                acc_h = CA.acc(hp[m], ty[m])
                acc_r = CA.acc(ro[m], ty[m])
                acc_p = float((pred[mb] == yt[mb]).mean() * 100)
                star = ceil[int(b)]
                rows.append(dict(
                    backbone=bb, seed=seed, snr=int(b), n=int(m.sum()),
                    bayes=round(star, 2), probe=round(acc_p, 2),
                    readout=round(acc_r, 2), hard=round(acc_h, 2),
                    T1_plus_T2=round(star - acc_p, 2),
                    T3_probe=round(acc_p - acc_h, 2),
                    T3_readout=round(acc_r - acc_h, 2)))
            print(f"{bb}/{seed} done", flush=True)

    with open(a.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)", flush=True)

    # compact band summary: bins where the median hard curve is in its 20-80% range
    import collections
    by_bb = collections.defaultdict(list)
    for r in rows:
        by_bb[r["backbone"]].append(r)
    print("\nband summary (seed-mean over hard-curve 20-80% band):", flush=True)
    for bb, rs in by_bb.items():
        snrs = sorted({r["snr"] for r in rs})
        hard_curve = {s: np.mean([r["hard"] for r in rs if r["snr"] == s])
                      for s in snrs}
        vals = np.array([hard_curve[s] for s in snrs])
        lo, hi = vals.min(), vals.max()
        band = [s for s in snrs
                if lo + .2 * (hi - lo) <= hard_curve[s] <= lo + .8 * (hi - lo)]
        sel = [r for r in rs if r["snr"] in band]
        f = lambda k: np.mean([r[k] for r in sel])
        print(f"  {bb:9s} band={band}  Bayes {f('bayes'):5.1f}  probe {f('probe'):5.1f}"
              f"  readout {f('readout'):5.1f}  hard {f('hard'):5.1f}"
              f"  | T1+T2 {f('T1_plus_T2'):+5.1f}  T3_probe {f('T3_probe'):+5.1f}"
              f"  T3_readout {f('T3_readout'):+5.1f}", flush=True)


if __name__ == "__main__":
    main()
