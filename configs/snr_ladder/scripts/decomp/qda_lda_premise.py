#!/usr/bin/env python3
"""Pre-registered premise check for Proposition A (CEILING_DECOMP_PREREG.md):

    per-bin QDA - LDA probe gap on frozen features; > 1 pp on a cell demotes
    the exponential-family (shared within-bin dispersion) premise for that cell.

Both sides get their fair shot: LDA uses lsqr + Ledoit-Wolf shrinkage; QDA tunes
its regularizer on a held-out fifth of the validation bin, then refits on the
full validation bin. Evaluation is always on the test bin. Fitting subsamples
the validation bin to at most FIT_CAP frames (stratified) so 2018A stays cheap;
both estimators see the same subsample.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

WORK = "work_dirs"
OUT = "work_dirs/qda_lda_premise.csv"
SYNA_META = "data/synthetic_awgn_amc_v1/test.json"
DIGITAL = ["8PSK", "BPSK", "CPFSK", "4PAM", "16QAM", "64QAM", "QPSK"]
REG_GRID = [0.05, 0.15, 0.3, 0.6, 0.9]
FIT_CAP = 30000
MIN_PER_CLASS = 8
RNG = np.random.RandomState(20260816)


def load(path):
    d = pickle.load(open(path, "rb"))
    return (np.asarray(d["phi"], np.float64), np.asarray(d["gts"]).astype(int),
            np.asarray(d["snrs"]).astype(int))


def stratified_cap(y, cap):
    if len(y) <= cap:
        return np.arange(len(y))
    keep = []
    frac = cap / len(y)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        RNG.shuffle(idx)
        keep.append(idx[:max(MIN_PER_CLASS, int(round(len(idx) * frac)))])
    return np.concatenate(keep)


def fit_eval_bin(Xv, yv, Xt, yt):
    """Return (lda_acc, qda_acc, chosen_reg) on this bin, or None to skip."""
    classes, counts = np.unique(yv, return_counts=True)
    if len(classes) < 2 or counts.min() < MIN_PER_CLASS:
        return None
    if not np.isin(yt, classes).all():
        mask = np.isin(yt, classes)
        Xt, yt = Xt[mask], yt[mask]
        if not len(yt):
            return None
    keep = stratified_cap(yv, FIT_CAP)
    Xv, yv = Xv[keep], yv[keep]

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(Xv, yv)
    lda_acc = float(np.mean(lda.predict(Xt) == yt))

    # 80/20 selection split, stratified by taking every 5th index per class
    sel = np.zeros(len(yv), bool)
    for c in np.unique(yv):
        idx = np.flatnonzero(yv == c)
        sel[idx[::5]] = True
    best, best_reg = -1.0, REG_GRID[0]
    for reg in REG_GRID:
        q = QuadraticDiscriminantAnalysis(reg_param=reg)
        try:
            q.fit(Xv[~sel], yv[~sel])
            acc = float(np.mean(q.predict(Xv[sel]) == yv[sel]))
        except Exception:
            continue
        if acc > best:
            best, best_reg = acc, reg
    q = QuadraticDiscriminantAnalysis(reg_param=best_reg)
    q.fit(Xv, yv)
    qda_acc = float(np.mean(q.predict(Xt) == yt))
    return lda_acc, qda_acc, best_reg


def digital_index():
    mods = json.load(open(SYNA_META))["metainfo"]["modulations"]
    return np.array([mods.index(m) for m in DIGITAL])


def cells():
    out = []
    dig = digital_index()
    for f in sorted(glob.glob(f"{WORK}/repr_phi_synA/*_validation.pkl")):
        bb = os.path.basename(f).split("_hard_")[0]
        seed = f.split("seed")[1][:4]
        out.append((f"synA/{bb}", seed, f, f.replace("_validation", "_test"), dig))
    for tag, sub in [("10B/mcformer", "repr_phi_mcformer10b"),
                     ("2018A/petcgdnn", "repr_phi_2018A")]:
        for f in sorted(glob.glob(f"{WORK}/{sub}/hard_seed*_validation.pkl")):
            seed = f.split("seed")[1][:4]
            out.append((tag, seed, f, f.replace("_validation", "_test"), None))
    for f in sorted(glob.glob(f"{WORK}/repr_phi_rml22/*_validation.pkl")):
        bb = os.path.basename(f).split("_hard_")[0]
        seed = f.split("seed")[1][:4]
        out.append((f"RML22/{bb}", seed, f, f.replace("_validation", "_test"), None))
    return out


def main():
    rows = ["cell,seed,snr,n_fit,n_test,lda,qda,reg,diff_pp"]
    for cell, seed, fv, ft, dig in cells():
        Xv, yv, sv = load(fv)
        Xt, yt, st = load(ft)
        if dig is not None:  # restrict to the ceiling's 7-class label space
            relab = -np.ones(int(max(yv.max(), yt.max())) + 1, int)
            relab[dig] = np.arange(len(dig))
            mv, mt = relab[yv] >= 0, relab[yt] >= 0
            Xv, yv, sv = Xv[mv], relab[yv[mv]], sv[mv]
            Xt, yt, st = Xt[mt], relab[yt[mt]], st[mt]
        for snr in sorted(set(sv.tolist()) & set(st.tolist())):
            bv, bt = sv == snr, st == snr
            r = fit_eval_bin(Xv[bv], yv[bv], Xt[bt], yt[bt])
            if r is None:
                continue
            lda, qda, reg = r
            rows.append(f"{cell},{seed},{snr},{bv.sum()},{bt.sum()},"
                        f"{lda:.4f},{qda:.4f},{reg},{100*(qda-lda):.2f}")
            print(rows[-1], flush=True)
        with open(OUT, "w") as fh:
            fh.write("\n".join(rows) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
