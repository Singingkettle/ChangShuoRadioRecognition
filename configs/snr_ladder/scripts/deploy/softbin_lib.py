#!/usr/bin/env python3
"""Family E shared library: per-bin affine readout, hard/soft routing, band metrics.

Protocol is a strict extension of the paper's readout tools (scripts/ladder):
  - logits z = log(clip(pps, 1e-12, 1))
  - per-bin F_aff (W_b, beta_b) fit on VALIDATION at the TRUE bin, L2 to identity,
    L-BFGS-B mi=300 (identical fit_affine as the paper tools)
  - transition band fixed on hard test probs at TRUE SNR (20-80% rule of audit.py)
  - hard router: estimate coarsened to the 2 dB grid, clipped to grid range
  - soft router: P(y|x, shat) = sum_b p(b|shat) softmax(z W_b^T + beta_b) with
    p(b|shat) a Gaussian kernel over bin centres (uniform prior), sigma_r given.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12
CAMP = Path(__file__).resolve().parent


# ------------------------------- io / metrics ------------------------------ #
def load_pred(p):
    d = pickle.load(open(p, "rb"))
    Z = np.log(np.clip(np.asarray(d["pps"], float), EPS, 1))
    return Z, np.asarray(d["gts"]).astype(int), np.asarray(d["snrs"]).astype(float)


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def acc(p, y):
    return float((p.argmax(1) == y).mean() * 100)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(y.size), y], EPS, 1)).mean())


# ------------------------------- fits (paper-identical) -------------------- #
def fit_affine(Z, y, l2=1e-2, mi=300):
    K = Z.shape[1]

    def o(t):
        W = t[:K * K].reshape(K, K)
        b = t[K * K:]
        return nll(sm(Z @ W.T + b[None]), y) + l2 * (np.mean((W - np.eye(K)) ** 2) + np.mean(b ** 2))

    r = minimize(o, np.concatenate([np.eye(K).ravel(), np.zeros(K)]),
                 method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K * K].reshape(K, K), r[K * K:]


def fit_bias(Z, y, l2=1e-3, mi=100):
    K = Z.shape[1]

    def o(b):
        b = b - b.mean()
        return nll(sm(Z + b[None]), y) + l2 * np.mean(b ** 2)

    r = minimize(o, np.zeros(K), method="L-BFGS-B", options={"maxiter": mi}).x
    return r - r.mean()


# ------------------------------- band -------------------------------------- #
def band_mask(hp, y, r):
    bins = np.unique(r)
    a = np.array([acc(hp[r == b], y[r == b]) for b in bins])
    s = a.max() - a.min()
    if s < 1e-9:
        return np.ones(y.size, bool)
    lo, hi = a.min() + .2 * s, a.min() + .8 * s
    keep = bins[(a >= lo) & (a <= hi)]
    if keep.size == 0:
        keep = bins[np.argsort(a)[len(bins) // 2:len(bins) // 2 + 1]]
    return np.isin(r, keep)


def coarsen(snr, width=2.0):
    return np.round(snr / width) * width


# ------------------------------- affine table (cached) --------------------- #
def get_affine_table(cell, seed, val_pkl, min_bin=50, bin_width=2.0):
    """Fit (or load cached) per-bin affine table + global bias for one cell/seed."""
    cache = CAMP / f"affine_table_{cell}_seed{seed}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    vZ, vY, vR0 = load_pred(val_pkl)
    vR = coarsen(vR0, bin_width)
    grid = np.unique(vR)
    counts = {float(b): int((vR == b).sum()) for b in grid}
    assert all(c >= min_bin for c in counts.values()), f"bin under min_bin: {counts}"
    aff = {}
    for b in grid:
        m = vR == b
        aff[float(b)] = fit_affine(vZ[m], vY[m])
        print(f"  [{cell} seed{seed}] fit bin {b:+.0f} dB (n={m.sum()})", flush=True)
    gb = fit_bias(vZ, vY)
    tab = {"grid": grid.astype(float), "aff": aff, "global_bias": gb,
           "bin_width": bin_width, "counts": counts}
    pickle.dump(tab, open(cache, "wb"))
    return tab


# ------------------------------- routing ----------------------------------- #
def perbin_probs(Z, tab):
    """Precompute softmax(Z W_b^T + beta_b) for every bin b. Returns dict b->probs."""
    out = {}
    for b, (W, bb) in tab["aff"].items():
        out[b] = sm(Z @ W.T + bb[None]).astype(np.float32)
    return out

def route_hard(Pb, tab, shat):
    """Hard routing: quantize shat to grid (clip to range), pick that bin's probs."""
    grid = tab["grid"]
    est = np.clip(coarsen(shat, tab["bin_width"]), grid.min(), grid.max())
    P = np.empty_like(next(iter(Pb.values())))
    for b in grid:
        m = est == b
        if m.any():
            P[m] = Pb[float(b)][m]
    return P


def route_soft(Pb, tab, shat, sigma_r, trunc=1e-4):
    """Soft-bin marginalised readout with Gaussian kernel p(b|shat), uniform prior."""
    grid = tab["grid"]
    if sigma_r <= 0:
        return route_hard(Pb, tab, shat)
    W = np.exp(-((shat[:, None] - grid[None, :]) ** 2) / (2.0 * sigma_r ** 2))
    W /= W.sum(1, keepdims=True)
    W[W < trunc] = 0.0
    W /= W.sum(1, keepdims=True)
    P = np.zeros_like(next(iter(Pb.values())))
    for i, b in enumerate(grid):
        w = W[:, i]
        nz = w > 0
        if nz.any():
            P[nz] += w[nz, None] * Pb[float(b)][nz]
    return P


# ------------------------------- isotonic (PAV) ---------------------------- #
def isotonic_fit(x, y):
    """Monotone non-decreasing fit y ~ f(x) by pool-adjacent-violators.
    Returns (xs, fs) usable with np.interp."""
    o = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[o], np.asarray(y, float)[o]
    # collapse duplicate x by mean (weights)
    ux, inv = np.unique(xs, return_inverse=True)
    wsum = np.bincount(inv)
    ymean = np.bincount(inv, weights=ys) / wsum
    # PAV
    blocks_v, blocks_w, blocks_n = [], [], []
    for vi, wi in zip(ymean, wsum.astype(float)):
        blocks_v.append(vi)
        blocks_w.append(wi)
        blocks_n.append(1)
        while len(blocks_v) > 1 and blocks_v[-2] > blocks_v[-1]:
            v2, w2, n2 = blocks_v.pop(), blocks_w.pop(), blocks_n.pop()
            v1, w1, n1 = blocks_v.pop(), blocks_w.pop(), blocks_n.pop()
            blocks_v.append((v1 * w1 + v2 * w2) / (w1 + w2))
            blocks_w.append(w1 + w2)
            blocks_n.append(n1 + n2)
    fit = np.repeat(blocks_v, blocks_n)
    return ux, fit


def isotonic_apply(xs, fs, xq):
    return np.interp(xq, xs, fs)
