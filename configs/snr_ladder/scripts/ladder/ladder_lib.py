#!/usr/bin/env python3
"""Comprehensive null-ladder audit across architectures x datasets (server-side).

Auto-discovers every well-trained, matched (hard, reliability-aware-method) pair
on the same dataset split, and runs the ladder (F_shift < F_VS < F_aff) on each.
For each pair reports, in the transition band: whether the method helps over hard,
and whether the per-bin affine READOUT of the frozen hard model reproduces/exceeds
it (decision-layer) -- the paper's claim, now stress-tested across many backbones
and radio datasets. Reads prediction pkls in place; no retraining. seed_2026.
"""
from __future__ import annotations
import pickle, glob
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

W = "work_dirs"
EPS = 1e-12


def alive(p):
    try:
        d = pickle.load(open(p, "rb")); pps = np.asarray(d["pps"]); y = np.asarray(d["gts"])
        return round((pps.argmax(1) == y).mean()*100, 1), pps.shape[1]
    except Exception:
        return None, None

def backbone(run):
    for t in ["_hard", "_rcps", "_dpc", "_static", "_entproj", "_paperlike", "_confidence"]:
        if t in run: return run.split(t)[0]
    return run.split("_")[0]

def load(p):
    d = pickle.load(open(p, "rb"))
    return (np.log(np.clip(np.asarray(d["pps"], float), EPS, 1)),
            np.asarray(d["gts"]).astype(int), np.asarray(d["snrs"]).astype(float))

def sm(z): z = z-z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)
def acc(p, y): return float((p.argmax(1) == y).mean()*100)
def nll(p, y): return float(-np.log(np.clip(p[np.arange(y.size), y], EPS, 1)).mean())

def fit_bias(Z, y, l2=1e-3, mi=80):
    K = Z.shape[1]
    def o(b): b = b-b.mean(); return nll(sm(Z+b[None]), y)+l2*np.mean(b**2)
    r = minimize(o, np.zeros(K), method="L-BFGS-B", options={"maxiter": mi}).x; return r-r.mean()
def fit_vs(Z, y, l2=1e-3, mi=150):
    K = Z.shape[1]
    def o(t):
        a, c = t[:K], t[K:]; return nll(sm(Z*a[None]+c[None]), y)+l2*(np.mean((a-1)**2)+np.mean(c**2))
    r = minimize(o, np.concatenate([np.ones(K), np.zeros(K)]), method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K], r[K:]
def fit_aff(Z, y, l2=1e-2, mi=250):
    K = Z.shape[1]
    def o(t):
        Wm = t[:K*K].reshape(K, K); b = t[K*K:]
        return nll(sm(Z@Wm.T+b[None]), y)+l2*(np.mean((Wm-np.eye(K))**2)+np.mean(b**2))
    r = minimize(o, np.concatenate([np.eye(K).ravel(), np.zeros(K)]), method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K*K].reshape(K, K), r[K*K:]
AP = {"shift": lambda Z, p: sm(Z+p[None]), "vs": lambda Z, p: sm(Z*p[0][None]+p[1][None]),
      "aff": lambda Z, p: sm(Z@p[0].T+p[1][None])}
FT = {"shift": fit_bias, "vs": fit_vs, "aff": fit_aff}

def band_mask(hp, y, r):
    bins = np.unique(r); a = np.array([acc(hp[r == b], y[r == b]) for b in bins]); s = a.max()-a.min()
    if s < 1e-9: return np.ones(y.size, bool)
    keep = bins[(a >= a.min()+.2*s) & (a <= a.min()+.8*s)]
    if keep.size == 0: keep = bins[np.argsort(a)[len(bins)//2:len(bins)//2+1]]
    return np.isin(r, keep)

def per_bin(rung, vZ, vY, vR, tZ, tR):
    g = FT[rung](vZ, vY); out = np.empty_like(tZ)
    for b in np.unique(vR):
        vm = vR == b; p = FT[rung](vZ[vm], vY[vm]) if vm.sum() >= 50 else g
        m = tR == b
        if m.any(): out[m] = AP[rung](tZ[m], p)
    un = ~np.isin(tR, np.unique(vR))
    if un.any(): out[un] = AP[rung](tZ[un], g)
    return out


def discover():
    hard, meth = defaultdict(list), defaultdict(list)
    for p in glob.glob(W+"/*/amc/*/*/seed_2026/predictions/test.pkl"):
        a, K = alive(p)
        if a is None or a < 200/K: continue
        run = p.replace(os.sep, "/").split("/work_dirs/")[1].split("/")[3]
        key = (backbone(run), p.replace(os.sep, "/").split("/work_dirs/")[1].split("/")[2])
        if "hard-ce" in run or "hard_ce" in run: hard[key].append((a, p))
        elif any(t in run for t in ["rcps", "dpc", "entproj"]): meth[key].append((a, p))
    pairs = []
    for key in sorted(set(hard) & set(meth)):
        hp = max(hard[key])[1]; mp = max(meth[key])[1]
        pairs.append((key[0], key[1], hp, mp))
    return pairs


def main():
    pairs = discover()
    print(f"Comprehensive ladder audit: {len(pairs)} matched pairs (seed_2026, band)\n")
    print(f"{'backbone':12s} {'dataset':18s} {'K':>3s} | {'hard':>5s} {'meth':>5s} {'readout':>7s} | "
          f"{'m-const':>7s} {'m-VS':>6s} {'m-aff':>6s} | verdict")
    print("-"*104)
    dec = 0
    for bb, ds, hpath, mpath in pairs:
        try:
            vZ, vY, vR = load(hpath.replace("test.pkl", "validation.pkl"))
            tZ, tY, tR = load(hpath)
            mP = sm(load(mpath)[0])
            if mP.shape[0] != tY.size:
                print(f"{bb:12s} {ds:18s} skip (size mismatch)"); continue
            hp = sm(tZ); band = band_mask(hp, tY, tR)
            d = {}
            for rung in ("shift", "vs", "aff"):
                nP = per_bin(rung, vZ, vY, vR, tZ, tR)
                d[rung] = acc(mP[band], tY[band]) - acc(nP[band], tY[band])
                if rung == "aff": ro = acc(nP[band], tY[band])
            hb, mb = acc(hp[band], tY[band]), acc(mP[band], tY[band])
            v = "decision-layer" if d["aff"] <= 0 else "survives F_aff"
            if d["aff"] <= 0: dec += 1
            print(f"{bb:12s} {ds:18s} {tZ.shape[1]:>3d} | {hb:>5.1f} {mb:>5.1f} {ro:>7.1f} | "
                  f"{d['shift']:>+7.2f} {d['vs']:>+6.2f} {d['aff']:>+6.2f} | {v}")
        except Exception as e:
            print(f"{bb:12s} {ds:18s} ERROR {e}")
    print(f"\n{dec}/{len(pairs)} pairs: decision-layer (per-bin affine readout reproduces/exceeds the method).")


if __name__ == "__main__":
    main()
