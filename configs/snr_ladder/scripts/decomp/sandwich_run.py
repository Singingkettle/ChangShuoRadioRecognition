#!/usr/bin/env python3
"""Family B: real-data Bayes sandwich per SNR bin.

E1  = best single-architecture per-bin affine readout (objective/l2/maxiter
      identical to ../ladder/ladder_lib.py per_bin("aff", ...)).
E2  = cross-architecture stacked ensemble: concatenated log-prob logits +
      per-bin multinomial logistic (fit on validation, evaluated on test).
CH  = Cover-Hart bounds on the strongest backbone's phi (test split, LOO
      within each SNR bin, Euclidean distance, GPU chunked cdist):
      CH_lower = 1 - R15 (15-NN plug-in), CH_upper = A_CH (1-NN inversion).

Read-only on work_dirs. Temp/output under work_dirs/sandwich/.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np
from scipy.optimize import minimize

EPS = 1e-12
W = "work_dirs"
P1 = "work_dirs/sandwich"

# ---------------------------------------------------------------- shared utils
def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def acc(p, y):
    return float((p.argmax(1) == y).mean() * 100)

def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(y.size), y], EPS, 1)).mean())

def load(p):
    d = pickle.load(open(p, "rb"))
    return (np.log(np.clip(np.asarray(d["pps"], float), EPS, 1)),
            np.asarray(d["gts"]).astype(int), np.asarray(d["snrs"]).astype(float))

# ------------------------------------------------- E1: audit-identical affine
def fit_aff_ref(Z, y, l2=1e-2, mi=250):
    """Exact copy of ladder_lib.fit_aff (numeric jac)."""
    K = Z.shape[1]
    def o(t):
        Wm = t[:K*K].reshape(K, K); b = t[K*K:]
        return nll(sm(Z @ Wm.T + b[None]), y) + l2*(np.mean((Wm-np.eye(K))**2) + np.mean(b**2))
    r = minimize(o, np.concatenate([np.eye(K).ravel(), np.zeros(K)]),
                 method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K*K].reshape(K, K), r[K*K:]

def fit_aff_fast(Z, y, l2=1e-2, mi=250):
    """Same convex objective as fit_aff_ref, analytic gradient (speed only)."""
    n, K = Z.shape[0], Z.shape[1]
    Y = np.zeros((n, K)); Y[np.arange(n), y] = 1.0
    I = np.eye(K)
    def o(t):
        Wm = t[:K*K].reshape(K, K); b = t[K*K:]
        L = Z @ Wm.T + b[None]
        L = L - L.max(1, keepdims=True)
        e = np.exp(L); P = e / e.sum(1, keepdims=True)
        f = nll(P, y) + l2*(np.mean((Wm-I)**2) + np.mean(b**2))
        G = (P - Y) / n
        gW = G.T @ Z + 2.0*l2*(Wm - I)/(K*K)
        gb = G.sum(0) + 2.0*l2*b/K
        return f, np.concatenate([gW.ravel(), gb])
    r = minimize(o, np.concatenate([I.ravel(), np.zeros(K)]), jac=True,
                 method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K*K].reshape(K, K), r[K*K:]

def aff_apply(Z, p):
    return sm(Z @ p[0].T + p[1][None])

def per_bin_aff(vZ, vY, vR, tZ, tR, fitter):
    """Identical control flow to ladder_lib.per_bin (rung='aff')."""
    g = fitter(vZ, vY); out = np.empty_like(tZ)
    for b in np.unique(vR):
        vm = vR == b
        p = fitter(vZ[vm], vY[vm]) if vm.sum() >= 50 else g
        m = tR == b
        if m.any(): out[m] = aff_apply(tZ[m], p)
    un = ~np.isin(tR, np.unique(vR))
    if un.any(): out[un] = aff_apply(tZ[un], g)
    return out

def band_mask_bins(hp, y, r):
    """ladder_lib.band_mask, returning the kept bin values."""
    bins = np.unique(r)
    a = np.array([acc(hp[r == b], y[r == b]) for b in bins])
    s = a.max() - a.min()
    if s < 1e-9: return bins
    keep = bins[(a >= a.min() + .2*s) & (a <= a.min() + .8*s)]
    if keep.size == 0: keep = bins[np.argsort(a)[len(bins)//2:len(bins)//2+1]]
    return keep

# ------------------------------------------------------- E2: stacked ensemble
def fit_stack(Zc, y, K, A, l2=1e-2, mi=400):
    """Multinomial logistic on concatenated logits; init = logit averaging
    (W0 = [I|...|I]/A), L2 pulls toward W0. Convex; analytic gradient."""
    n, D = Zc.shape
    Y = np.zeros((n, K)); Y[np.arange(n), y] = 1.0
    W0 = np.concatenate([np.eye(K)]*A, axis=1) / A       # (K, A*K)
    def o(t):
        Wm = t[:K*D].reshape(K, D); b = t[K*D:]
        L = Zc @ Wm.T + b[None]
        L = L - L.max(1, keepdims=True)
        e = np.exp(L); P = e / e.sum(1, keepdims=True)
        f = nll(P, y) + l2*(np.mean((Wm-W0)**2) + np.mean(b**2))
        G = (P - Y) / n
        gW = G.T @ Zc + 2.0*l2*(Wm - W0)/(K*D)
        gb = G.sum(0) + 2.0*l2*b/K
        return f, np.concatenate([gW.ravel(), gb])
    r = minimize(o, np.concatenate([W0.ravel(), np.zeros(K)]), jac=True,
                 method="L-BFGS-B", options={"maxiter": mi}).x
    return r[:K*D].reshape(K, D), r[K*D:]

def fit_stack_cv(vZb, vYb, K, A, grid=(1e-2, 1e-1, 1.0, 10.0, 100.0), folds=4, seed=0):
    """Per-bin l2 chosen by K-fold CV inside the validation bin (never sees test).
    l2->inf recovers plain logit averaging (W0), so averaging is in the family."""
    n = vZb.shape[0]
    if n < 200:
        best = 10.0
    else:
        rng = np.random.RandomState(seed)
        fold_id = rng.permutation(n) % folds
        cv = {}
        for l2 in grid:
            hit = 0
            for f in range(folds):
                tr = fold_id != f
                Wb, bb = fit_stack(vZb[tr], vYb[tr], K, A, l2=l2)
                P = sm(vZb[~tr] @ Wb.T + bb[None])
                hit += int((P.argmax(1) == vYb[~tr]).sum())
            cv[l2] = hit / n
        best = max(grid, key=lambda l: (cv[l], l))   # tie -> stronger reg
    return fit_stack(vZb, vYb, K, A, l2=best), best

def per_bin_stack(vZc, vY, vR, tZc, tR, K, A):
    g = fit_stack(vZc, vY, K, A)     # global fallback: n large, overfit negligible
    out = np.empty((tZc.shape[0], K)); l2s = {}
    for b in np.unique(vR):
        vm = vR == b
        if vm.sum() >= 50:
            p, l2b = fit_stack_cv(vZc[vm], vY[vm], K, A); l2s[float(b)] = l2b
        else:
            p = g; l2s[float(b)] = "global"
        m = tR == b
        if m.any(): out[m] = sm(tZc[m] @ p[0].T + p[1][None])
    un = ~np.isin(tR, np.unique(vR))
    if un.any(): out[un] = sm(tZc[un] @ g[0].T + g[1][None])
    print("  E2 per-bin l2:", {k: l2s[k] for k in sorted(l2s)}, flush=True)
    return out

# ------------------------------------------------------------------ CH-on-phi
def ch_upper_from_r1(r1, M):
    """Invert Cover-Hart: R1 <= R*(2 - M/(M-1) R*)  =>  Bayes-accuracy upper bound."""
    disc = max(0.0, 1.0 - (M / (M - 1.0)) * min(r1, (M - 1.0) / M))
    r_star_lb = ((M - 1.0) / M) * (1.0 - np.sqrt(disc))
    return 1.0 - r_star_lb

def knn_loo_bin(X, y, k, device, chunk=8192):
    import torch
    n = X.shape[0]
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    kk = min(k + 1, n)
    err1 = 0; errk = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(Xt[s:e], Xt)               # (m, n)
        rows = torch.arange(s, e, device=device)
        d[torch.arange(e - s, device=device), rows] = float("inf")  # LOO
        idx = d.topk(kk - 1, largest=False).indices  # k neighbours, self removed
        lab = yt[idx]                                # (m, k)
        err1 += int((lab[:, 0] != yt[s:e]).sum())
        votes = torch.mode(lab[:, :min(k, kk - 1)], dim=1).values
        errk += int((votes != yt[s:e]).sum())
    return err1 / n, errk / n

# ------------------------------------------------------------------- commands
def cmd_preds(args):
    cfg = json.load(open(args.config))
    ds = cfg["dataset"]; runs = cfg["archs"]  # {name: predictions_dir}
    names = list(runs)
    data = {}
    for a in names:
        vd = os.path.join(runs[a], "validation.pkl")
        td = os.path.join(runs[a], "test.pkl")
        data[a] = dict(zip(["vZ", "vY", "vR"], load(vd))) | dict(zip(["tZ", "tY", "tR"], load(td)))
    # row-wise (gts, snrs) consistency. Group archs by (val,test) row signature;
    # the largest group is the E2 stack. E1 additionally admits any arch whose
    # TEST rows match the stack's test rows (its own val is self-consistent for
    # fitting the readout; only stacking needs val alignment).
    def sig(d, keys):
        import hashlib
        h = hashlib.sha1()
        for k in keys: h.update(np.ascontiguousarray(d[k]).tobytes())
        return h.hexdigest()
    from collections import Counter
    full_sig = {a: sig(data[a], ["vY", "vR", "tY", "tR"]) for a in names}
    grp = Counter(full_sig.values()).most_common(1)[0][0]
    kept = [a for a in names if full_sig[a] == grp]              # E2 stack
    ref = data[kept[0]]
    tsig_ref = sig(ref, ["tY", "tR"])
    kept_e1 = [a for a in names if sig(data[a], ["tY", "tR"]) == tsig_ref]
    skipped = [a for a in names if a not in kept_e1]
    e1_only = [a for a in kept_e1 if a not in kept]
    if e1_only:
        print(f"[{ds}] E1-only (val rows mismatch stack; excluded from E2): {e1_only}", flush=True)
    if skipped:
        print(f"[{ds}] SKIPPED entirely (test gts/snrs mismatch): {skipped}", flush=True)
    if len(kept) < 2:
        print(f"[{ds}] <2 stackable archs; E2 not computed", flush=True)
    tY, tR = ref["tY"], ref["tR"]; vY, vR = ref["vY"], ref["vR"]
    bins = np.unique(tR); K = ref["tZ"].shape[1]

    # per-arch raw acc and E1 readout (each arch fits on its OWN validation)
    fitter = fit_aff_ref if args.ref_fitter else fit_aff_fast
    e1 = {a: {} for a in kept_e1}; raw = {}
    for a in kept_e1:
        d = data[a]
        t0 = time.time()
        out = per_bin_aff(d["vZ"], d["vY"], d["vR"], d["tZ"], tR, fitter)
        raw[a] = acc(sm(d["tZ"]), tY)
        for b in bins:
            m = tR == b
            e1[a][float(b)] = acc(out[m], tY[m])
        print(f"[{ds}] E1 {a}: raw={raw[a]:.2f} readout done in {time.time()-t0:.0f}s", flush=True)
    best_arch = max(kept_e1, key=lambda a: raw[a])

    # band from best single arch's raw hard test probabilities (audit rule)
    band = band_mask_bins(sm(data[best_arch]["tZ"]), tY, tR)

    # E2 stacked
    e2 = {}
    if len(kept) >= 2:
        A = len(kept)
        vZc = np.concatenate([data[a]["vZ"] for a in kept], axis=1)
        tZc = np.concatenate([data[a]["tZ"] for a in kept], axis=1)
        t0 = time.time()
        out2 = per_bin_stack(vZc, vY, vR, tZc, tR, K, A)
        for b in bins:
            m = tR == b
            e2[float(b)] = acc(out2[m], tY[m])
        print(f"[{ds}] E2 stacked ({A} archs) done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for b in bins:
        m = tR == b
        e1b = {a: e1[a][float(b)] for a in kept_e1}
        ab = max(e1b, key=e1b.get)
        rows.append({"dataset": ds, "bin": float(b), "n_test": int(m.sum()),
                     "E1_best_single_readout": round(e1b[ab], 3), "e1_arch": ab,
                     "E2_stacked": round(e2[float(b)], 3) if e2 else "",
                     "in_band": int(b in band)})
    out_csv = os.path.join(P1, f"{ds}_e1e2.csv")
    with open(out_csv, "w") as f:
        f.write("dataset,bin,n_test,E1_best_single_readout,e1_arch,E2_stacked,in_band\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in ["dataset", "bin", "n_test",
                    "E1_best_single_readout", "e1_arch", "E2_stacked", "in_band"]) + "\n")
    meta = {"dataset": ds, "archs_kept": kept, "archs_e1": kept_e1,
            "archs_e1_only": e1_only, "archs_skipped": skipped,
            "raw_acc": {a: round(raw[a], 3) for a in kept_e1}, "best_arch": best_arch,
            "band_bins": [float(b) for b in band], "K": int(K),
            "n_val": int(vY.size), "n_test": int(tY.size),
            "fitter": "ref" if args.ref_fitter else "fast"}
    json.dump(meta, open(os.path.join(P1, f"{ds}_meta.json"), "w"), indent=1)
    print(f"[{ds}] wrote {out_csv}", flush=True)

def cmd_qa(args):
    """Numeric-vs-analytic fitter equivalence on one arch (per-bin readout accs)."""
    cfg = json.load(open(args.config)); ds = cfg["dataset"]
    a, pdir = list(cfg["archs"].items())[0]
    vZ, vY, vR = load(os.path.join(pdir, "validation.pkl"))
    tZ, tY, tR = load(os.path.join(pdir, "test.pkl"))
    o_ref = per_bin_aff(vZ, vY, vR, tZ, tR, fit_aff_ref)
    o_fast = per_bin_aff(vZ, vY, vR, tZ, tR, fit_aff_fast)
    print(f"QA {ds}/{a}: bin, acc_ref, acc_fast, diff(pp)")
    mx = 0.0
    for b in np.unique(tR):
        m = tR == b
        ar, af = acc(o_ref[m], tY[m]), acc(o_fast[m], tY[m])
        mx = max(mx, abs(ar - af))
        print(f"  {b:6.1f} {ar:7.3f} {af:7.3f} {ar-af:+7.3f}")
    print(f"QA max |diff| = {mx:.4f} pp")

def cmd_ch(args):
    import torch
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    d = pickle.load(open(args.phi, "rb"))
    X = np.ascontiguousarray(np.asarray(d["phi"], np.float32))
    y = np.asarray(d["gts"]).astype(np.int64)
    r = np.asarray(d["snrs"]).astype(float)
    pps = np.asarray(d["pps"]); M = pps.shape[1]
    print(f"[CH {args.dataset}] phi={X.shape} M={M} backbone_acc={acc(pps, y):.2f} "
          f"ckpt={str(d.get('checkpoint'))[:100]}", flush=True)
    rows = []
    for b in np.unique(r):
        m = r == b; n = int(m.sum())
        t0 = time.time()
        r1, r15 = knn_loo_bin(X[m], y[m], 15, dev)
        au = ch_upper_from_r1(r1, M)
        rows.append((float(b), n, r1, r15, 1.0 - r15, au))
        print(f"[CH {args.dataset}] bin {b:+.0f}: n={n} R1={r1:.4f} R15={r15:.4f} "
              f"lower={100*(1-r15):.2f} upper={100*au:.2f} ({time.time()-t0:.0f}s)", flush=True)
    out_csv = os.path.join(P1, f"{args.dataset}_ch.csv")
    with open(out_csv, "w") as f:
        f.write("dataset,bin,n_bin,R1,R15,CH_lower_1mR15,CH_upper,phi\n")
        for b, n, r1, r15, lo, up in rows:
            f.write(f"{args.dataset},{b},{n},{r1:.6f},{r15:.6f},{100*lo:.3f},{100*up:.3f},"
                    f"{os.path.basename(args.phi)}\n")
    print(f"[CH {args.dataset}] wrote {out_csv}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("preds"); p1.add_argument("--config", required=True)
    p1.add_argument("--ref-fitter", action="store_true")
    p2 = sub.add_parser("qa"); p2.add_argument("--config", required=True)
    p3 = sub.add_parser("ch"); p3.add_argument("--dataset", required=True)
    p3.add_argument("--phi", required=True); p3.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(P1, exist_ok=True)
    {"preds": cmd_preds, "qa": cmd_qa, "ch": cmd_ch}[args.cmd](args)
