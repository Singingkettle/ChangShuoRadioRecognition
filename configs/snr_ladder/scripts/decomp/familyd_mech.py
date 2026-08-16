#!/usr/bin/env python
"""Family D: decision-deficit mechanism statistics (pre-registered, frozen criteria).

Pure-CPU analysis. Read-only on data pkls; writes JSON per (cell, seed) into --out.

Per cell per seed, on the *validation* phi (test phi used for consistency re-check):
  band  : SNR bins whose hard accuracy (from val pps/gts/snrs) lies in
          [a_min + 0.2R, a_min + 0.8R], R = a_max - a_min   (paper band_mask rule)
  S_rot : adjacent-bin discriminant-subspace rotation, 1 - mean_i cos^2(theta_i),
          cell-level = median over adjacent pairs inside the band
  S_drift: whitened class-mean drift ratio (pooled two-bin LW covariance), band median
  S_cov : || W_b/tr - W_{b+1}/tr ||_F, band median
  delta_plugin: Gaussian plug-in decision deficit (per-bin LDA optimum vs single
          shared affine head fit on the band mixture), MC 20k pts/bin, band-weighted
          by real validation bin counts. Reported in percentage points.
  Delta(b): real-data per-bin linear-probe acc - global linear-probe acc
          (probes fit on val, evaluated on test; features standardized; LR max_iter=3000)
  rho_delta_srot: Spearman rho over ALL bins between Delta(b) and per-bin S_rot(b)
          (per-bin S_rot(b) = mean of the adjacent-pair values touching bin b);
          band-restricted rho also recorded as secondary.
"""
import argparse
import json
import os
import pickle
import time
import traceback
import zlib

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_MC = 20000  # MC points per bin (train draw; an independent 20k eval draw is used for unbiased accuracy)


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


def load_pkl(path):
    with open(os.path.expanduser(path), "rb") as f:
        d = pickle.load(f)
    return (
        np.asarray(d["phi"], dtype=np.float64),
        np.asarray(d["pps"], dtype=np.float64),
        np.asarray(d["gts"]).astype(int),
        np.asarray(d["snrs"]).astype(int),
    )


def band_from_hard(pps, gts, snrs):
    bins = np.unique(snrs)
    acc = np.array([(pps[snrs == b].argmax(1) == gts[snrs == b]).mean() for b in bins])
    amin, amax = acc.min(), acc.max()
    r = amax - amin
    mask = (acc >= amin + 0.2 * r) & (acc <= amin + 0.8 * r)
    return bins, acc, mask


def fit_bin(X, y, classes):
    """Class means + Ledoit-Wolf pooled within-class covariance for one SNR bin."""
    ks, mus, cnts, res = [], [], [], []
    for k in classes:
        Xi = X[y == k]
        if len(Xi) < 2:
            continue
        mu = Xi.mean(0)
        ks.append(k)
        mus.append(mu)
        cnts.append(len(Xi))
        res.append(Xi - mu)
    mus = np.asarray(mus)
    res = np.vstack(res)
    W = LedoitWolf(assume_centered=True).fit(res).covariance_
    return dict(ks=np.asarray(ks), mus=mus, cnts=np.asarray(cnts), W=W,
                mu_bar=mus.mean(0), res=res)


def inv_sqrt_psd(W):
    lam, V = np.linalg.eigh(W)
    lam = np.clip(lam, lam.max() * 1e-12, None)
    return (V / np.sqrt(lam)) @ V.T


def disc_basis(st):
    """Top-(K-1) left singular vectors of whitened class-mean matrix."""
    M = inv_sqrt_psd(st["W"]) @ (st["mus"] - st["mu_bar"]).T  # d x K
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    K = M.shape[1]
    return U[:, : max(K - 1, 1)]


def geometry_stats(X, y, snrs, bins, classes):
    """Per-bin fits + adjacent-pair S_rot / S_drift / S_cov arrays."""
    fits = []
    for b in bins:
        m = snrs == b
        fits.append(fit_bin(X[m], y[m], classes))
    bases = [disc_basis(st) for st in fits]

    s_rot, s_drift, s_cov = [], [], []
    for i in range(len(bins) - 1):
        a, c = fits[i], fits[i + 1]
        # S_rot
        th = subspace_angles(bases[i], bases[i + 1])
        s_rot.append(float(1.0 - np.mean(np.cos(th) ** 2)))
        # S_drift: pooled two-bin LW covariance
        Wp = LedoitWolf(assume_centered=True).fit(np.vstack([a["res"], c["res"]])).covariance_
        Wm = inv_sqrt_psd(Wp)
        shared = np.intersect1d(a["ks"], c["ks"])
        ia = np.searchsorted(a["ks"], shared)
        ic = np.searchsorted(c["ks"], shared)
        num = np.linalg.norm((c["mus"][ic] - a["mus"][ia]) @ Wm.T, axis=1).mean()
        den = np.linalg.norm((a["mus"] - a["mu_bar"]) @ Wm.T, axis=1).mean()
        s_drift.append(float(num / den))
        # S_cov
        A = a["W"] / np.trace(a["W"])
        C = c["W"] / np.trace(c["W"])
        s_cov.append(float(np.linalg.norm(A - C, "fro")))
    # free residuals
    for st in fits:
        st.pop("res", None)
    return fits, np.array(s_rot), np.array(s_drift), np.array(s_cov)


def pair_median_in_band(pair_vals, mask):
    """Median over adjacent pairs whose BOTH bins are in the band."""
    sel = [i for i in range(len(pair_vals)) if mask[i] and mask[i + 1]]
    if not sel:
        return float("nan"), sel
    return float(np.median(pair_vals[sel])), sel


def per_bin_from_pairs(pair_vals, n_bins):
    """S(b) = mean of adjacent-pair values touching bin b."""
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = []
        if b - 1 >= 0:
            vals.append(pair_vals[b - 1])
        if b < n_bins - 1:
            vals.append(pair_vals[b])
        out[b] = np.mean(vals)
    return out


def sample_mixture(st, n, rng):
    """Draw n points from the fitted per-bin Gaussian mixture (class priors = empirical)."""
    p = st["cnts"] / st["cnts"].sum()
    nk = rng.multinomial(n, p)
    L = np.linalg.cholesky(st["W"])
    Xs, ys = [], []
    for j in range(len(st["ks"])):
        if nk[j] == 0:
            continue
        Z = rng.standard_normal((nk[j], L.shape[0]))
        Xs.append(st["mus"][j] + Z @ L.T)
        ys.append(np.full(nk[j], st["ks"][j]))
    return np.vstack(Xs), np.concatenate(ys)


def lda_predict(st, X):
    """Bayes rule of the fitted per-bin Gaussian mixture (shared covariance -> LDA)."""
    A = np.linalg.solve(st["W"], st["mus"].T).T          # K x d
    pri = st["cnts"] / st["cnts"].sum()
    b0 = -0.5 * np.einsum("kd,kd->k", A, st["mus"]) + np.log(pri)
    return st["ks"][np.argmax(X @ A.T + b0, axis=1)]


def plugin_deficit(fits, bins, mask, n_val_per_bin, cell, seed):
    """Gaussian plug-in deficit: per-bin LDA optimum vs shared affine head on band mixture."""
    rng = np.random.default_rng(zlib.crc32(f"{cell}|{seed}|mc".encode()) & 0xFFFFFFFF)
    band_idx = [i for i in range(len(bins)) if mask[i]]
    tr_X, tr_y, ev = [], [], {}
    for i in band_idx:
        Xtr, ytr = sample_mixture(fits[i], N_MC, rng)
        Xev, yev = sample_mixture(fits[i], N_MC, rng)
        tr_X.append(Xtr)
        tr_y.append(ytr)
        ev[i] = (Xev, yev)
    tr_X = np.vstack(tr_X)
    tr_y = np.concatenate(tr_y)
    sc = StandardScaler().fit(tr_X)
    head = LogisticRegression(max_iter=3000).fit(sc.transform(tr_X), tr_y)

    per_bin = {}
    num = den = 0.0
    for i in band_idx:
        Xev, yev = ev[i]
        acc_lda = float((lda_predict(fits[i], Xev) == yev).mean())
        acc_sh = float((head.predict(sc.transform(Xev)) == yev).mean())
        per_bin[int(bins[i])] = dict(lda=acc_lda, shared=acc_sh)
        w = n_val_per_bin[i]
        num += w * (acc_lda - acc_sh)
        den += w
    return 100.0 * num / den, per_bin  # percentage points


def real_deficit(Xv, yv, sv, Xt, yt, stn, bins):
    """Delta(b) = per-bin probe acc - global probe acc; probes fit on val, eval on test."""
    sc = StandardScaler().fit(Xv)
    Xvs = sc.transform(Xv)
    Xts = sc.transform(Xt)
    gl = LogisticRegression(max_iter=3000).fit(Xvs, yv)
    pred_gl = gl.predict(Xts)
    acc_gl, acc_pb, delta = {}, {}, np.full(len(bins), np.nan)
    for i, b in enumerate(bins):
        mt = stn == b
        mv = sv == b
        ag = float((pred_gl[mt] == yt[mt]).mean())
        lr = LogisticRegression(max_iter=3000).fit(Xvs[mv], yv[mv])
        ap = float((lr.predict(Xts[mt]) == yt[mt]).mean())
        acc_gl[int(b)] = ag
        acc_pb[int(b)] = ap
        delta[i] = ap - ag
    return delta, acc_gl, acc_pb


def run_cell_seed(cell, seed, val_path, test_path, out_dir):
    out_json = os.path.join(out_dir, f"{cell}_seed{seed}.json")
    if os.path.exists(out_json):
        log(f"skip {cell} seed{seed} (exists)")
        return
    t0 = time.time()
    log(f"start {cell} seed{seed}")
    Xv, ppv, yv, sv = load_pkl(val_path)
    Xt, ppt, yt, stn = load_pkl(test_path)
    classes = np.unique(np.concatenate([yv, yt]))

    bins, acc_val, mask = band_from_hard(ppv, yv, sv)
    bins_t, acc_test, mask_t = band_from_hard(ppt, yt, stn)
    assert np.array_equal(bins, bins_t)
    n_val_per_bin = np.array([(sv == b).sum() for b in bins])
    log(f"  band(val) = {list(bins[mask])}")

    # --- geometry on validation (primary) ---
    fits_v, rot_v, drift_v, cov_v = geometry_stats(Xv, yv, sv, bins, classes)
    S_rot, band_pairs = pair_median_in_band(rot_v, mask)
    S_drift, _ = pair_median_in_band(drift_v, mask)
    S_cov, _ = pair_median_in_band(cov_v, mask)
    log(f"  val geometry done  S_rot={S_rot:.4f} S_drift={S_drift:.4f} S_cov={S_cov:.4f} ({time.time()-t0:.0f}s)")

    # --- geometry on test (consistency re-check; same val-derived band) ---
    fits_t, rot_t, drift_t, cov_t = geometry_stats(Xt, yt, stn, bins, classes)
    S_rot_t, _ = pair_median_in_band(rot_t, mask)
    S_drift_t, _ = pair_median_in_band(drift_t, mask)
    S_cov_t, _ = pair_median_in_band(cov_t, mask)
    del fits_t
    log(f"  test geometry done S_rot={S_rot_t:.4f} ({time.time()-t0:.0f}s)")

    # --- Gaussian plug-in deficit (validation fits only; no real test labels touched) ---
    d_plugin, plugin_bins = plugin_deficit(fits_v, bins, mask, n_val_per_bin, cell, seed)
    log(f"  plugin deficit = {d_plugin:.3f} pp ({time.time()-t0:.0f}s)")

    # --- real-data deficit curve (val fit -> test eval) ---
    delta_b, acc_gl, acc_pb = real_deficit(Xv, yv, sv, Xt, yt, stn, bins)
    srot_bin = per_bin_from_pairs(rot_v, len(bins))
    ok = ~np.isnan(delta_b) & ~np.isnan(srot_bin)
    rho_all, p_all = spearmanr(delta_b[ok], srot_bin[ok])
    bidx = np.where(mask)[0]
    if len(bidx) >= 3:
        rho_band, p_band = spearmanr(delta_b[bidx], srot_bin[bidx])
    else:
        rho_band, p_band = float("nan"), float("nan")
    log(f"  probes done rho_all={rho_all:.3f} rho_band={rho_band:.3f} ({time.time()-t0:.0f}s)")

    res = dict(
        cell=cell,
        seed=seed,
        bins=[int(b) for b in bins],
        acc_val=[float(a) for a in acc_val],
        acc_test=[float(a) for a in acc_test],
        band_bins=[int(b) for b in bins[mask]],
        band_bins_test_derived=[int(b) for b in bins[mask_t]],
        n_val_per_bin=[int(n) for n in n_val_per_bin],
        S_rot=S_rot, S_drift=S_drift, S_cov=S_cov,
        S_rot_test=S_rot_t, S_drift_test=S_drift_t, S_cov_test=S_cov_t,
        pair_S_rot_val=[float(x) for x in rot_v],
        pair_S_drift_val=[float(x) for x in drift_v],
        pair_S_cov_val=[float(x) for x in cov_v],
        pair_S_rot_test=[float(x) for x in rot_t],
        delta_plugin_pp=float(d_plugin),
        plugin_per_bin=plugin_bins,
        delta_b=[float(x) for x in delta_b],
        acc_global_test=acc_gl,
        acc_perbin_test=acc_pb,
        srot_per_bin=[float(x) for x in srot_bin],
        rho_delta_srot=float(rho_all), rho_p=float(p_all),
        rho_delta_srot_band=float(rho_band), rho_p_band=float(p_band),
        elapsed_s=time.time() - t0,
    )
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    log(f"done {cell} seed{seed} in {time.time()-t0:.0f}s -> {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cell", default=None, help="process only this cell")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.manifest) as f:
        mani = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    for cell, spec in mani["cells"].items():
        if args.cell and cell != args.cell:
            continue
        for seed in mani["seeds"]:
            try:
                run_cell_seed(cell, seed,
                              spec["val"].format(seed=seed),
                              spec["test"].format(seed=seed),
                              args.out)
            except Exception:
                log(f"ERROR {cell} seed{seed}:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
