#!/usr/bin/env python
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Analyze the per-box quality JSONL from `returniq_pipeline.py diag-quality`.

Tests the hypothesis: IoU (a VISION metric) predicts return-to-IQ recognition success POORLY,
while SIGNAL-fidelity metrics (cf-error, containment, coverage, energy coverage/contamination) predict
it well. Outputs:
  (a) recog_vs_quality.csv      -- recog success rate vs decile-binned quality, per metric
  (b) headline_iou_band.csv     -- within a fixed IoU band, containment(HIGH) vs offset(LOW) recog acc + z-test
  (c) predictive_power.csv      -- rank-AUC + univariate & multivariate logistic coeffs per metric
  (d) by_family_snr.csv         -- stratified by family group and SNR bucket
Plots are written if matplotlib is available (optional).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

# metric -> +1 if higher is better, -1 if lower is better (oriented so AUC>0.5 == predictive)
METRIC_DIR = {
    "iou": +1, "gt_containment": +1, "pred_containment": +1, "freq_coverage": +1,
    "time_coverage": +1, "energy_coverage": +1, "energy_in_window": +1,
    "cf_err_bins_abs": -1, "cf_err_cyc_abs": -1, "energy_contamination": -1,
    "bw_log_abs": -1,  # |log(bw_ratio)| derived below
}
CONSTELLATION = {"psk", "ask", "qam"}
FREQ_ANALOG = {"fsk", "msk", "chirp", "lfm", "fm", "am", "ofdm"}


def summarize(values):
    a = np.asarray([v for v in values if v is not None and v == v], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def load(jsonl):
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("recog_correct") is None:  # only boxes that were recognized
                continue
            br = d.get("bw_ratio")
            d["bw_log_abs"] = abs(math.log(br)) if (br and br > 0) else float("nan")
            rows.append(d)
    return rows


def col(rows, key):
    return np.array([(r.get(key) if r.get(key) is not None else np.nan) for r in rows], dtype=float)


def rank_auc(metric, y, direction):
    """AUC of metric predicting y (1=correct). Oriented by direction so >0.5 == predictive.
    Average-rank Mann-Whitney; ignores NaNs."""
    x = np.asarray(metric, dtype=float) * direction
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    pos, neg = (y == 1), (y == 0)
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan"), npos, nneg
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # 1-based average rank
        i = j + 1
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc), npos, nneg


def logistic_irls(X, y, iters=50, l2=1e-4):
    """IRLS logistic regression. X already standardized, intercept appended. Returns beta."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        Wv = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / Wv
        XtW = X.T * Wv
        H = XtW @ X + l2 * np.eye(p)
        try:
            beta_new = np.linalg.solve(H, XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-7:
            beta = beta_new; break
        beta = beta_new
    return beta


def zscore(a):
    m = np.nanmean(a); s = np.nanstd(a)
    return (a - m) / (s if s > 0 else 1.0)


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    # two-sided p via erfc
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def boot_diff(y_hi, y_lo, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(y_hi) == 0 or len(y_lo) == 0:
        return float("nan"), float("nan")
    d = np.empty(B)
    for b in range(B):
        d[b] = rng.choice(y_hi, len(y_hi)).mean() - rng.choice(y_lo, len(y_lo)).mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_auc(metric, y, direction, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b], _, _ = rank_auc(metric[idx], y[idx], direction)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else (f"{v:.5f}" if isinstance(v, float) else str(v)) for v in r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iou-band", nargs=2, type=float, default=[0.75, 0.85])
    ap.add_argument("--snr-edges", nargs="*", type=float, default=[-20, -10, 0, 10, 20, 30])
    args = ap.parse_args()
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    rows = load(args.jsonl)
    y = np.array([r["recog_correct"] for r in rows], dtype=float)
    print(f"[analyze] {len(rows)} recognized boxes; overall recog_acc = {y.mean():.4f}")

    metrics = [m for m in METRIC_DIR if any(m in r for r in rows[:1]) or m == "bw_log_abs"]

    # (a) recog vs binned quality (deciles)
    a_rows = []
    for m in metrics:
        x = col(rows, m); ok = ~np.isnan(x)
        if ok.sum() < 20:
            continue
        xo, yo = x[ok], y[ok]
        qs = np.quantile(xo, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        binid = np.clip(np.digitize(xo, qs[1:-1]), 0, len(qs) - 2)
        for b in range(len(qs) - 1):
            sel = binid == b
            n = int(sel.sum())
            if n == 0:
                continue
            k = int(yo[sel].sum())
            lo, hi = wilson(k, n)
            a_rows.append([m, b, round(float(qs[b]), 5), round(float(qs[b + 1]), 5), n, round(k / n, 5),
                           round(lo, 5), round(hi, 5)])
    write_csv(outd / "recog_vs_quality.csv",
              ["metric", "decile", "edge_lo", "edge_hi", "n", "recog_acc", "wilson_lo", "wilson_hi"], a_rows)

    # (b) HEADLINE: within IoU band, containment HIGH vs LOW
    iou = col(rows, "iou"); gtc = col(rows, "gt_containment")
    lo_b, hi_b = args.iou_band
    band = (iou >= lo_b) & (iou <= hi_b)
    hi_mask = band & (gtc >= 0.95)
    lo_mask = band & (gtc <= 0.80)
    yh, yl = y[hi_mask], y[lo_mask]
    z, pval = two_prop_z(int(yh.sum()), len(yh), int(yl.sum()), len(yl))
    bl, bh = boot_diff(yh, yl)
    head_rows = [
        ["IoU_band", f"[{lo_b},{hi_b}]", "", "", "", "", "", ""],
        ["HIGH_containment(>=0.95)", len(yh), round(float(yh.mean()) if len(yh) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(hi_mask)[0]], 'cf_err_cyc_abs'))) if len(yh) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(hi_mask)[0]], 'energy_coverage'))) if len(yh) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(hi_mask)[0]], 'energy_contamination'))) if len(yh) else float('nan'), 5),
         "", ""],
        ["LOW_containment(<=0.80)", len(yl), round(float(yl.mean()) if len(yl) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(lo_mask)[0]], 'cf_err_cyc_abs'))) if len(yl) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(lo_mask)[0]], 'energy_coverage'))) if len(yl) else float('nan'), 5),
         round(float(np.nanmean(col([rows[i] for i in np.where(lo_mask)[0]], 'energy_contamination'))) if len(yl) else float('nan'), 5),
         "", ""],
        ["DELTA(acc_hi-acc_lo)", round((float(yh.mean()) - float(yl.mean())) if (len(yh) and len(yl)) else float('nan'), 5),
         f"z={z:.3f}", f"p={pval:.2e}", f"boot95=[{bl:.4f},{bh:.4f}]", "", "", ""],
    ]
    write_csv(outd / "headline_iou_band.csv",
              ["group", "n_or_val", "recog_acc", "mean_cf_err_cyc_abs", "mean_energy_coverage", "mean_energy_contam", "c7", "c8"], head_rows)
    print(f"[analyze] HEADLINE IoU[{lo_b},{hi_b}]: HIGH-containment acc={yh.mean() if len(yh) else float('nan'):.4f} (n={len(yh)}) "
          f"vs LOW acc={yl.mean() if len(yl) else float('nan'):.4f} (n={len(yl)}); delta z={z:.3f} p={pval:.2e} boot95=[{bl:.4f},{bh:.4f}]")

    # (c) predictive power: rank-AUC + univariate logit + multivariate logit
    c_rows = []
    multi_feats = ["iou", "cf_err_cyc_abs", "energy_coverage", "energy_contamination", "gt_containment"]
    for m in metrics:
        x = col(rows, m); d = METRIC_DIR[m]
        auc, npos, nneg = rank_auc(x, y, d)
        al, ah = boot_auc(x, y, d) if not math.isnan(auc) else (float("nan"), float("nan"))
        ok = ~np.isnan(x)
        if ok.sum() > 20 and len(np.unique(y[ok])) == 2:
            Xs = np.column_stack([zscore(x[ok]), np.ones(ok.sum())])
            beta = logistic_irls(Xs, y[ok])
            uni = float(beta[0])
        else:
            uni = float("nan")
        c_rows.append([m, round(auc, 4), round(al, 4), round(ah, 4), round(uni, 4), npos + nneg])
    # multivariate
    M = np.column_stack([zscore(col(rows, f)) for f in multi_feats])
    okm = ~np.isnan(M).any(axis=1)
    mv = {}
    if okm.sum() > 50 and len(np.unique(y[okm])) == 2:
        Xs = np.column_stack([M[okm], np.ones(okm.sum())])
        beta = logistic_irls(Xs, y[okm])
        mv = {f: float(beta[i]) for i, f in enumerate(multi_feats)}
    write_csv(outd / "predictive_power.csv",
              ["metric", "rank_auc", "auc_lo95", "auc_hi95", "univar_logit_coef", "n"], c_rows)
    with open(outd / "predictive_power_multivar.csv", "w", encoding="utf-8") as f:
        f.write("feature,multivar_logit_coef\n")
        for fea in multi_feats:
            f.write(f"{fea},{mv.get(fea, float('nan')):.4f}\n")
    print("[analyze] rank-AUC (>.5 predictive): " + ", ".join(f"{r[0]}={r[1]}" for r in c_rows))
    print("[analyze] multivar logit coefs: " + ", ".join(f"{k}={v:.3f}" for k, v in mv.items()))

    # (d) stratify by family group and SNR
    fam = np.array([r.get("family", "other") for r in rows])
    snr = col(rows, "snr_db")
    grp = np.where(np.isin(fam, list(CONSTELLATION)), "constellation",
                   np.where(np.isin(fam, list(FREQ_ANALOG)), "freq_analog", "other"))
    edges = args.snr_edges
    d_rows = []
    for g in ["constellation", "freq_analog", "other"]:
        for mi in range(len(edges) - 1):
            sel = (grp == g) & (snr >= edges[mi]) & (snr < edges[mi + 1])
            n = int(sel.sum())
            if n < 10:
                continue
            ysel = y[sel]
            row = [g, f"[{edges[mi]},{edges[mi+1]})", n, round(float(ysel.mean()), 4)]
            for m in ["iou", "cf_err_cyc_abs", "energy_coverage", "gt_containment"]:
                a, _, _ = rank_auc(col([rows[i] for i in np.where(sel)[0]], m), ysel, METRIC_DIR[m])
                row.append(round(a, 4) if not math.isnan(a) else None)
            d_rows.append(row)
    write_csv(outd / "by_family_snr.csv",
              ["group", "snr_bucket", "n", "recog_acc", "auc_iou", "auc_cf_err", "auc_energy_cov", "auc_gt_contain"], d_rows)

    # optional plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        plot_metrics = ["iou", "cf_err_cyc_abs", "gt_containment", "energy_coverage", "energy_contamination", "freq_coverage"]
        for ax, m in zip(axes.ravel(), plot_metrics):
            sub = [r for r in a_rows if r[0] == m]
            if not sub:
                continue
            cx = [(r[2] + r[3]) / 2 for r in sub]; cy = [r[5] for r in sub]
            ax.plot(cx, cy, "o-"); ax.set_title(f"{m}  (AUC={dict((r[0],r[1]) for r in c_rows).get(m)})")
            ax.set_xlabel(m); ax.set_ylabel("recog acc"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        fig.suptitle("Recognition success vs box-quality metric (deciles)")
        fig.tight_layout(); fig.savefig(outd / "recog_vs_quality.png", dpi=110)
        # headline bar
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        accs = [float(yh.mean()) if len(yh) else 0, float(yl.mean()) if len(yl) else 0]
        ax2.bar(["HIGH containment", "LOW (offset)"], accs, color=["#2c7", "#c44"])
        ax2.set_title(f"recog acc within IoU[{lo_b},{hi_b}]\n(n_hi={len(yh)}, n_lo={len(yl)}, p={pval:.1e})")
        ax2.set_ylabel("recog acc"); ax2.set_ylim(0, 1)
        for i, v in enumerate(accs):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fig2.tight_layout(); fig2.savefig(outd / "headline_iou_band.png", dpi=110)
        print(f"[analyze] plots -> {outd}")
    except Exception as e:
        print(f"[analyze] plotting skipped ({e})")
    print(f"[analyze] CSVs -> {outd}")


if __name__ == "__main__":
    main()
