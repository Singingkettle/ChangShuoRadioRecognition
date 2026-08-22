#!/usr/bin/env python
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0.
"""Scene-grouped cross-validated AUC of box-quality scalars with scene-bootstrap CIs.

Closes ledger item C17: the archived AUC table was a set of in-sample point estimates with
no grouped cross-validation, no nested hyper-parameter selection, and no uncertainty.

Protocol (fixed before running):
  * groups = scene id (``sid``); outer 5-fold GroupKFold (deterministic contiguous split of
    a seeded permutation of scenes); inner 3-fold GroupKFold on the training folds selects the
    L2 strength C of the 11-feature logistic model from {0.01, 0.1, 1, 10} by mean inner AUC.
  * univariate scalars are evaluated as-is (no fitting) on the same out-of-fold rows, so every
    number in the table is computed on the same held-out rows.
  * uncertainty: 2,000 scene-level bootstrap resamples (numpy default_rng(20260821)) of the
    out-of-fold scores; 95% percentile intervals.  The multivariate OOF scores are fixed before
    bootstrapping (no refit inside the bootstrap).
  * two populations, as in the archived table: all matched detections, and the oracle-correct
    subset (oracle_correct == 1).
Only numpy/scipy are used (Mann-Whitney rank AUC with average ranks; L-BFGS logistic fit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

FEATURES = {  # name -> (field, higher_is_better)
    "iou": ("iou", True),
    "gt_containment": ("gt_containment", True),
    "pred_containment": ("pred_containment", True),
    "freq_coverage": ("freq_coverage", True),
    "time_coverage": ("time_coverage", True),
    "energy_coverage": ("energy_coverage", True),
    "energy_in_window": ("energy_in_window", True),
    "energy_contamination": ("energy_contamination", False),
    "cf_err_bins_abs": ("cf_err_bins_abs", False),
    "cf_err_cyc_abs": ("cf_err_cyc_abs", False),
    "size_fit": ("__bw_log_abs", False),
}
C_GRID = (0.01, 0.1, 1.0, 10.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.sum()
    neg = label.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)  # average ranks on ties
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sids = np.array([r["sid"] for r in rows])
    y = np.array([int(r["recog_correct"]) for r in rows])
    oracle = np.array([int(r["oracle_correct"]) if r.get("oracle_correct") is not None else -1 for r in rows])
    X = np.empty((len(rows), len(FEATURES)))
    for j, (name, (field, hib)) in enumerate(FEATURES.items()):
        if field == "__bw_log_abs":
            v = np.array([abs(math.log(max(r["bw_ratio"], 1e-9))) for r in rows])
        else:
            v = np.array([float(r[field]) for r in rows])
        X[:, j] = v if hib else -v  # orient: larger == better box
    return sids, y, oracle, X


def group_folds(sids: np.ndarray, n_folds: int, rng: np.random.Generator):
    uniq = np.unique(sids)
    perm = rng.permutation(uniq)
    fold_of_scene = {s: i % n_folds for i, s in enumerate(perm)}
    return np.array([fold_of_scene[s] for s in sids])


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    lam = 1.0 / (C * n)
    yy = 2.0 * y - 1.0

    def f(w):
        z = Xb @ w
        loss = np.mean(np.logaddexp(0.0, -yy * z)) + 0.5 * lam * np.dot(w[:-1], w[:-1])
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y) / n
        g[:-1] += lam * w[:-1]
        return loss, g

    res = optimize.minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                            options={"maxiter": 500})
    return res.x


def standardise(Xtr: np.ndarray, Xte: np.ndarray):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def run_population(name: str, sids, y, X, rng_folds: np.random.Generator, n_boot: int, rng_boot: np.random.Generator):
    t0 = time.time()
    folds = group_folds(sids, 5, rng_folds)
    oof = np.zeros(len(y))
    chosen = []
    for k in range(5):
        tr, te = folds != k, folds == k
        # inner 3-fold grouped selection of C
        inner = group_folds(sids[tr], 3, rng_folds)
        Xtr_all, ytr_all = X[tr], y[tr]
        best_c, best_auc = None, -1.0
        for C in C_GRID:
            aucs = []
            for j in range(3):
                itr, ite = inner != j, inner == j
                a, b = standardise(Xtr_all[itr], Xtr_all[ite])
                w = fit_logistic(a, ytr_all[itr], C)
                aucs.append(rank_auc(b @ w[:-1] + w[-1], ytr_all[ite]))
            m = float(np.mean(aucs))
            if m > best_auc:
                best_auc, best_c = m, C
        a, b = standardise(X[tr], X[te])
        w = fit_logistic(a, y[tr], best_c)
        oof[te] = b @ w[:-1] + w[-1]
        chosen.append({"fold": k, "C": best_c, "inner_auc": best_auc,
                       "fold_auc": rank_auc(oof[te], y[te]), "n_test": int(te.sum())})
    scores = {f: X[:, j] for j, f in enumerate(FEATURES)}
    scores["multivariate_11_features"] = oof
    # per-fold AUC of every scalar on the same held-out rows
    per_fold = {f: [rank_auc(s[folds == k], y[folds == k]) for k in range(5)] for f, s in scores.items()}
    point = {f: rank_auc(s, y) for f, s in scores.items()}
    # scene bootstrap of pooled OOF AUC
    uniq, inv = np.unique(sids, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]
    starts = np.searchsorted(inv_sorted, np.arange(len(uniq) + 1))
    boots = {f: np.empty(n_boot) for f in scores}
    for b in range(n_boot):
        pick = rng_boot.integers(0, len(uniq), len(uniq))
        counts = np.bincount(pick, minlength=len(uniq))
        idx = np.repeat(order, np.repeat(counts, starts[1:] - starts[:-1]))
        yb = y[idx]
        for f, s in scores.items():
            boots[f][b] = rank_auc(s[idx], yb)
        if (b + 1) % 250 == 0:
            print(f"  [{name}] bootstrap {b + 1}/{n_boot} ({time.time() - t0:.0f}s)", flush=True)
    table = {}
    for f in scores:
        table[f] = {"auc_pooled_oof": point[f],
                    "auc_fold_mean": float(np.nanmean(per_fold[f])),
                    "auc_fold_sd_ddof1": float(np.nanstd(per_fold[f], ddof=1)),
                    "ci95_scene_bootstrap": [float(np.quantile(boots[f], 0.025)), float(np.quantile(boots[f], 0.975))]}
    # paired comparison: multivariate minus IoU (same resamples)
    diff = boots["multivariate_11_features"] - boots["iou"]
    ranking = sorted(table, key=lambda f: table[f]["auc_pooled_oof"], reverse=True)
    return {
        "n_rows": int(len(y)), "n_scenes": int(len(uniq)), "positive_rate": float(y.mean()),
        "outer_folds": chosen, "table": table, "ranking_by_pooled_oof_auc": ranking,
        "multivariate_minus_iou": {"point": point["multivariate_11_features"] - point["iou"],
                                   "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]},
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = {"schema_version": 1, "protocol": __doc__.strip(), "seed": args.seed,
           "resamples": args.resamples, "c_grid": list(C_GRID), "dumps": {}}
    for item in args.jsonl:
        label, raw = item.split("=", 1)
        path = Path(raw).resolve()
        print(f"== {label}: {path}", flush=True)
        sids, y, oracle, X = load(path)
        entry = {"path": str(path), "sha256": sha256(path), "populations": {}}
        for pop, mask in (("all_matched", np.ones(len(y), dtype=bool)), ("oracle_correct", oracle == 1)):
            rng_folds = np.random.default_rng(args.seed)
            rng_boot = np.random.default_rng(args.seed + 1)
            entry["populations"][pop] = run_population(f"{label}/{pop}", sids[mask], y[mask], X[mask],
                                                       rng_folds, args.resamples, rng_boot)
            t = entry["populations"][pop]["table"]
            print(f"  {pop}: n={mask.sum()} iou={t['iou']['auc_pooled_oof']:.4f} "
                  f"{t['iou']['ci95_scene_bootstrap']} multi={t['multivariate_11_features']['auc_pooled_oof']:.4f} "
                  f"{t['multivariate_11_features']['ci95_scene_bootstrap']}", flush=True)
        out["dumps"][label] = entry
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
