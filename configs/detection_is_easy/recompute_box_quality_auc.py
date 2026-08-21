# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Rank-AUC of every box-quality scalar as a predictor of recognition success.

Two populations are reported, and the difference between them is the point of the
analysis:

* **all matched detections** -- raw IoU looks useless here (AUC ~0.49), but the number is
  confounded: a large share of signals are unrecognisable even from a perfect box, so
  box quality cannot predict an outcome that was never available.
* **the oracle-correct subset** (``oracle_correct == 1``) -- signals the recognizer does
  get right when handed the ground-truth box. Conditioning on this removes intrinsic
  signal difficulty and asks the actual question: given a recognisable signal, does box
  quality decide whether we recognise it?

Also fits a multivariate logistic model over the candidate scalars and reports its AUC,
which bounds what any hand-designed "signal-fidelity IoU" could achieve.

    python configs/detection_is_easy/recompute_box_quality_auc.py \\
      --jsonl work_dirs/returniq_cache/box_quality_oracle.jsonl \\
      --out work_dirs/returniq_cache/box_quality_auc.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# name -> (jsonl field, higher_is_better). Scalars are oriented so that a LARGER value
# should mean a BETTER box; AUC > 0.5 then means "predictive in the expected direction".
FEATURES = {
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
    "size_fit": ("__bw_log_abs", False),      # |log(bw_ratio)|: 0 == perfectly sized
}


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney rank AUC with ties handled by average ranks."""
    pos, neg = int(label.sum()), int((1 - label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    i = 0
    while i < len(sorted_score):
        j = i
        while j + 1 < len(sorted_score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[label == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def logistic_auc(X: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.5):
    """Standardised multivariate logistic fit; returns (in-sample AUC, coefficients)."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (y - p) / len(y)
        w += lr * grad
    return rank_auc(Z @ w, y), w[:-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="dump from `bridge.py diag-quality --with-oracle`")
    ap.add_argument("--out", default=None, help="write the table as CSV here")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("recog_correct") is None:
                continue
            bw = d.get("bw_ratio")
            d["__bw_log_abs"] = abs(math.log(bw)) if bw and bw > 0 else float("nan")
            rows.append(d)
    if not rows:
        raise SystemExit(f"no usable records in {args.jsonl}")

    recog = np.array([float(r["recog_correct"]) for r in rows])
    oracle = np.array([float(r.get("oracle_correct") or 0.0) for r in rows])
    has_oracle = any(r.get("oracle_correct") is not None for r in rows)

    populations = [("all_matched", np.ones(len(rows), dtype=bool))]
    if has_oracle and oracle.sum() > 0:
        populations.append(("oracle_correct", oracle == 1))

    out_rows = []
    for pop_name, mask in populations:
        y = recog[mask]
        print(f"\n=== population: {pop_name}  n={int(mask.sum())}  "
              f"recognised={y.mean():.4f} ===")
        usable = []
        for name, (field, higher_better) in FEATURES.items():
            v = np.array([float(r.get(field, np.nan) or 0.0) if r.get(field) is not None
                          else np.nan for r in rows])[mask]
            if not np.isfinite(v).all():
                good = np.isfinite(v)
                if good.sum() < 10:
                    continue
                auc = rank_auc(v[good] if higher_better else -v[good], y[good])
                n_used = int(good.sum())
            else:
                auc = rank_auc(v if higher_better else -v, y)
                n_used = int(mask.sum())
                usable.append((name, v if higher_better else -v))
            print(f"  {name:22} AUC={auc:.4f}  n={n_used}")
            out_rows.append({"population": pop_name, "metric": name,
                             "rank_auc": round(float(auc), 5), "n": n_used})
        if usable:
            X = np.stack([v for _, v in usable], axis=1)
            mv_auc, coefs = logistic_auc(X, y)
            print(f"  {'MULTIVARIATE(' + str(len(usable)) + ' features)':22} AUC={mv_auc:.4f}")
            out_rows.append({"population": pop_name,
                             "metric": f"multivariate_{len(usable)}_features",
                             "rank_auc": round(float(mv_auc), 5), "n": int(mask.sum())})
            for (name, _), c in zip(usable, coefs):
                out_rows.append({"population": pop_name, "metric": f"multivar_coef::{name}",
                                 "rank_auc": round(float(c), 5), "n": int(mask.sum())})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["population", "metric", "rank_auc", "n"])
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
