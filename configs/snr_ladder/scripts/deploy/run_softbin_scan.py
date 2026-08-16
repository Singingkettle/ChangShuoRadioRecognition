#!/usr/bin/env python3
"""Family E step 1+2: hard-routing baseline reproduction + soft-bin sigma scan.

Cells:
  rml201610b_mcformer  (headline cell, 3 seeds)
  ucsd_rml22_mcformer  (deployment cell, 3 seeds)

For each seed: fit per-bin F_aff table on val at TRUE bin (paper-identical),
then sweep sigma in {0,0.5,1,2,3,4,6} dB. For each sigma one noise draw
shat = true + N(0, sigma) shared by both routers (paired comparison):
  router=hard  quantize-to-nearest-bin (paper baseline)
  router=soft  Gaussian-kernel marginalised readout, sigma_r = sigma (matched)
Reference lines (sigma-independent): hard model, training-time method, global bias.

Output: softbin_scan.csv  rows = cell,sigma,router,band_acc  (3-seed mean; plus
        per-seed columns kept in softbin_scan_perseed.csv for audit).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

CAMP = Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from softbin_lib import (load_pred, sm, acc, band_mask, get_affine_table,
                         perbin_probs, route_hard, route_soft)

ART10B = Path("work_dirs/prediction_artifacts/rml201610b_mcformer")
ARTR22 = Path("work_dirs/prediction_artifacts/ucsd_rml22_mcformer")

SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
SEEDS = [2026, 2027, 2028]

CELLS = {
    "rml201610b_mcformer": {
        "val": lambda s: ART10B / f"seed_{s}" / "hard_val.pkl",
        "test": lambda s: ART10B / f"seed_{s}" / "hard_test.pkl",
        "method": lambda s: ART10B / f"seed_{s}" / "dpc_rcps_test.pkl",
    },
    "ucsd_rml22_mcformer": {
        "val": lambda s: ARTR22 / f"seed_{s}" / "hard_val.pkl",
        "test": lambda s: ARTR22 / f"seed_{s}" / "hard_test.pkl",
        "method": lambda s: ARTR22 / f"seed_{s}" / "method_test.pkl",
    },
}


def run_cell_seed(cell, seed):
    c = CELLS[cell]
    tab = get_affine_table(cell, seed, c["val"](seed))
    tZ, tY, tR0 = load_pred(c["test"](seed))
    mP = sm(np.log(np.clip(np.asarray(
        __import__("pickle").load(open(c["method"](seed), "rb"))["pps"], float), 1e-12, 1)))
    hp = sm(tZ)
    band = band_mask(hp, tY, tR0)
    bi = np.where(band)[0]
    Zb, Yb, Rb = tZ[bi], tY[bi], tR0[bi]
    Pb = perbin_probs(Zb, tab)
    row = {"hard": acc(hp[bi], tY[bi]), "method": acc(mP[bi], tY[bi]),
           "global": acc(sm(Zb + tab["global_bias"][None]), Yb)}
    res = {}
    for s in SIGMAS:
        rng = np.random.default_rng([seed, int(round(s * 10))])
        shat = Rb if s == 0 else Rb + rng.normal(0, s, Rb.shape)
        res[(s, "hard")] = acc(route_hard(Pb, tab, shat), Yb)
        res[(s, "soft")] = acc(route_soft(Pb, tab, shat, sigma_r=s), Yb)
    row["scan"] = res
    row["band_bins"] = sorted(np.unique(Rb).tolist())
    row["n_band"] = int(bi.size)
    return row


def main():
    per_seed = {}
    for cell in CELLS:
        for seed in SEEDS:
            t0 = time.time()
            per_seed[(cell, seed)] = run_cell_seed(cell, seed)
            r = per_seed[(cell, seed)]
            print(f"[{cell} seed{seed}] band={r['n_band']} bins={r['band_bins']} "
                  f"hard={r['hard']:.2f} method={r['method']:.2f} global={r['global']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for s in SIGMAS:
                print(f"    sigma={s:<4} hard-route={r['scan'][(s,'hard')]:6.2f} "
                      f"soft-route={r['scan'][(s,'soft')]:6.2f}", flush=True)

    # aggregate + write
    rows = [["cell", "sigma", "router", "band_acc"]]
    rows_ps = [["cell", "seed", "sigma", "router", "band_acc"]]
    for cell in CELLS:
        seeds = [s for s in SEEDS if (cell, s) in per_seed]
        for ref in ("hard", "method", "global"):
            m = np.mean([per_seed[(cell, s)][ref] for s in seeds])
            rows.append([cell, "", ref, f"{m:.3f}"])
            for s in seeds:
                rows_ps.append([cell, s, "", ref, f"{per_seed[(cell, s)][ref]:.3f}"])
        for sg in SIGMAS:
            for router in ("hard", "soft"):
                m = np.mean([per_seed[(cell, s)]["scan"][(sg, router)] for s in seeds])
                rows.append([cell, sg, router, f"{m:.3f}"])
                for s in seeds:
                    rows_ps.append([cell, s, sg, router,
                                    f"{per_seed[(cell, s)]['scan'][(sg, router)]:.3f}"])
    with open(CAMP / "softbin_scan.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(CAMP / "softbin_scan_perseed.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows_ps)

    # crossover report
    for cell in CELLS:
        seeds = [s for s in SEEDS if (cell, s) in per_seed]
        meth = np.mean([per_seed[(cell, s)]["method"] for s in seeds])
        print(f"\n=== {cell}: method={meth:.2f} ===")
        for router in ("hard", "soft"):
            cross = None
            for sg in SIGMAS:
                m = np.mean([per_seed[(cell, s)]["scan"][(sg, router)] for s in seeds])
                d = m - meth
                if cross is None and d <= 0:
                    cross = sg
                print(f"  {router:4s} sigma={sg:<4} {m:6.2f} vs method {d:+6.2f}")
            print(f"  {router} crossover sigma* = "
                  f"{'>6 (none in range)' if cross is None else cross}")
    print("\nwrote softbin_scan.csv / softbin_scan_perseed.csv")


if __name__ == "__main__":
    main()
