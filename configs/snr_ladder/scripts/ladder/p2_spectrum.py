#!/usr/bin/env python3
"""P2 spectrum, final table generator.

Audits every 'how to spend the SNR' method against the SAME frozen hard model, on
TWO settings (RML2016.10B/PETCGDNN and RML2016.10A/MCformer). Same ladder math as
comprehensive_audit; three seeds each; paired t-CI on method-minus-affine-readout.
Writes one CSV that is the single source of truth for the paper's spectrum table.

The ladder depends only on (setting, seed), never on the method, so it is fit ONCE
per (setting, seed) and reused across methods. Thread count is capped because the
box also runs training jobs; unbounded MKL threads oversubscribe and slow everything
down.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import csv  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np
from scipy.stats import t as student_t  # noqa: E402

from ladder_lib import sm, acc, band_mask, per_bin, EPS  # noqa: E402

W = "work_dirs"
SEEDS = ["seed_2026", "seed_2027", "seed_2028"]
T_975_DF2 = 4.302653

SETTINGS = [
    dict(
        tag="RML2016.10B / PETCGDNN",
        hard=f"{W}/secondsurv_petcgdnn10b_hard_100ep/amc/deepsig201610B/petcgdnn_hard-ce",
        base=f"{W}/p2_spectrum/amc/deepsig201610B",
        methods=[
            ("target: focal loss", "petcgdnn_focal"),
            ("loss weighting: SNR-inverse", "petcgdnn_snrweight"),
            ("curriculum: high-SNR first", "petcgdnn_curriculum"),
            ("architecture: FiLM (scale 0.1)", "petcgdnn_film"),
        ],
    ),
    dict(
        tag="RML2016.10A / MCformer",
        hard=f"{W}/baseline_gate_v2/amc/deepsig201610A/mcformer_hard-ce",
        base=f"{W}/p2_spectrum_10A/amc/deepsig201610A",
        methods=[
            ("target: focal loss", "petcgdnn_focal"),
            ("loss weighting: SNR-inverse", "petcgdnn_snrweight"),
            ("curriculum: high-SNR first", "petcgdnn_curriculum"),
            ("architecture: FiLM (scale 0.1)", "mcformer_film"),
        ],
    ),
]


def load(p):
    d = pickle.load(open(p, "rb"))
    return (np.log(np.clip(np.asarray(d["pps"], float), EPS, 1)),
            np.asarray(d["gts"]).astype(int), np.asarray(d["snrs"]).astype(float))


def tci(xs):
    """Two-sided 95% t-CI of the mean, critical value at df=n-1 (12.706 at n=2)."""
    xs = np.asarray(xs, float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = float(student_t.ppf(0.975, n - 1)) * float(xs.std(ddof=1)) / np.sqrt(n)
    return m, m - h, m + h


_LADDER = {}


def ladder(hard_dir, seed):
    """Fit the frozen hard model's per-bin affine readout once per (hard, seed)."""
    key = (hard_dir, seed)
    if key in _LADDER:
        return _LADDER[key]
    hv = Path(hard_dir) / seed / "predictions" / "validation.pkl"
    ht = Path(hard_dir) / seed / "predictions" / "test.pkl"
    if not (hv.exists() and ht.exists()):
        _LADDER[key] = None
        return None
    vZ, vY, vR = load(str(hv))
    tZ, tY, tR = load(str(ht))
    hp = sm(tZ)
    band = band_mask(hp, tY, tR)
    ro = per_bin("aff", vZ, vY, vR, tZ, tR)
    out = dict(tY=tY, band=band,
               hard=acc(hp[band], tY[band]), read=acc(ro[band], tY[band]),
               hard_all=acc(hp, tY))
    print(f"    [ladder] {Path(hard_dir).name}/{seed}: band hard {out['hard']:.1f}, "
          f"readout {out['read']:.1f}, n_band {int(band.sum())}", flush=True)
    _LADDER[key] = out
    return out


rows = []
for st in SETTINGS:
    print(f"\n=== {st['tag']} ===", flush=True)
    for s in SEEDS:
        ladder(st["hard"], s)
    print(f"{'method':32s} {'seed':>5s} | {'hard':>5s} {'meth':>5s} {'read':>7s} | {'m-aff':>7s}",
          flush=True)
    for label, run in st["methods"]:
        gaps, per = [], []
        for s in SEEDS:
            L = ladder(st["hard"], s)
            mt = Path(f"{st['base']}/{run}") / s / "predictions" / "test.pkl"
            if L is None or not mt.exists():
                print(f"{label:32s} {s[-4:]:>5s} |  (missing "
                      f"{'hard' if L is None else 'method'})", flush=True)
                continue
            mP = sm(load(str(mt))[0])
            if mP.shape[0] != L["tY"].size:
                print(f"{label:32s} {s[-4:]:>5s} |  (size mismatch)", flush=True)
                continue
            meth = acc(mP[L["band"]], L["tY"][L["band"]])
            meth_all = acc(mP, L["tY"])
            g = meth - L["read"]
            gaps.append(g)
            per.append((s[-4:], L["hard"], meth, L["read"], L["hard_all"], meth_all))
            print(f"{label:32s} {s[-4:]:>5s} | {L['hard']:5.1f} {meth:5.1f} {L['read']:7.1f} "
                  f"| {g:7.2f}   (all-SNR hard {L['hard_all']:.1f} / meth {meth_all:.1f})",
                  flush=True)
        if not gaps:
            continue
        m, lo, hi = tci(gaps)
        verdict = ("decision-layer" if hi < 0 else
                   "representation-level" if lo > 0 else "inconclusive")
        hb = float(np.mean([p[1] for p in per]))
        mb = float(np.mean([p[2] for p in per]))
        rb = float(np.mean([p[3] for p in per]))
        print(f"{label:32s} {'MEAN':>5s} | {hb:5.1f} {mb:5.1f} {rb:7.1f} | {m:7.2f}  "
              f"CI [{lo:.2f},{hi:.2f}] -> {verdict}", flush=True)
        print("-" * 92, flush=True)
        rows.append(dict(
            setting=st["tag"], method=label, run=run, n_seeds=len(gaps),
            hard_band=round(hb, 2), method_band=round(mb, 2), readout_band=round(rb, 2),
            hard_allsnr=round(float(np.mean([p[4] for p in per])), 2),
            method_allsnr=round(float(np.mean([p[5] for p in per])), 2),
            gap_mean=round(m, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
            verdict=verdict,
            per_seed="; ".join(f"{p[0]}:{p[1]:.1f}/{p[2]:.1f}/{p[3]:.1f}" for p in per)))

if not rows:
    print("no rows produced", flush=True)
    sys.exit(1)

out = Path("work_dirs") / "p2_spectrum.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out} ({len(rows)} rows)", flush=True)
