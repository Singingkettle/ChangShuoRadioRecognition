# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""SNR-stratified analysis on the CORRECTED (signal-block) SNR axis.

metadata snr_db is a whole-observation average (signal occupies only ~2% of the time-freq area), so it
understates the signal's true visibility by the time-freq dilution. We re-label every signal with the
block-level SNR = snr_db + 10log10(1/(time_frac*freq_frac)) and re-stratify localization recall vs
class-aware recognition (vision / return-to-IQ / perfect-box) + constellation-family rescue."""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MM = ROOT / "data/torchsig_hardshort_lowsnr_stft3_memmap"
RAWDS = ROOT / "data/torchsig_hardshort_lowsnr_iq_65k_nvme"
JSONL = ROOT / "work_dirs/returniq_cache/box_quality_oracle.jsonl"
LIMIT = 2000

cats = sorted(json.loads((MM / "coco_multiclass/annotations/instances_train.json").read_text())["categories"], key=lambda c: c["id"])
cid2idx = {c["id"]: i for i, c in enumerate(cats)}
CONSTELLATION = {"psk", "qam", "ask"}

# ---- corrected SNR per (sid, gi); also total GT per corrected-SNR for recall ----
csnr_map = {}
tot_csnr = []
for line in (RAWDS / "metadata/test.jsonl").read_text(encoding="utf-8").splitlines()[:LIMIT]:
    if not line.strip():
        continue
    r = json.loads(line); nq = r["num_iq_samples"]; fs = r["sample_rate"]; sid = r["sample_id"]
    gi = 0
    for inst in r["instances"]:
        if inst["category_id"] not in cid2idx:
            continue
        tf = inst["duration_in_samples"] / nq; ff = inst["bandwidth"] / fs
        cs = float(inst["snr_db"]) + 10.0 * np.log10(1.0 / max(tf * ff, 1e-12))
        csnr_map[(sid, gi)] = cs; tot_csnr.append(cs); gi += 1
tot_csnr = np.array(tot_csnr)

# ---- recognition from jsonl (per-GT dedup), re-keyed to corrected SNR ----
byg = {}
for line in open(JSONL, encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    if d.get("recog_correct") is None or d.get("oracle_correct") is None:
        continue
    k = (d["sid"], d["gi"])
    if k not in byg or d["det_score"] > byg[k]["det_score"]:
        byg[k] = d
recs = [r for r in byg.values() if (r["sid"], r["gi"]) in csnr_map]
m_csnr = np.array([csnr_map[(r["sid"], r["gi"])] for r in recs], float)
m_meta = np.array([r["snr_db"] for r in recs], float)
m_fam = np.array([r["family"] for r in recs])
vis = np.array([r["vision_correct"] for r in recs], float)
iqp = np.array([r["recog_correct"] for r in recs], float)
iqo = np.array([r["oracle_correct"] for r in recs], float)

print(f"corrected (block) SNR range: min {tot_csnr.min():.1f}  median {np.median(tot_csnr):.1f}  max {tot_csnr.max():.1f}")
print(f"(metadata snr range was -20..+10; correction = +10log10(1/(tf*ff)) median +16.7dB)\n")
EDGES = [-5, 0, 5, 10, 15, 20, 25, 30, 35]
def acc(mask, arr):
    m = mask & ~np.isnan(arr); return float(arr[m].mean()) if m.any() else float("nan")
print(f"{'block-SNR':>12} {'#GT':>6} {'locRecall':>9} {'visRecog':>8} {'IQ(pred)':>8} {'IQ(perf)':>8} | {'const:vis':>9} {'const:IQ':>8}")
rows = []
for i in range(len(EDGES)-1):
    lo, hi = EDGES[i], EDGES[i+1]
    tot = int(((tot_csnr >= lo) & (tot_csnr < hi)).sum())
    mb = (m_csnr >= lo) & (m_csnr < hi)
    recall = int(mb.sum()) / tot if tot else float("nan")
    va, ia, oa = acc(mb, vis), acc(mb, iqp), acc(mb, iqo)
    cm = mb & np.isin(m_fam, list(CONSTELLATION))
    cva, cia = acc(cm, vis), acc(cm, iqp)
    if tot >= 30:
        print(f"[{lo:>4},{hi:>4}) {tot:>6} {recall:>9.3f} {va:>8.3f} {ia:>8.3f} {oa:>8.3f} | {cva:>9.3f} {cia:>8.3f}")
        rows.append([lo, hi, tot, recall, va, ia, oa, cva, cia])

with open(ROOT / "work_dirs/returniq_cache/snr_stratified_corrected.csv", "w") as f:
    f.write("blocksnr_lo,blocksnr_hi,n_gt,loc_recall,vision_recog,IQ_pred,IQ_perfect,constel_vision,constel_IQ\n")
    for r in rows:
        f.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
print("\n[saved] work_dirs/returniq_cache/snr_stratified_corrected.csv")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xc = [(r[0]+r[1])/2 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(xc, [r[3] for r in rows], "o-", label="localization recall (class-agnostic)", lw=2, color="#1f77b4")
    ax[0].plot(xc, [r[4] for r in rows], "s-", label="recognition: vision", lw=2, color="#d62728")
    ax[0].plot(xc, [r[5] for r in rows], "^-", label="recognition: return-to-IQ (pred box)", lw=2, color="#2ca02c")
    ax[0].plot(xc, [r[6] for r in rows], "^--", label="recognition: return-to-IQ (perfect box)", lw=1.5, color="#2ca02c", alpha=0.5)
    ax[0].set_xlabel("signal-block SNR (dB)  [corrected]"); ax[0].set_ylabel("accuracy / recall"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Localization easy, recognition hard (block-SNR axis)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(xc, [r[7] for r in rows], "s-", label="constellation: vision", lw=2, color="#d62728")
    ax[1].plot(xc, [r[8] for r in rows], "^-", label="constellation: return-to-IQ", lw=2, color="#2ca02c")
    ax[1].set_xlabel("signal-block SNR (dB)  [corrected]"); ax[1].set_ylabel("recognition accuracy"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("Return-to-IQ rescues constellation families"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "work_dirs/returniq_cache/snr_stratified_corrected.png", dpi=130)
    print("[saved] work_dirs/returniq_cache/snr_stratified_corrected.png")
except Exception as e:
    print(f"[plot skipped] {e}")
