#!/usr/bin/env python3
"""Figure: in-band decomposition of the gap to the Bayes ceiling (waterfall).

For each backbone, one vertical distance bar from the hard head to Bayes*,
split into a decision deficit (hard -> readout) and a representation deficit
(readout -> Bayes*). Hard / readout / Bayes* levels are drawn as short tick
lines with numeric labels (hard tick extends left, readout tick right, so
coincident levels stay visible). Band summaries are aggregated from
../../results/decomp_synA.csv: per backbone, mean over its transition-band
bins and seeds (bands differ per backbone). Writes figs/decomp_waterfall.pdf.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEC = Path(__file__).resolve().parents[2] / "results" / "decomp_synA.csv"

# audited per-backbone transition bands (dB, inclusive)
BANDS = {"petcgdnn": (-8, 2), "mcformer": (-10, -2), "cgdnet": (-8, 4)}
LABEL = {"petcgdnn": "PETCGDNN", "mcformer": "MCformer", "cgdnet": "CGDNet"}
ORDER = ["petcgdnn", "mcformer", "cgdnet"]

C_DEC = "#0072B2"   # decision deficit  (hard -> readout)
C_REP = "#D55E00"   # representation deficit (readout -> Bayes*)
C_HARD = "#444444"


def band_summaries():
    agg = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(DEC.open(encoding="utf-8-sig")):
        bb = r["backbone"]
        lo, hi = BANDS[bb]
        if lo <= float(r["snr"]) <= hi:
            for k in ("bayes", "readout", "hard"):
                agg[bb][k].append(float(r[k]))
    out = []
    for bb in ORDER:
        m = {k: sum(v) / len(v) for k, v in agg[bb].items()}
        out.append((LABEL[bb], m["bayes"], m["readout"], m["hard"]))
    return out


def main():
    data = band_summaries()
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    bw = 0.16          # half-width of the distance bar
    tick = 0.30        # tick lines: hard to the left, readout to the right

    for i, (name, bayes, ro, hard) in enumerate(data):
        dec, rep = ro - hard, bayes - ro
        ax.add_patch(Rectangle((i - bw, ro), 2 * bw, rep, facecolor=C_REP,
                               alpha=0.85, edgecolor="none", zorder=2))
        if abs(dec) > 1e-9:
            ax.add_patch(Rectangle((i - bw, min(hard, ro)), 2 * bw, abs(dec),
                                   facecolor=C_DEC, edgecolor="none", zorder=3))
        ax.plot([i - tick, i + tick], [bayes] * 2, color="black", lw=1.6, zorder=4)
        ax.plot([i - tick, i + 0.02], [hard] * 2, color=C_HARD, lw=1.2, zorder=4)
        ax.plot([i - 0.02, i + tick], [ro] * 2, color=C_DEC, lw=1.2, zorder=4)
        ax.text(i, bayes + 0.6, f"Bayes* {bayes:.1f}", ha="center", va="bottom",
                fontsize=7, zorder=5)
        ax.text(i - tick - 0.04, hard - 0.4, f"hard {hard:.1f}", ha="right",
                va="top", fontsize=6.5, color=C_HARD, zorder=5)
        ax.text(i + tick + 0.04, ro, f"readout {ro:.1f}", ha="left",
                va="center", fontsize=6.5, color=C_DEC, zorder=5)
        ax.text(i + tick + 0.04, ro + rep / 2 + 1.2, f"$+{rep:.1f}$ pp",
                ha="left", va="center", fontsize=7, color=C_REP, zorder=5)
        ax.text(i - tick - 0.04, (hard + ro) / 2 + 0.4, f"${dec:+.1f}$",
                ha="right", va="bottom", fontsize=6.5, color=C_DEC, zorder=5)

    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=C_REP, alpha=0.85),
                       Rectangle((0, 0), 1, 1, facecolor=C_DEC)],
              labels=["representation deficit (readout $\\to$ Bayes*)",
                      "decision deficit (hard $\\to$ readout)"],
              fontsize=6.3, frameon=False, loc="upper left",
              handlelength=1.1, handleheight=1.0, labelspacing=0.35)
    ax.text(0.99, 0.02, "transition bands differ per backbone",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#777777", style="italic")

    ax.set_xlim(-0.95, 2.95)
    lo = min(d[3] for d in data)
    hi = max(d[1] for d in data)
    ax.set_ylim(lo - 5, hi + 6)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=8.5)
    ax.set_ylabel("transition-band accuracy (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "decomp_waterfall.pdf", bbox_inches="tight")
    print("wrote", out / "decomp_waterfall.pdf")


if __name__ == "__main__":
    main()
