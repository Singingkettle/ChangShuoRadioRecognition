#!/usr/bin/env python3
"""Figure: the Bayes ceiling vs three trained backbones (synthetic AWGN anchor).

Overlays the exact/SIS-certified Bayes-optimal accuracy curve (black, thick;
SIS-certified tail dashed) with the 3-seed mean accuracy of each backbone
(PETCGDNN / MCformer / CGDNet): hard head solid, per-bin affine readout as a
thin dashed line in the same colour. Readout ~= hard everywhere, so the
headroom is not decision-layer slack. Writes figs/bayes_ceiling.pdf next to
this script. Inputs default to ../../results/.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parents[2] / "results"
CEIL = RESULTS / "ceiling_final.csv"
DEC = RESULTS / "decomp_synA.csv"

# Okabe-Ito, one colour per backbone (paper spellings)
BB = [("petcgdnn", "PETCGDNN", "#0072B2"),
      ("mcformer", "MCformer", "#D55E00"),
      ("cgdnet", "CGDNet", "#009E73")]


def main():
    crows = list(csv.DictReader(CEIL.open(encoding="utf-8-sig")))
    csnr = [float(r["snr"]) for r in crows]
    cacc = [100.0 * float(r["bayes_acc"]) for r in crows]
    exact = [r["source"] == "tier_e_exact" for r in crows]
    n_ex = sum(exact)  # exact rows come first, SIS-certified tail after

    # 3-seed mean per (backbone, snr) for hard and readout
    acc = defaultdict(list)
    for r in csv.DictReader(DEC.open(encoding="utf-8-sig")):
        key = (r["backbone"], float(r["snr"]))
        acc[key].append((float(r["hard"]), float(r["readout"])))

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    # ceiling: solid over the exact tier, dashed over the SIS-certified tail
    # (shared point at the boundary keeps the curve visually continuous)
    ax.plot(csnr[:n_ex], cacc[:n_ex], "-", color="black", lw=2.2, zorder=3,
            solid_capstyle="round")
    ax.plot(csnr[n_ex - 1:], cacc[n_ex - 1:], "--", color="black", lw=2.2,
            zorder=3, dashes=(3.2, 1.6))

    for key_bb, _, col in BB:
        snrs = sorted(s for b, s in acc if b == key_bb)
        hard = [sum(h for h, _ in acc[(key_bb, s)]) / len(acc[(key_bb, s)])
                for s in snrs]
        ro = [sum(r for _, r in acc[(key_bb, s)]) / len(acc[(key_bb, s)])
              for s in snrs]
        ax.plot(snrs, hard, "-", color=col, lw=1.5, zorder=2)
        ax.plot(snrs, ro, "--", color=col, lw=0.9, zorder=2, dashes=(4, 2))

    handles = [
        Line2D([], [], color="black", lw=2.2, label="Bayes ceiling (exact)"),
        Line2D([], [], color="black", lw=2.2, ls="--", dashes=(3.2, 1.6),
               label="Bayes ceiling (SIS-certified)"),
        *[Line2D([], [], color=col, lw=1.5, label=name) for _, name, col in BB],
        Line2D([], [], color="#666666", lw=0.9, ls="--", dashes=(4, 2),
               label="readout $F_{\\mathrm{aff}}$ (per backbone)"),
    ]
    ax.legend(handles=handles, fontsize=6.3, frameon=False, loc="upper left",
              borderaxespad=0.3, handlelength=1.9, labelspacing=0.35)

    ax.set_xlabel("SNR (dB)", fontsize=9)
    ax.set_ylabel("accuracy (%)", fontsize=9)
    ax.set_xlim(-20.6, 18.6)
    ax.set_ylim(11, 102)
    ax.set_xticks(range(-20, 19, 4))
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "bayes_ceiling.pdf", bbox_inches="tight")
    print("wrote", out / "bayes_ceiling.pdf")


if __name__ == "__main__":
    main()
