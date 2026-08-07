# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Submission-grade figures for the TCCN paper, unified style (Okabe-Ito colorblind-safe palette,
serif fonts matched to the IEEEtran Times body, editable vector PDF). Builds: teaser schematic,
SNR 2-panel, complexity curve, per-family bars. Detection example is rendered separately on the server.
Run: python paper/tccn/make_figs.py   (outputs paper/tccn/figs/*.pdf)"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

# ---- unified style: serif (match Times body), editable PDF text, thin spines ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 200,
})
# Okabe-Ito colorblind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}
VIS, IQ = OI["blue"], OI["vermillion"]   # vision vs return-to-IQ everywhere


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("SAVED", name + ".pdf")


# ============================ Fig 1 — teaser schematic ============================
def fig_teaser():
    from matplotlib.patches import Rectangle, Circle, Polygon
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(7.2, 2.75))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # synthetic time-frequency scene
    H, W = 48, 76
    spec = rng.random((H, W)) * 0.22
    blocks = [(8, 14, 6, 30), (27, 32, 34, 68), (36, 41, 3, 21), (17, 22, 48, 72)]
    for f0, f1, t0, t1 in blocks:
        spec[f0:f1, t0:t1] += rng.uniform(0.62, 0.92)

    panel = lambda cx, cy, w, h: fig.add_axes([cx - w / 2, cy - h / 2, w, h])

    def show_spec(a, boxes=False, tags=False):
        a.axis("off")
        a.imshow(spec, aspect="auto", cmap="magma", origin="lower", interpolation="nearest")
        if boxes:
            for f0, f1, t0, t1 in blocks:
                a.add_patch(Rectangle((t0 - 0.5, f0 - 0.5), t1 - t0, f1 - f0, edgecolor=OI["green"],
                            facecolor="none", lw=1.0, zorder=3))
            a.set_xlim(-0.5, W - 0.5); a.set_ylim(-0.5, H - 0.5)
        if tags:  # vision-label tags: the detector outputs a class with each box
            import matplotlib.patheffects as pe
            halo = [pe.withStroke(linewidth=1.1, foreground="black")]
            a.text(6, 14.6, "fsk", fontsize=5.2, color=OI["green"], va="bottom", zorder=4,
                   path_effects=halo)
            a.text(34, 32.6, "psk", fontsize=5.2, color=OI["green"], va="bottom", zorder=4,
                   path_effects=halo)

    def draw_nn(a, node_fc, node_ec):
        a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        layers, sizes = [0.13, 0.38, 0.63, 0.88], [3, 4, 4, 2]
        coords = [[(lx, yy) for yy in np.linspace(0.2, 0.8, n)] for lx, n in zip(layers, sizes)]
        for li in range(len(coords) - 1):
            for (x0, y0) in coords[li]:
                for (x1, y1) in coords[li + 1]:
                    a.plot([x0, x1], [y0, y1], color=OI["grey"], lw=0.4, alpha=0.6, zorder=1)
        for layer in coords:
            for (x0, y0) in layer:
                a.add_patch(Circle((x0, y0), 0.055, facecolor=node_fc, edgecolor=node_ec, lw=0.6, zorder=2))

    def rbox(cx, cy, w, h, text, ec, fc, fs=6.2, tc="black"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.004,rounding_size=0.018", linewidth=1.0,
                     edgecolor=ec, facecolor=fc, zorder=3))
        if text:  # optical centering: two-line blocks sit ~0.008 high without the nudge
            ax.text(cx, cy - 0.008, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

    def harrow(x0, x1, y, color=OI["black"], lw=1.1):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                     lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=5))

    # ---------------- top lane: the pure-vision recipe ----------------
    ymid_t, ph, pw = 0.73, 0.30, 0.115
    cxs = [0.062, 0.200, 0.338, 0.476]
    a1 = panel(cxs[0], ymid_t, pw, ph); a1.axis("off"); a1.set_ylim(-1.28, 1.28)
    t = np.linspace(0, 1, 240); env = np.exp(-((t - 0.5) / 0.4) ** 2)
    a1.plot(t, env * np.sin(2 * np.pi * 8.5 * t), color=OI["sky"], lw=0.9)
    a1.plot(t, env * np.cos(2 * np.pi * 8.5 * t), color=OI["orange"], lw=0.9)
    a1.text(0.02, 0.98, "I", color=OI["sky"], fontsize=6, transform=a1.transAxes, va="top")
    a1.text(0.02, 0.30, "Q", color=OI["orange"], fontsize=6, transform=a1.transAxes, va="top")
    show_spec(panel(cxs[1], ymid_t, pw, ph))
    draw_nn(panel(cxs[2], ymid_t, pw, ph), OI["sky"], OI["blue"])
    show_spec(panel(cxs[3], ymid_t, pw, ph), boxes=True, tags=True)
    for cx, tlab in zip(cxs, ["Raw IQ", "STFT", "Detector", "TF boxes + vision labels"]):
        ax.text(cx, 0.545, tlab, ha="center", va="top", fontsize=7)
    for i in range(3):
        harrow(cxs[i] + pw / 2 + 0.004, cxs[i + 1] - pw / 2 - 0.004, ymid_t)
    ax.text(0.269, 0.945, "localization mAP $0.948$ (easy)", ha="center", va="bottom",
            fontsize=7.6, color=OI["green"], style="italic")

    # decision DIAMOND keyed on the vision label's family (Algorithm 1: family(c_i) in {PSK, ASK})
    sw_cx, sw_hw, sw_hh = 0.630, 0.052, 0.115
    ax.add_patch(Polygon([(sw_cx - sw_hw, ymid_t), (sw_cx, ymid_t + sw_hh),
                          (sw_cx + sw_hw, ymid_t), (sw_cx, ymid_t - sw_hh)],
                 closed=True, facecolor="#FFF6D8", edgecolor="black", lw=1.0, zorder=3))
    ax.text(sw_cx, ymid_t, "family?", ha="center", va="center", fontsize=6.0, zorder=4)
    harrow(cxs[3] + pw / 2 + 0.004, sw_cx - sw_hw - 0.004, ymid_t)
    ax.text(sw_cx, ymid_t + sw_hh + 0.025, r"$\{(b_i,\hat{c}_i)\}$", ha="center", va="bottom",
            fontsize=5.8, color=OI["black"])          # detector output: boxes AND coarse labels

    # spectral-family exit: keep the vision label
    tag_cx, tag_w, tag_h = 0.855, 0.16, 0.17
    rbox(tag_cx, ymid_t, tag_w, tag_h, "keep vision label\nFSK/OFDM/FM", VIS, "#E7F0F7", fs=6.2)
    harrow(sw_cx + sw_hw + 0.004, tag_cx - tag_w / 2 - 0.009, ymid_t, color=VIS)
    ax.text((sw_cx + sw_hw + tag_cx - tag_w / 2) / 2, ymid_t + 0.030, "FSK/OFDM/FM",
            ha="center", va="bottom", fontsize=5.6, color=VIS)

    # ---------------- bottom lane: the return-to-IQ branch ----------------
    ymid_b = 0.24
    ch_cx, ch_w, ch_h = 0.295, 0.165, 0.17
    con_cx, rec_cx = cxs[3], 0.635          # constellation aligned to the TF-boxes panel above
    rbox(ch_cx, ymid_b, ch_w, ch_h, "", IQ, "#FBEAE0")
    ax.text(ch_cx, ymid_b + 0.033, "channelize", ha="center", va="center", fontsize=6.2, zorder=4)
    ax.text(ch_cx, ymid_b - 0.030, r"$x\,e^{-j2\pi f_c n/f_s}\!\to\mathrm{LPF}(B)\to\,\downarrow\!D$",
            ha="center", va="center", fontsize=5.6, zorder=4)
    ac = panel(con_cx, ymid_b, pw, ph); ac.axis("off"); ac.set_xlim(-1.7, 1.7); ac.set_ylim(-1.7, 1.7)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ac.scatter(sx + 0.12 * rng.standard_normal(16), sy + 0.12 * rng.standard_normal(16),
                       s=3.0, color=IQ, alpha=0.9, linewidths=0)
    draw_nn(panel(rec_cx, ymid_b, pw, ph), "#F9CDB0", IQ)
    for cx, tlab in zip([con_cx, rec_cx], ["Baseband IQ $x_b$", "IQ recognizer"]):
        ax.text(cx, 0.055, tlab, ha="center", va="top", fontsize=7)
    rbox(tag_cx, ymid_b, tag_w, tag_h, "IQ label\nPSK/ASK/QAM", IQ, "#FBEAE0", fs=6.2)
    harrow(ch_cx + ch_w / 2 + 0.009, con_cx - pw / 2 - 0.004, ymid_b, color=IQ)
    harrow(con_cx + pw / 2 + 0.004, rec_cx - pw / 2 - 0.004, ymid_b, color=IQ)
    harrow(rec_cx + pw / 2 + 0.004, tag_cx - tag_w / 2 - 0.009, ymid_b, color=IQ)

    # the RETURN path: raw IQ tapped from the far left down into the channelizer
    lw_b = 1.15
    ax.plot([cxs[0], cxs[0]], [0.505, ymid_b], color=OI["grey"], lw=lw_b,
            solid_capstyle="round", solid_joinstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch((cxs[0], ymid_b), (ch_cx - ch_w / 2 - 0.009, ymid_b),
                 arrowstyle="-|>", mutation_scale=8, lw=lw_b, color=OI["grey"], shrinkA=0, shrinkB=0, zorder=2))
    ax.text(cxs[0] + 0.013, 0.42, r"$x[n]$", ha="left", va="center",
            fontsize=6.0, color=OI["grey"])

    # PSK/ASK/QAM exit: the routed boxes' parameters drive the channelizer.
    # One dashed polyline (single dash phase through both corners, mitred joins);
    # the arrowhead rides a short SOLID stub so the head triangle stays intact.
    y_gap = 0.47
    ax.plot([sw_cx, sw_cx, ch_cx, ch_cx],
            [ymid_t - sw_hh - 0.012, y_gap, y_gap, ymid_b + ch_h / 2 + 0.026],
            color=IQ, lw=1.0, ls=(0, (3, 2)), solid_joinstyle="miter", zorder=2)
    ax.add_patch(FancyArrowPatch((ch_cx, ymid_b + ch_h / 2 + 0.028), (ch_cx, ymid_b + ch_h / 2 + 0.008),
                 arrowstyle="-|>", mutation_scale=8, lw=1.0, color=IQ,
                 shrinkA=0, shrinkB=0, zorder=2))
    ax.text(sw_cx + 0.012, 0.555, "PSK/ASK/QAM", ha="left", va="center",
            fontsize=5.6, color=IQ)
    ax.text((ch_cx + sw_cx) / 2, y_gap - 0.028, r"$\{(f_{c,i},\,B_i)\}$", ha="center", va="top",
            fontsize=5.8, color=IQ)

    ax.text(tag_cx, 0.45, "recognition $\\sim\\!0.5$ (hard)", ha="center", va="center",
            fontsize=7.6, color=OI["black"], style="italic")
    save(fig, "fig1_teaser")


# ============================ Fig 2 — SNR two-panel (full width) ============================
def fig_snr():
    rows = list(csv.DictReader(open(os.path.join(HERE, "snr_data.csv"))))
    snr = [(float(r["blocksnr_lo"]) + float(r["blocksnr_hi"])) / 2 for r in rows]
    col = lambda k: [float(r[k]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.55))

    a.plot(snr, col("loc_recall"), color=OI["green"], marker="o", label="localization recall")
    a.plot(snr, col("vision_recog"), color=VIS, marker="s", label="recognition: vision")
    a.plot(snr, col("IQ_pred"), color=IQ, marker="^", label="recognition: return-to-IQ")
    a.plot(snr, col("IQ_perfect"), color=IQ, marker="^", ls=(0, (4, 2)), alpha=0.7,
           label="return-to-IQ (perfect box)")
    a.set_xlabel("corrected block-SNR (dB)"); a.set_ylabel("accuracy / recall")
    a.set_ylim(0, 1.05); a.grid(True, color=OI["grey"], alpha=0.25, lw=0.5)
    a.legend(loc="lower right", handlelength=1.6)   # bottom band is empty in both panels
    a.text(-0.16, 1.02, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold")

    b.plot(snr, col("constel_vision"), color=VIS, marker="s", label="constellation: vision")
    b.plot(snr, col("constel_IQ"), color=IQ, marker="^", label="constellation: return-to-IQ")
    b.set_xlabel("corrected block-SNR (dB)"); b.set_ylabel("recognition accuracy")
    b.set_ylim(0, 0.55); b.grid(True, color=OI["grey"], alpha=0.25, lw=0.5)
    b.legend(loc="lower right", handlelength=1.6)
    b.text(-0.16, 1.02, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(pad=0.5)
    save(fig, "fig2_snr")


# ============================ Fig 4 — complexity curve ============================
def fig_complexity():
    sizes = ["tiny", "small", "medium", "large"]
    mAP = [0.431, 0.449, 0.460, 0.462]        # uniform recipe, 3-seed (7/17/27) means
    mAP_sd = [0.010, 0.010, 0.011, 0.015]     # 3-seed standard deviations
    own = [0.408, 0.429, 0.477, 0.486]        # own gentler schedule (lr 1e-4), seed means
    own_sd = [0.041, 0.017, 0.039, 0.005]     # n=2/3/3/2 seeds
    APs = [0.192, 0.226, 0.245, 0.249]        # uniform seed 7
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    h1 = ax.errorbar(x, mAP, yerr=mAP_sd, color=VIS, marker="o", capsize=2.5, lw=1.2,
                     label="mAP, uniform recipe (3 seeds)")
    h2 = ax.errorbar(x, own, yerr=own_sd, color=OI["green"], marker="D", ms=4.0, ls=(0, (4, 2)),
                     capsize=2.5, lw=1.2, label="mAP, own schedule (2–3 seeds)")
    h3, = ax.plot(x, APs, color=IQ, marker="s", label=r"small-object AP$_s$")
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("detector size"); ax.set_ylabel("class-aware mAP")
    ax.set_ylim(0.15, 0.56); ax.grid(True, color=OI["grey"], alpha=0.25, lw=0.5)
    # legend compacted into the empty mid-band between the mAP cluster (~0.41-0.49) and AP_s (~0.19-0.25)
    ax.legend(handles=[h2, h1, h3], loc="center", bbox_to_anchor=(0.5, 0.45),
              handlelength=1.6, fontsize=6.0, labelspacing=0.3, handletextpad=0.5,
              borderpad=0.3, borderaxespad=0.2)
    save(fig, "fig4_complexity")


# ============================ Fig 5 — per-family bars ============================
def fig_family():
    fams = ["PSK", "ASK", "QAM", "FSK", "FM", "OFDM"]
    vis = [0.309, 0.233, 0.133, 0.750, 0.863, 0.484]
    iq = [0.451, 0.350, 0.217, 0.696, 0.636, 0.290]
    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ax.bar(x - w / 2, vis, w, color=VIS, edgecolor="black", linewidth=0.5, label="vision")
    ax.bar(x + w / 2, iq, w, color=IQ, edgecolor="black", linewidth=0.5, label="return-to-IQ")
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel("class-aware mAP"); ax.set_ylim(0, 0.95)
    ax.grid(True, axis="y", color=OI["grey"], alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", handlelength=1.6, ncol=1)  # PSK/ASK bars are short -> corner is free
    # one unified bracket over the three constellation families the router sends to the IQ branch
    x0, x1, yb = x[0] - w / 2 - 0.04, x[2] + w / 2 + 0.04, 0.53
    ax.plot([x0, x1], [yb, yb], color=IQ, lw=1.0, clip_on=False)
    ax.plot([x0, x0], [yb - 0.025, yb], color=IQ, lw=1.0, clip_on=False)
    ax.plot([x1, x1], [yb - 0.025, yb], color=IQ, lw=1.0, clip_on=False)
    ax.text((x0 + x1) / 2, yb + 0.012, "routed to IQ", ha="center", va="bottom",
            fontsize=6, color=IQ, fontweight="bold")
    save(fig, "fig5_family")


if __name__ == "__main__":
    fig_teaser(); fig_snr(); fig_complexity(); fig_family()
    print("ALL_FIGS_DONE")
