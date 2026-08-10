# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Submission-grade figures for the TCCN paper, unified style (Okabe-Ito colorblind-safe palette,
serif fonts matched to the IEEEtran Times body, editable vector PDF). Builds: teaser schematic,
SNR 2-panel, complexity curve, per-family bars. The detection example (Fig. 3) is rendered
separately by render_example.py, which needs the dataset and a prediction dump.

Self-contained: every value these figures plot is either a literal in this file or a row of
snr_data.csv beside it (the committed copy of what analyze_snr_stratified.py emits).

Run: python tools/detection_is_easy/make_figs.py   (writes tools/detection_is_easy/figs/*.pdf)"""
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
# IEEEtran journal geometry: \textwidth = 43pc and \columnwidth = 21pc, in TeX points
# (1 TeX pt = 1/72.27 in). Figures are saved at exactly these widths so LaTeX applies a
# scale factor of 1.0 and a nominal 8 pt label really prints at 8 pt.
W_DOUBLE = 43 * 12 / 72.27      # 7.1399 in
W_SINGLE = 21 * 12 / 72.27      # 3.4869 in

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6,
    # thinner strokes and smaller markers; markeredgewidth 0 removes matplotlib's
    # default 1.0 pt halo, which silently inflated every marker by 1 pt of diameter.
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "patch.linewidth": 0.4,
    # one grid definition for every figure
    "axes.axisbelow": True,
    "grid.color": "#B3B3B3", "grid.linewidth": 0.4, "grid.alpha": 0.35,
    # a faint opaque legend box keeps gridlines and near-miss curves out of the labels
    "legend.frameon": True, "legend.facecolor": "white", "legend.framealpha": 0.85,
    "legend.edgecolor": "none", "legend.fancybox": False,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.35, "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.4, "legend.columnspacing": 1.2,
    "figure.dpi": 200,
})
ERRBAR = dict(elinewidth=0.8, capsize=1.5, capthick=0.8)
# Okabe-Ito colorblind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}
VIS, IQ = OI["blue"], OI["vermillion"]   # vision vs return-to-IQ everywhere


def save(fig, name, exact=True):
    """exact=True saves at precisely `figsize`, so the PDF's width already equals the
    printed width. `bbox_inches="tight"` is deliberately avoided there: it trims to the
    ink, which made the saved width differ from figsize by up to 10% and left LaTeX to
    rescale each figure by a different factor (the reason the five figures did not look
    like one set). fig1 keeps the tight box because its content is drawn to the figure
    edges by hand."""
    path = os.path.join(FIG, name + ".pdf")
    if exact:
        fig.savefig(path)
    else:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
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
            a.text(6, 14.6, "FSK", fontsize=5.2, color=OI["green"], va="bottom", zorder=4,
                   path_effects=halo)
            a.text(34, 32.6, "PSK", fontsize=5.2, color=OI["green"], va="bottom", zorder=4,
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
    save(fig, "fig1_teaser", exact=False)


# ============================ Fig 2 — SNR two-panel (full width) ============================
def fig_snr():
    rows = list(csv.DictReader(open(os.path.join(HERE, "snr_data.csv"))))
    snr = [(float(r["blocksnr_lo"]) + float(r["blocksnr_hi"])) / 2 for r in rows]
    col = lambda k: [float(r[k]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.55), layout="constrained")

    a.plot(snr, col("loc_recall"), color=OI["green"], marker="o", label="localization recall")
    a.plot(snr, col("vision_recog"), color=VIS, marker="s", label="recognition: vision head")
    a.plot(snr, col("IQ_pred"), color=IQ, marker="^", label="recognition: return-to-IQ")
    a.plot(snr, col("IQ_perfect"), color=IQ, marker="^", ls=(0, (4, 2)), alpha=0.7,
           label="return-to-IQ (perfect-box)")
    a.set_xlabel("corrected block-SNR (dB)")
    # (a) carries one recall curve and three accuracy curves; (b) carries accuracy only.
    # The shared noun is "accuracy", and the legend names which curve is which.
    a.set_ylabel("recall / accuracy")
    a.set_ylim(0, 1.05); a.grid(True)
    a.legend(loc="lower left")      # 12.4 pt of clearance; "lower right" grazed the vision tail
    a.set_title("(a)", loc="left", fontsize=8, fontweight="bold", pad=2)

    b.plot(snr, col("constel_vision"), color=VIS, marker="s", label="vision head")
    b.plot(snr, col("constel_IQ"), color=IQ, marker="^", label="return-to-IQ")
    b.set_xlabel("corrected block-SNR (dB)"); b.set_ylabel("accuracy")
    b.set_ylim(0, 0.55); b.grid(True)
    b.legend(loc="lower right", title="constellation families", title_fontsize=7)
    b.set_title("(b)", loc="left", fontsize=8, fontweight="bold", pad=2)
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
    # The two mAP series share an x position, so their markers and whiskers merged (at
    # "medium" the two markers were 0.15 pt apart). Dodge them by +-0.08 index units.
    d = 0.08
    # Colour: this figure has no vision/return-to-IQ semantics, so it must not reuse the
    # blue/vermillion pair that means those elsewhere. The two mAP curves share one hue
    # family (same metric, different recipe); AP_s takes a distinct hue.
    C_UNI, C_OWN, C_APS = OI["blue"], OI["sky"], OI["purple"]
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.53), layout="constrained")
    h1 = ax.errorbar(x - d, mAP, yerr=mAP_sd, color=C_UNI, marker="o",
                     label="uniform recipe", **ERRBAR)
    h2 = ax.errorbar(x + d, own, yerr=own_sd, color=C_OWN, marker="D", ls=(0, (4, 2)),
                     label="own schedule", **ERRBAR)
    h3, = ax.plot(x, APs, color=C_APS, marker="s", label=r"small-object AP$_s$")
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_xlabel("detector size"); ax.set_ylabel("class-aware mAP")
    ax.set_ylim(0.15, 0.56)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5])          # match the tick density of Figs. 2 and 5
    ax.grid(True, axis="y")                       # x is categorical: vertical gridlines say nothing
    # The band 0.25 < y < 0.37 is empty across the whole x-range (upper envelope of AP_s
    # is 0.249, lower envelope of the mAP whiskers is 0.367): the legend goes there.
    ax.legend(handles=[h1, h2, h3], loc="center right", bbox_to_anchor=(1.0, 0.40),
              fontsize=6.5)
    save(fig, "fig4_complexity")


# ============================ Fig 5 — per-family bars ============================
def fig_family():
    fams = ["PSK", "ASK", "QAM", "FSK", "FM", "OFDM"]
    vis = [0.309, 0.233, 0.133, 0.750, 0.863, 0.484]
    iq = [0.451, 0.350, 0.217, 0.696, 0.636, 0.290]
    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.33), layout="constrained")
    ax.bar(x - w / 2, vis, w, color=VIS, edgecolor="black", label="vision head")
    ax.bar(x + w / 2, iq, w, color=IQ, edgecolor="black", label="return-to-IQ")
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel("class-aware mAP"); ax.set_ylim(0, 0.95)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=1)   # PSK/ASK bars are short -> the corner is free
    # One bracket over the three constellation families the router sends to the IQ branch.
    # bar() centres each rectangle on its x, so a group spans x-w .. x+w -- not x-w/2 .. x+w/2.
    # The old half-width put each bracket end 4.5 pt INSIDE the first and last bar.
    x0, x1, yb = x[0] - w - 0.04, x[2] + w + 0.04, 0.53
    ax.plot([x0, x1], [yb, yb], color=IQ, lw=0.9, clip_on=False)
    ax.plot([x0, x0], [yb - 0.025, yb], color=IQ, lw=0.9, clip_on=False)
    ax.plot([x1, x1], [yb - 0.025, yb], color=IQ, lw=0.9, clip_on=False)
    ax.text((x0 + x1) / 2, yb + 0.012, "routed to IQ", ha="center", va="bottom",
            fontsize=6.5, color=IQ, fontweight="bold")
    save(fig, "fig5_family")


if __name__ == "__main__":
    fig_teaser(); fig_snr(); fig_complexity(); fig_family()
    print("ALL_FIGS_DONE")
