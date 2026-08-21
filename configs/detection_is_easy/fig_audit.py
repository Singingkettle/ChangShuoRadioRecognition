# -*- coding: utf-8 -*-
"""Programmatic overlap audit for every figure produced by make_figs.py.

For each figure it re-runs the figure function with `save` patched, draws the canvas, and then
checks, in display coordinates:
  * text vs text overlaps (tick labels, axis labels, annotations, legend entries, titles),
  * text/legend boxes vs data lines (sampled along every Line2D segment) and vs bar patches,
  * legend box vs data lines / markers,
  * any text or legend that leaves the figure canvas.
Run: python fig_audit.py            (prints a report; exit 1 if any overlap is found)
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs as M  # noqa: E402

REPORT = []


def _bbox(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer=renderer)
    except Exception:
        return None
    if bb is None or not np.isfinite([bb.x0, bb.x1, bb.y0, bb.y1]).all() or bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def _shrink(bb, pad):
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def _line_points(line, n_per_seg=12):
    """Sample points along a Line2D in display coords (markers + segments)."""
    ax = line.axes
    xy = np.asarray(line.get_xydata(), dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return np.empty((0, 2))
    disp = ax.transData.transform(xy)
    pts = [disp]
    if line.get_linestyle() not in ("None", "none", " ", ""):
        for a, b in zip(disp[:-1], disp[1:]):
            t = np.linspace(0, 1, n_per_seg)[1:-1]
            pts.append(a[None, :] + (b - a)[None, :] * t[:, None])
    return np.vstack(pts)


def audit(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    issues = []

    texts = []   # (label, bbox)
    for ax in fig.axes:
        if not ax.axison:          # decorative panels (axis off): no ticks/labels are drawn
            for t in ax.texts:
                bb = _bbox(t, r)
                if bb is not None and t.get_text().strip() and t.get_visible():
                    texts.append((f"text '{t.get_text()[:28]}'", bb))
            continue
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if isinstance(t, Text) and t.get_text().strip() and t.get_visible():
                bb = _bbox(t, r)
                if bb is not None:
                    texts.append((f"text '{t.get_text()[:28]}'", bb))
        xlo, xhi = sorted(ax.get_xlim()); ylo, yhi = sorted(ax.get_ylim())
        xt = [(l, v) for l, v in zip(ax.get_xticklabels(), ax.get_xticks()) if xlo - 1e-9 <= v <= xhi + 1e-9]
        yt = [(l, v) for l, v in zip(ax.get_yticklabels(), ax.get_yticks()) if ylo - 1e-9 <= v <= yhi + 1e-9]
        for lab, _ in xt + yt:       # only ticks inside the view limits are actually drawn
            if lab.get_text().strip() and lab.get_visible():
                bb = _bbox(lab, r)
                if bb is not None:
                    texts.append((f"tick '{lab.get_text()}'", bb))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            bb = _bbox(leg, r)
            if bb is not None:
                texts.append(("legend", bb))
    for t in fig.texts:
        bb = _bbox(t, r)
        if bb is not None and t.get_text().strip():
            texts.append((f"figtext '{t.get_text()[:28]}'", bb))

    # 1) text vs text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = _shrink(texts[i][1], 0.6), _shrink(texts[j][1], 0.6)
            if a.overlaps(b):
                # allow legend to contain its own entries: skip pairs where one is the legend and
                # the other lies fully inside it
                if texts[i][0] == "legend" and texts[i][1].contains(b.x0, b.y0) and texts[i][1].contains(b.x1, b.y1):
                    continue
                if texts[j][0] == "legend" and texts[j][1].contains(a.x0, a.y0) and texts[j][1].contains(a.x1, a.y1):
                    continue
                issues.append(f"TEXT/TEXT overlap: {texts[i][0]}  <->  {texts[j][0]}")

    # 2) text / legend vs data lines and bars
    for ax in fig.axes:
        lines = [l for l in ax.get_lines() if l.get_visible()]
        bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_visible() and p.get_width() > 0 and p.get_height() > 0]
        # also errorbar container lines are in ax.get_lines(); fine
        pts_all = [(_line_points(l), l) for l in lines]
        for label, bb in texts:
            if label.startswith("tick"):
                continue  # tick labels live outside the axes area
            bbs = _shrink(bb, 1.0)
            for pts, l in pts_all:
                if len(pts) == 0:
                    continue
                inside = (pts[:, 0] >= bbs.x0) & (pts[:, 0] <= bbs.x1) & (pts[:, 1] >= bbs.y0) & (pts[:, 1] <= bbs.y1)
                if inside.any():
                    # ignore axis-spanning reference lines (axhline/axvline) only when the text was
                    # deliberately placed on them: report anyway but tag them
                    tag = " (reference line)" if l.get_linestyle() in ((0, (3, 2)), "--", "dashed") and len(l.get_xdata()) == 2 else ""
                    issues.append(f"TEXT/LINE overlap: {label} covers data of '{l.get_label()}'{tag}")
                    break
            for p in bars:
                pb = p.get_window_extent(r)
                if bbs.overlaps(_shrink(pb, 0.5)):
                    issues.append(f"TEXT/BAR overlap: {label} covers a bar")
                    break

    # 3) anything outside the canvas
    for label, bb in texts:
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            issues.append(f"OUTSIDE canvas: {label} ({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) canvas {W}x{H}")

    w_in, h_in = fig.get_size_inches()
    REPORT.append((name, w_in, h_in, issues))


def patched_save(fig, name, exact=True):
    audit(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    M.save = patched_save
    for fn in (M.fig_teaser, M.fig_snr, M.fig_complexity, M.fig_predbox_family,
               M.fig_box_injection, M.fig_taxonomy_scatter, M.fig_iou_decile):
        fn()
    bad = 0
    for name, w, h, issues in REPORT:
        print(f"=== {name}  ({w:.3f} x {h:.2f} in) ===")
        if not issues:
            print("  OK: no overlaps")
        for it in sorted(set(issues)):
            print("  " + it)
        bad += len(issues)
    print(f"\nTOTAL issues: {bad}")
    sys.exit(1 if bad else 0)
