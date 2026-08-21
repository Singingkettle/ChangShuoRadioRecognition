# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Render the qualitative detection example (Fig. 3): one test scene's STFT log-magnitude
with ground-truth boxes and the detector's predictions, each carrying its class label.

Two corrections over the first version of this script:

1. The predicted class was looked up as ``cats[category_id + 1]``. mmdet's ``CocoMetric``
   dumps the *real* COCO category id, and this dataset's ids are 0-based, so the offset
   shifted every predicted label by one class. Measured over the whole test split, the
   no-offset convention agrees with the ground truth on 47.4% of matched boxes (which is
   the paper's recognition level) while the +1 convention agrees on 5.8% (chance). The
   offset is gone; the labels below are the detector's actual output.
2. Colours now follow the paper's contract: green = ground truth, blue = the vision
   head's prediction (blue is the vision path throughout the paper). The previous red
   contradicted Figs. 1, 2 and 5, where red/vermillion means the return-to-IQ branch.

The background spectrogram is recomputed from the scene's raw IQ with the same STFT the
dataset export used, so it is geometrically identical to the stored tensor.

Three inputs are needed; all default to the repo-relative layout and can be pointed
elsewhere on the command line:

    python configs/detection_is_easy/render_example.py \\
      --ann  <memmap-root>/coco_multiclass/annotations/instances_test.json \\
      --raw  <raw-root>/raw/test \\
      --pred work_dirs/<detector-run>_testdump/source_data/test_predictions.bbox.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def _repo_root():
    _p = Path(__file__).resolve()
    for _up in [_p, *_p.parents]:
        if (_up / "tools" / "train.py").exists() and (_up / "csrr").is_dir():
            return _up
    raise RuntimeError("CSRR repo root not found above " + str(_p))


ROOT = _repo_root()

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ann", default=str(ROOT / "data/torchsig_hardshort_lowsnr_stft3_memmap"
                                            "/coco_multiclass/annotations/instances_test.json"),
                help="57-class COCO annotations for the test split")
ap.add_argument("--raw", default=str(ROOT / "data/torchsig_hardshort_lowsnr_iq_65k_nvme/raw/test"),
                help="directory of raw test scenes (<stem>.npz)")
ap.add_argument("--pred", default=str(ROOT / "work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump"
                                             "/source_data/test_predictions.bbox.json"),
                help="detector prediction dump (run_mmdet_train_eval.py --eval-only --dump-results)")
ap.add_argument("--out", default=str(ROOT / "fig3_example.pdf"), help="output PDF")
args = ap.parse_args()

ANN, RAW, PRED = args.ann, args.raw, args.pred
N_FFT = HOP = IMG = 512

# paper colour contract: green = ground truth, blue = vision head
GT_C, PR_C = "#009E73", "#0072B2"
W_SINGLE = 21 * 12 / 72.27      # \columnwidth = 21pc, in inches

ann = json.load(open(ANN))
cats = {c["id"]: c["name"] for c in ann["categories"]}
imgs = {im["id"]: im for im in ann["images"]}
by_img = defaultdict(list)
for a in ann["annotations"]:
    by_img[a["image_id"]].append(a)

preds_by_img = defaultdict(list)
for p in json.load(open(PRED)):
    preds_by_img[p["image_id"]].append(p)

# The scene is chosen by a fixed, stated rule: the first test scene (by image id) with 3-5
# ground-truth signals whose top-n predictions each overlap a distinct signal at IoU >= 0.75
# and carry exactly one wrong label. The paper's figure is that scene (8psk read as 64psk).
def _iou(a, b):
    ax_, ay_, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax_ + aw, bx + bw) - max(ax_, bx))
    iy = max(0.0, min(ay_ + ah, by + bh) - max(ay_, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def _pick_scene():
    for iid in sorted(by_img):
        gts_ = by_img[iid]
        if not 3 <= len(gts_) <= 5:
            continue
        preds_ = sorted(preds_by_img.get(iid, []), key=lambda d: -d["score"])[:len(gts_)]
        if len(preds_) < len(gts_):
            continue
        used, wrong, ok = set(), 0, True
        for p in preds_:
            best, bj = 0.0, None
            for j, g in enumerate(gts_):
                if j in used:
                    continue
                v = _iou(p["bbox"], g["bbox"])
                if v > best:
                    best, bj = v, j
            if best < 0.75:
                ok = False
                break
            used.add(bj)
            wrong += cats[p["category_id"]] != cats[gts_[bj]["category_id"]]
        if ok and wrong == 1:
            return iid
    raise SystemExit("no test scene satisfies the selection rule")


pick = _pick_scene()
im, gts = imgs[pick], by_img[pick]

stem = im["file_name"].rsplit(".", 1)[0]
z = np.load(f"{RAW}/{stem}.npz")
iq = np.asarray(z[z.files[0]]).ravel().astype(np.complex64)

# Same framing, window and orientation as the dataset export (n_fft = hop = 512, Hann,
# fftshift, transpose then flip), but kept as log(1+|X|) -- the very channel the detector
# reads from the stored STFT3 tensor -- instead of the export's percentile-stretched uint8.
# With N = 262144 the result is already 512 x 512, so no resampling is involved.
starts = np.arange(0, max(1, iq.size - N_FFT + 1), HOP)
window = np.hanning(N_FFT).astype(np.float32)
frames = np.stack([iq[s:s + N_FFT] * window for s in starts], axis=0)
spec = np.fft.fftshift(np.fft.fft(frames, n=N_FFT, axis=1), axes=1)
logmag = np.log1p(np.abs(spec)).T[::-1, :]
assert logmag.shape == (IMG, IMG), logmag.shape
H, W = logmag.shape

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "legend.frameon": True, "legend.facecolor": "white",
    "legend.framealpha": 0.85, "legend.edgecolor": "none", "legend.handlelength": 1.4,
})
LW_GT, LW_PR = 1.0, 1.0          # the legend key below uses these exact widths

fig, ax = plt.subplots(figsize=(W_SINGLE, 3.10), layout="constrained")
# interpolation="none" embeds the 512 x 512 array at native resolution instead of resampling
# it to the figure dpi (which silently produced a ~100 dpi raster in the PDF).
ax.imshow(logmag, aspect="auto", origin="upper", cmap="magma", interpolation="none")
lbl_bbox = dict(facecolor="black", alpha=0.45, pad=0.6, edgecolor="none")

for a in gts:
    x, y, w, h = a["bbox"]
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=GT_C, lw=LW_GT))
    ax.text(x + 1, max(y - 2, 6), cats[a["category_id"]], color=GT_C, fontsize=6.5,
            va="bottom", ha="left", bbox=lbl_bbox)

for p in sorted(preds_by_img.get(pick, []), key=lambda d: -d["score"])[:len(gts)]:
    if p["score"] < 0.25:
        continue
    x, y, w, h = p["bbox"]
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=PR_C, lw=LW_PR,
                                    ls=(0, (4, 2))))
    # prediction labels go below-right of the box (GT labels sit above-left), unless the box
    # touches the bottom edge, in which case they go above-right so the label never clips
    if y + h + 14 > H:
        ly, pva = max(y - 2, 6), "bottom"
    else:
        ly, pva = y + h + 2, "top"
    ax.text(x + w - 1, ly, cats[p["category_id"]], color=PR_C, fontsize=6.5,
            va=pva, ha="right", bbox=lbl_bbox)

ax.set_xlabel("time"); ax.set_ylabel("frequency")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=[Line2D([0], [0], color=GT_C, lw=LW_GT, label="ground truth"),
                   Line2D([0], [0], color=PR_C, lw=LW_PR, ls=(0, (4, 2)), label="prediction")],
          loc="upper right")
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=400)
n_wrong = sum(1 for a in gts
              if cats[a["category_id"]] not in
              [cats[p["category_id"]] for p in sorted(preds_by_img.get(pick, []),
                                                      key=lambda d: -d["score"])[:len(gts)]])
print(f"SAVED {out} (+ .png)  scene={pick} ({im['file_name']}) "
      f"n_gt={len(gts)} labels_not_matched={n_wrong}")
