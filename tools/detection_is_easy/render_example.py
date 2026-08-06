# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Render a clean, publication-style qualitative detection example: a test scene's STFT log-magnitude
with ground-truth boxes (green) and the detector's predicted boxes (red dashed), with a legend and
readable class labels. No debug title. Saves a single-column PNG for the paper.
Run on server: python tools/detection_is_easy/render_example.py"""
import sys, json, numpy as np
from collections import defaultdict
sys.path.insert(0, ".")
sys.path.insert(0, "tools/detection_is_easy")
from run_mmdet_smoke import maybe_stub_mmcv_ext
maybe_stub_mmcv_ext()
from mmdet_plugins import LoadTensorMemmapFromCOCOStem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = "data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass"
MM = "data/torchsig_hardshort_lowsnr_stft3_memmap/memmap"
ann = json.load(open(f"{ROOT}/annotations/instances_test.json"))
cats = {c["id"]: c["name"] for c in ann["categories"]}
imgs = {im["id"]: im for im in ann["images"]}
by_img = defaultdict(list)
for a in ann["annotations"]:
    by_img[a["image_id"]].append(a)
preds_by_img = defaultdict(list)
PRED = "work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump/source_data/test_predictions.bbox.json"
try:
    for p in json.load(open(PRED)):
        preds_by_img[p["image_id"]].append(p)
except Exception:
    pass

# prefer a scene with 3-4 signals that are reasonably separated
pick = next((iid for iid, a in by_img.items() if 3 <= len(a) <= 4), next(iter(by_img)))
im = imgs[pick]
gts = by_img[pick]
loader = LoadTensorMemmapFromCOCOStem(memmap_root=MM, expected_channels=3)
res = loader.transform({"img_path": f"test/tensors/{im['file_name']}"})
img = np.asarray(res["img"])
img = img if (img.ndim == 3 and img.shape[-1] == 3) else np.transpose(img, (1, 2, 0))
H, W = img.shape[:2]
logmag = img[..., 2]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
})
fig, ax = plt.subplots(figsize=(3.4, 3.05), dpi=300)
ax.imshow(logmag, aspect="auto", origin="upper", cmap="magma")
GT_C, PR_C = "#00e5a0", "#ff5b4d"  # bright green / orange-red, high contrast on magma
lbl_bbox = dict(facecolor="black", alpha=0.45, pad=0.6, edgecolor="none")

for a in gts:
    x, y, w, h = a["bbox"]
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=GT_C, lw=1.6))
    ax.text(x + 1, max(y - 2, 6), cats[a["category_id"]], color=GT_C, fontsize=7,
            va="bottom", ha="left", bbox=lbl_bbox)
# predicted boxes: top-scoring, drawn dashed; label only if it differs from a nearby GT (illustrate the gap)
for p in sorted(preds_by_img.get(pick, []), key=lambda d: -d["score"])[:len(gts)]:
    if p["score"] < 0.25:
        continue
    x, y, w, h = p["bbox"]
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=PR_C, lw=1.2, ls=(0, (4, 2))))
    # pred label below the box; but for boxes near the bottom edge, put it above-right (GT sits
    # above-left, so no collision) so the label background never clips at the axis.
    if y + h > 0.80 * H:
        ly, pva = max(y - 2, 6), "bottom"
    else:
        ly, pva = y + h + 2, "top"
    ax.text(x + w - 1, ly, cats.get(p["category_id"] + 1, "?"), color=PR_C,
            fontsize=7, va=pva, ha="right", bbox=lbl_bbox)

ax.set_xlabel("time"); ax.set_ylabel("frequency")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=[Line2D([0], [0], color=GT_C, lw=1.8, label="ground truth"),
                   Line2D([0], [0], color=PR_C, lw=1.4, ls=(0, (4, 2)), label="prediction")],
          loc="upper right", fontsize=7, framealpha=0.8)
plt.tight_layout(pad=0.3)
plt.savefig("work_dirs/detection_example.pdf", bbox_inches="tight")
plt.savefig("work_dirs/detection_example.png", bbox_inches="tight", dpi=300)
print(f"SAVED detection_example.pdf/png  scene={pick} n_gt={len(gts)}")
