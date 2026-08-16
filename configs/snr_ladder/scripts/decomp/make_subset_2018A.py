#!/usr/bin/env python3
"""Build a stratified 25% training subset of RadioML2018.01A.

Only the training annotation is subsampled. validation.json and test.json are left
untouched, so the ladder is still fit and scored on exactly the same data as the
full-data run; the only thing that changes is how much the hard model saw.

Stratification is by (modulation, snr) so every class keeps the same SNR profile.
One fixed seed, one subset file, shared by all three training seeds: we do not want
subset-sampling variance folded into training-seed variance.
"""
import json
import os
from collections import defaultdict

FRAC = 0.25
SEED = 2026
ROOT = ("data/ModulationClassification/"
        "DeepSig/RadioML.2018.01A")
SRC = os.path.join(ROOT, "train.json")
DST = os.path.join(ROOT, f"train_p{int(FRAC * 100)}.json")

import random

d = json.load(open(SRC))
items = d["data_list"]
print(f"source: {len(items)} training samples")

groups = defaultdict(list)
for it in items:
    groups[(it["modulation"], it["snr"])].append(it)
print(f"strata: {len(groups)} (modulation, snr) cells")

rng = random.Random(SEED)
kept = []
sizes = []
for key in sorted(groups):
    g = groups[key]
    k = max(1, int(round(len(g) * FRAC)))
    kept.extend(rng.sample(g, k))
    sizes.append((len(g), k))
kept.sort(key=lambda x: x["file_name"])

out = dict(d)
out["data_list"] = kept
json.dump(out, open(DST, "w"))

per_stratum = sizes[0]
print(f"kept: {len(kept)} ({100.0 * len(kept) / len(items):.2f}%)")
print(f"per stratum: {per_stratum[0]} -> {per_stratum[1]} (first stratum)")
print(f"min/max kept per stratum: {min(s[1] for s in sizes)} / {max(s[1] for s in sizes)}")
print(f"wrote {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")
