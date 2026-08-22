#!/usr/bin/env python
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0.
"""Same-prediction COCO AP (useCats=1 vs useCats=0) with a scene-paired bootstrap.

For every prediction dump the identical detection records are evaluated twice by the
standard pycocotools evaluator (categories retained / ignored).  The point estimates are
taken verbatim from COCOeval.stats[0].  The uncertainty comes from a scene-level bootstrap:
scenes (COCO image ids) are resampled with replacement, the SAME resample is applied to
both evaluations and to every seed (paired), and AP is re-accumulated from the cached
per-image match matrices produced by COCOeval.evaluate().

Implementation note.  Re-accumulation uses per-scene multiplicities as weights on the
cumulative TP/FP counts and on the positive count.  This is exactly equivalent to
duplicating the resampled scenes and re-running COCOeval.accumulate() (a duplicated
detection record contributes adjacent identical TP/FP steps; the precision envelope and the
left-sided recall look-up then give the same interpolated precision), except for the order of
exact score ties across different scenes.  With weights all equal to one the routine must
reproduce COCOeval.stats[0]; that identity is asserted before any bootstrap is run.

    python same_pred_bootstrap.py --annotation instances_test.json \
        --prediction 20262811=/path/test_predictions.bbox.json [--prediction SEED=PATH ...] \
        --resamples 2000 --seed 20260821 --output same_pred_bootstrap.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(gt: COCO, dt: COCO, use_cats: int):
    sink = StringIO()
    ev = COCOeval(gt, dt, "bbox")
    ev.params.useCats = use_cats
    ev.params.imgIds = sorted(gt.getImgIds())
    with redirect_stdout(sink):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    p = ev.params
    cat_ids = p.catIds if use_cats else [-1]
    K, A, I = len(cat_ids), len(p.areaRng), len(p.imgIds)
    max_det = p.maxDets[-1]
    cats = []
    for k in range(K):
        scores, dtm, dtig, img_of_det = [], [], [], []
        npig_img = np.zeros(I, dtype=np.float64)
        for i in range(I):
            e = ev.evalImgs[k * A * I + 0 * I + i]  # area range 0 == 'all'
            if e is None:
                continue
            gtig = np.asarray(e["gtIgnore"])
            npig_img[i] = np.count_nonzero(gtig == 0)
            s = np.asarray(e["dtScores"][:max_det], dtype=np.float64)
            if s.size == 0:
                continue
            scores.append(s)
            dtm.append(np.asarray(e["dtMatches"])[:, :max_det])
            dtig.append(np.asarray(e["dtIgnore"])[:, :max_det].astype(bool))
            img_of_det.append(np.full(s.size, i, dtype=np.int64))
        if not scores:
            cats.append(None)
            continue
        scores = np.concatenate(scores)
        dtm = np.concatenate(dtm, axis=1)
        dtig = np.concatenate(dtig, axis=1)
        img_of_det = np.concatenate(img_of_det)
        order = np.argsort(-scores, kind="mergesort")
        matched = dtm[:, order] > 0
        ignored = dtig[:, order]
        cats.append({
            "tps": np.logical_and(matched, ~ignored).astype(np.float64),
            "fps": np.logical_and(~matched, ~ignored).astype(np.float64),
            "img": img_of_det[order],
            "npig_img": npig_img,
        })
    return ev, cats, np.asarray(p.recThrs), I


def ap_weighted(cats, rec_thrs: np.ndarray, w: np.ndarray) -> float:
    R = rec_thrs.size
    values = []
    for c in cats:
        if c is None:
            continue
        npig = float(np.dot(w, c["npig_img"]))
        if npig == 0:
            continue
        wd = w[c["img"]]
        tp = np.cumsum(c["tps"] * wd, axis=1)
        fp = np.cumsum(c["fps"] * wd, axis=1)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        # precision envelope (max to the right), vectorised
        env = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        nd = tp.shape[1]
        q = np.zeros((tp.shape[0], R))
        for t in range(tp.shape[0]):
            inds = np.searchsorted(rc[t], rec_thrs, side="left")
            valid = inds < nd
            q[t, valid] = env[t, inds[valid]]
        values.append(q)
    if not values:
        return -1.0
    return float(np.mean(np.stack(values)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--prediction", action="append", required=True, metavar="SEED=PATH")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    annotation = Path(args.annotation).resolve()
    sink = StringIO()
    with redirect_stdout(sink):
        gt = COCO(str(annotation))
    entries = []
    rng = np.random.default_rng(args.seed)
    n_img = None
    weights = None
    for item in args.prediction:
        if "=" not in item:
            raise SystemExit("--prediction must be SEED=PATH")
        seed, raw = item.split("=", 1)
        path = Path(raw).resolve()
        t0 = time.time()
        with redirect_stdout(sink):
            dt = gt.loadRes(str(path))
        per = {"seed": seed, "path": str(path), "sha256": sha256(path)}
        prepared = {}
        for label, use_cats in (("AP_cls", 1), ("AP_loc", 0)):
            ev, cats, rec_thrs, I = prepare(gt, dt, use_cats)
            point = float(ev.stats[0])
            check = ap_weighted(cats, rec_thrs, np.ones(I))
            per[label] = point
            per[label + "_reaccumulated_w1"] = check
            per[label + "_selfcheck_abs_diff"] = abs(point - check)
            if abs(point - check) > 1e-9:
                raise SystemExit(f"self-check failed for {label}: {point} vs {check}")
            prepared[label] = (cats, rec_thrs)
            if n_img is None:
                n_img = I
                weights = np.stack([np.bincount(rng.integers(0, I, I), minlength=I)
                                    for _ in range(args.resamples)]).astype(np.float64)
        per["delta_AP_loc_minus_AP_cls"] = per["AP_loc"] - per["AP_cls"]
        boot_cls = np.empty(args.resamples)
        boot_loc = np.empty(args.resamples)
        for b in range(args.resamples):
            w = weights[b]
            boot_cls[b] = ap_weighted(*prepared["AP_cls"], w)
            boot_loc[b] = ap_weighted(*prepared["AP_loc"], w)
            if (b + 1) % 200 == 0:
                print(f"[{seed}] {b + 1}/{args.resamples} resamples, {time.time() - t0:.0f}s", flush=True)
        delta = boot_loc - boot_cls
        per["bootstrap"] = {
            "AP_cls_ci95": [float(np.quantile(boot_cls, 0.025)), float(np.quantile(boot_cls, 0.975))],
            "AP_loc_ci95": [float(np.quantile(boot_loc, 0.025)), float(np.quantile(boot_loc, 0.975))],
            "delta_ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
            "delta_mean": float(delta.mean()),
            "delta_sd": float(delta.std(ddof=1)),
            "fraction_delta_positive": float(np.mean(delta > 0)),
        }
        per["_boot_delta"] = delta
        per["seconds"] = time.time() - t0
        # Per-resample arrays, so parallel single-seed runs that share --seed (hence identical
        # scene weights) can be pooled offline with paired resamples.
        arrays_path = Path(args.output).with_suffix("").as_posix() + f"_{seed}_resamples.npz"
        np.savez_compressed(arrays_path, boot_cls=boot_cls, boot_loc=boot_loc, weights_seed=args.seed,
                            n_images=n_img, resamples=args.resamples)
        per["resample_arrays"] = arrays_path
        entries.append(per)
        print(json.dumps({k: v for k, v in per.items() if not k.startswith("_")}, indent=2), flush=True)

    pooled = None
    if len(entries) > 1:
        deltas = np.stack([e["_boot_delta"] for e in entries])  # seeds x B, paired resamples
        mean_delta = deltas.mean(axis=0)
        pooled = {
            "n_seeds": len(entries),
            "point_mean_delta": float(np.mean([e["delta_AP_loc_minus_AP_cls"] for e in entries])),
            "point_sd_delta_ddof1": float(np.std([e["delta_AP_loc_minus_AP_cls"] for e in entries], ddof=1)),
            "bootstrap_mean_delta_ci95": [float(np.quantile(mean_delta, 0.025)), float(np.quantile(mean_delta, 0.975))],
            "bootstrap_min_over_seeds_delta_ci95": [float(np.quantile(deltas.min(axis=0), 0.025)),
                                                    float(np.quantile(deltas.min(axis=0), 0.975))],
            "fraction_all_seeds_positive": float(np.mean(np.all(deltas > 0, axis=0))),
        }
    for e in entries:
        e.pop("_boot_delta", None)
    out = {
        "schema_version": 1,
        "annotation": {"path": str(annotation), "sha256": sha256(annotation), "n_images": n_img},
        "metric": "COCO bbox AP@[.50:.95], identical prediction records; useCats=0 vs useCats=1",
        "bootstrap": {"type": "scene-paired, scenes resampled with replacement; same resample for "
                              "both evaluations and all seeds", "resamples": args.resamples,
                      "rng": f"numpy default_rng({args.seed})",
                      "implementation": "weighted re-accumulation of cached COCOeval match matrices "
                                        "(self-checked against COCOeval.stats[0] at unit weights)"},
        "predictions": entries,
        "pooled": pooled,
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
