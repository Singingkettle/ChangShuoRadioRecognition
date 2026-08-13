# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# bridge.py == the return-to-IQ recognizer + detect->channelize->recognize
# orchestration (verbatim from the paper). Paper-pipeline subcommands:
#   build | train-hier | bridge | oracle | diag-quality
# (the other subcommands are exploratory, kept for completeness). A
# first-class CSRR-native recognizer lives at
# csrr/models/backbones/returniq_resnet1d.py + configs/detection_is_easy/.

"""OURS Stage-2: return-to-IQ recognition pipeline (server-side, 65k hardshort_lowsnr).

Subcommands:
  build  : channelize each GT instance from the raw complex IQ to a baseband snippet
           (downconvert by metadata center_freq = TRUE carrier, brick-wall LPF to bw,
           decimate ~2.5x, crop/pad to L, energy-normalize); cache (X,y) per split.
  train  : train a 57-class ResNet-1D recognizer on the cached baseband snippets, with
           on-the-fly residual-CFO rotation augmentation (simulates Stage-1 CF error).
  bridge : take the BASELINE detector's predicted boxes (COCO json), return each box to
           IQ via the VERIFIED render rule (f=(0.5 - y/H)*fs, t=x/W*num_iq), recognize,
           and SWAP the class. Then compute class-aware COCO mAP for BASELINE (vision
           class) vs OURS (return-to-IQ class) against the SAME GT -- the headline number.
           Same boxes/scores for both -> the mAP gap is the pure recognition contribution.

All synthetic, on-server, log-verifiable. Honest: OURS beats baseline only if the
recognizer is good enough; per-family breakdown shows where return-to-IQ actually helps
(PSK/QAM constellation order) vs where the spectrogram already suffices (FSK/OFDM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _root_env(var, default):
    """Resolve a data root: $IQDET_<var> if set, else the repo-relative default."""
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else ROOT / default


# Where the three heavy assets live. Override with environment variables when the data
# sits outside the repository (the usual case: a fast NVMe scratch disk).
#   IQDET_MEMMAP_ROOT  packed STFT memmap + coco/ + metadata/  (produced by pack_coco_tensors_to_memmap.py)
#   IQDET_RAW_ROOT     raw IQ scenes                           (produced by prepare_torchsig_iq_stratified.py)
#   IQDET_CACHE_ROOT   channelized crop caches *_L1024.npz     (produced by `bridge.py build`)
MM = _root_env("IQDET_MEMMAP_ROOT", "data/torchsig_hardshort_lowsnr_stft3_memmap")
RAWDS = _root_env("IQDET_RAW_ROOT", "data/torchsig_hardshort_lowsnr_iq_65k_nvme")
CACHE = _root_env("IQDET_CACHE_ROOT", "work_dirs/returniq_cache")
# Pre-decoded raw scenes (<split>/<sample_id>.npy). Written by the export step's
# --raw-cache-root, and on some machines the only surviving copy of the raw IQ.
RAWCACHE = Path(os.environ["IQDET_RAW_CACHE_ROOT"]).expanduser().resolve() \
    if os.environ.get("IQDET_RAW_CACHE_ROOT") else RAWDS / "raw_npy_cache"


def raw_scene_path(split: str, sid: str) -> Path:
    """Locate one scene's raw IQ, preferring the original ``.npz``.

    Falls back to the pre-decoded ``.npy`` cache, which is what remains on machines where
    the ``.npz`` scenes were deleted after packing. Returns the ``.npz`` path when neither
    exists, so callers' ``.exists()`` guards keep behaving as before.
    """
    npz = RAWDS / "raw" / split / f"{sid}.npz"
    if npz.exists():
        return npz
    npy = RAWCACHE / split / f"{sid}.npy"
    if npy.exists():
        return npy
    return npz


def load_raw_iq(path: Path) -> np.ndarray:
    """Read a scene's complex IQ from either the ``.npz`` or the ``.npy`` layout.

    A ``.npz`` left truncated by an interrupted generation run raises deep inside zipfile
    with no file name attached, which used to abort a multi-hour cache build on a single
    bad scene. Fall back to the decoded cache when one exists, and name the file when it
    does not.
    """
    if path.suffix == ".npy":
        return np.load(path)
    try:
        z = np.load(path)
        return z["iq"] if "iq" in z else z[list(z.keys())[0]]
    except Exception as exc:
        cached = RAWCACHE / path.parent.name / f"{path.stem}.npy"
        if cached.exists():
            print(f"[raw] {path} is unreadable ({exc}); falling back to {cached}", flush=True)
            return np.load(cached)
        raise RuntimeError(f"cannot read raw IQ scene {path}: {exc}") from exc


# ----------------------------- shared channelizer -----------------------------
def _calibrate_cfo(bb):
    """Stage-2 DSP calibration: estimate residual CFO via spectral centroid and remove it
    (the crossover-law remedy for Stage-1 center-frequency error)."""
    X = np.abs(np.fft.fft(bb)) ** 2
    f = np.fft.fftfreq(len(bb))  # cycles/sample
    fc = float((f * X).sum() / (X.sum() + 1e-12))
    n = np.arange(len(bb))
    return bb * np.exp(-2j * np.pi * fc * n)


def channelize(iq_slice, cf_hz, bw_hz, fs, L, calibrate=False):
    """Complex IQ time-slice -> baseband [2,L] (real,imag), energy-normalized.
    calibrate=True applies a second-stage residual-CFO correction (spectral centroid)."""
    iq_slice = np.asarray(iq_slice, dtype=np.complex64)
    n = np.arange(len(iq_slice), dtype=np.float64)
    bb = iq_slice * np.exp(-2j * np.pi * cf_hz / fs * n)
    # brick-wall low-pass to +-bw/2
    X = np.fft.fft(bb)
    f = np.fft.fftfreq(len(bb), d=1.0 / fs)
    X[np.abs(f) > max(bw_hz, fs / len(bb)) / 2.0] = 0.0
    bb = np.fft.ifft(X)
    # decimate to ~2.5x bandwidth
    target_fs = max(2.5 * bw_hz, fs / max(len(bb), 1))
    D = int(max(1, round(fs / target_fs)))
    if D > 1:
        bb = bb[::D]
    if calibrate and len(bb) > 8:
        bb = _calibrate_cfo(bb)
    if len(bb) >= L:
        st = (len(bb) - L) // 2
        seg = bb[st:st + L]
    else:
        seg = np.zeros(L, dtype=np.complex128)
        seg[: len(bb)] = bb
    out = np.stack([seg.real, seg.imag]).astype(np.float32)
    return out / (np.sqrt((out ** 2).mean()) or 1.0)


def channelize_torch(iq_t, s0, s1, cf_hz, bw_hz, fs, L, calibrate=False, refine_cf=False, power_cal=False, refine_bw=False):
    """GPU channelizer (torch.fft) -- CPU-load-immune. iq_t is a complex tensor on device.
    Returns [2,L] float tensor on device, or None if slice too short.
    refine_cf: sub-bin CF refinement (fine-FFT energy centroid in the detector band) before downconvert."""
    import torch
    sl = iq_t[s0:s1]
    Nb = sl.numel()
    if Nb < 16:
        return None
    MAXLEN = 16384  # cap FFT length for speed (recognition needs a representative window, not full burst)
    if Nb > MAXLEN:
        st = (Nb - MAXLEN) // 2
        sl = sl[st:st + MAXLEN]
        Nb = MAXLEN
    dev = sl.device
    if refine_cf and Nb > 64:
        # sub-bin CF refinement: energy centroid within the detector's band on the RAW window
        # (finer than the 512-bin Stage-1 STFT; fixes box CF error at the source, no training)
        X0 = torch.fft.fft(sl)
        f0 = torch.fft.fftfreq(Nb, d=1.0 / fs, device=dev)
        band = ((f0 >= cf_hz - bw_hz) & (f0 <= cf_hz + bw_hz)).float()
        P = (X0.abs() ** 2) * band
        s = P.sum()
        if float(s) > 0:
            cf_hz = float((f0 * P).sum() / s)
    n = torch.arange(Nb, device=dev, dtype=torch.float32)
    ph = (-2.0 * np.pi * cf_hz / fs) * n
    bb = sl * torch.complex(torch.cos(ph), torch.sin(ph))
    X = torch.fft.fft(bb)
    f = torch.fft.fftfreq(Nb, d=1.0 / fs, device=dev)
    if refine_bw:
        # estimate occupied bandwidth from PSD energy-containment, BOUNDED to the detector's plausible band
        # (search within +-1.5x predicted bw to exclude neighbor signals; detector bw is ~+-40% off).
        af = f.abs()
        search = (af <= 1.5 * bw_hz / 2.0)
        P = (X.abs() ** 2) * search
        o = torch.argsort(af)
        pc = torch.cumsum(P[o], dim=0)
        tot = pc[-1] + 1e-12
        idx = int((pc >= 0.90 * tot).to(torch.int8).argmax())
        bw_hz = max(2.0 * float(af[o[idx]]), fs / Nb)
    X = X * (f.abs() <= max(bw_hz, fs / Nb) / 2.0)
    bb = torch.fft.ifft(X)
    D = int(max(1, round(fs / max(2.5 * bw_hz, fs / Nb))))
    if D > 1:
        bb = bb[::D]
    if calibrate and bb.numel() > 8:
        Pf = torch.fft.fft(bb).abs() ** 2
        ff = torch.fft.fftfreq(bb.numel(), device=dev)
        fc = (ff * Pf).sum() / (Pf.sum() + 1e-12)
        m = torch.arange(bb.numel(), device=dev, dtype=torch.float32)
        phc = (-2.0 * np.pi * fc) * m
        bb = bb * torch.complex(torch.cos(phc), torch.sin(phc))
    if power_cal and bb.numel() > 16:
        # classic Mth-power blind CFO estimator (M=4 for QPSK/QAM): raise to 4th power -> tone at 4*cfo
        z = bb / (bb.abs() + 1e-6)            # amplitude-normalize so modulation->phase only
        z4 = z * z * z * z
        Zf = torch.fft.fft(z4).abs()
        ff2 = torch.fft.fftfreq(bb.numel(), device=dev)
        peak = ff2[int(Zf.argmax())]
        cfo = peak / 4.0
        m2 = torch.arange(bb.numel(), device=dev, dtype=torch.float32)
        phn = (-2.0 * np.pi * float(cfo)) * m2
        bb = bb * torch.complex(torch.cos(phn), torch.sin(phn))
    nb = bb.numel()
    if nb >= L:
        st = (nb - L) // 2
        seg = bb[st:st + L]
    else:
        seg = torch.zeros(L, dtype=bb.dtype, device=dev)
        seg[:nb] = bb
    out = torch.stack([seg.real, seg.imag])
    return out / (out.pow(2).mean().sqrt() + 1e-8)


def load_catmap():
    cats = sorted(json.loads((MM / "coco_multiclass" / "annotations" / "instances_train.json").read_text())["categories"], key=lambda c: c["id"])
    cid2idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    return cid2idx, names


def family_of(name):
    n = name.lower()
    for fam in ("ofdm", "qam", "psk", "ask", "msk", "gfsk", "gmsk", "fsk", "fm", "am", "lfm", "chirp"):
        if fam in n:
            return {"gfsk": "fsk", "gmsk": "msk"}.get(fam, fam)
    return "other"


# ------------------------------- build cache ----------------------------------
def cmd_build(args):
    cid2idx, names = load_catmap()
    fs = None
    X, y = [], []
    meta_lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if args.limit:
        meta_lines = meta_lines[: args.limit]
    miss = 0
    for k, line in enumerate(meta_lines):
        if not line.strip():
            continue
        r = json.loads(line)
        sid, nq, fs = r["sample_id"], r["num_iq_samples"], r["sample_rate"]
        rp = raw_scene_path(args.split, sid)
        if not rp.exists():
            miss += 1
            continue
        iq = load_raw_iq(rp)
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            s0 = max(0, int(inst["start_in_samples"]))
            s1 = min(len(iq), s0 + int(inst["duration_in_samples"]))
            if s1 - s0 < 16:
                continue
            X.append(channelize(iq[s0:s1], inst["center_freq"], inst["bandwidth"], fs, args.L))
            y.append(cid2idx[inst["category_id"]])
        if (k + 1) % 2000 == 0:
            print(f"[build:{args.split}] {k+1}/{len(meta_lines)} samples, {len(y)} snippets", flush=True)
    # A missing raw tree used to be counted and ignored, so an entirely absent raw/
    # directory wrote an empty cache and exited 0 -- and the recognizer then trained on
    # nothing. Refuse to save a cache that is mostly holes.
    seen = min(args.limit, len(meta_lines)) if args.limit else len(meta_lines)
    miss_frac = miss / max(seen, 1)
    if miss_frac > args.max_missing_frac:
        raise SystemExit(
            f"[build:{args.split}] {miss} of {seen} scenes had no readable raw IQ "
            f"({miss_frac:.1%} > --max-missing-frac {args.max_missing_frac:.1%}). Looked under "
            f"{RAWDS / 'raw' / args.split} (.npz) and {RAWCACHE / args.split} (.npy). "
            "Refusing to write a hollow cache."
        )
    if not y:
        raise SystemExit(f"[build:{args.split}] produced 0 snippets; nothing to save.")
    CACHE.mkdir(parents=True, exist_ok=True)
    outp = CACHE / f"{args.split}_L{args.L}{'_lim'+str(args.limit) if args.limit else ''}.npz"
    np.savez(outp, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64), fs=fs)
    print(f"[build:{args.split}] saved {outp}  X={np.asarray(X).shape} "
          f"miss_samples={miss} ({miss_frac:.2%})")


# --------------------------------- model --------------------------------------
def make_model(n_cls):
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, c, k=5):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, k, padding=k // 2); self.b1 = nn.BatchNorm1d(c)
            self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.b2 = nn.BatchNorm1d(c)

        def forward(self, x):
            h = F.relu(self.b1(self.c1(x)))
            h = self.b2(self.c2(h))
            return F.relu(h + x)

    class ResNet1D(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
            layers, cin = [], 64
            for c in [64, 128, 256]:
                if c != cin:
                    layers += [nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU()]; cin = c
                layers += [ResBlock(c), ResBlock(c), nn.MaxPool1d(2)]
            self.body = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_cls))

        def forward(self, x):
            return self.head(self.body(self.stem(x)))

    return ResNet1D(n_cls)


def cmd_train(args):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, names = load_catmap()
    d = np.load(CACHE / args.train_cache)
    X, y = d["X"], d["y"]
    print(f"[train] {X.shape} over {len(names)} classes on {dev}")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    model = make_model(len(names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long()
    nn_ = len(Xt); L = X.shape[2]
    narr = np.arange(L, dtype=np.float32)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(nn_)
        tot = 0.0
        for i in range(0, nn_, args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx].clone()
            # residual-CFO rotation augmentation (simulate Stage-1 CF error, +-aug_bins STFT bins)
            if args.aug_cfo > 0:
                b = xb.shape[0]
                eps = (rng.uniform(-args.aug_cfo, args.aug_cfo, size=b)).astype(np.float32)  # cycles/sample at channel rate
                ang = 2 * np.pi * eps[:, None] * narr[None, :]
                cos = torch.from_numpy(np.cos(ang)); sin = torch.from_numpy(np.sin(ang))
                re = xb[:, 0] * cos - xb[:, 1] * sin
                im = xb[:, 0] * sin + xb[:, 1] * cos
                xb = torch.stack([re, im], dim=1)
            xb = xb.to(dev); yb = yt[idx].to(dev)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); opt.step(); tot += float(loss) * len(idx)
        print(f"[train] epoch {ep+1}/{args.epochs} loss {tot/nn_:.4f}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        model.load_state_dict({k[len('module.'):] if k.startswith('module.') else k: v
                               for k, v in ema_model.module.state_dict().items()})
    torch.save(model.state_dict(), CACHE / args.out)
    # quick val accuracy if provided
    if args.val_cache and (CACHE / args.val_cache).exists():
        dv = np.load(CACHE / args.val_cache); Xv, yv = dv["X"], dv["y"]
        model.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(Xv), 512):
                pred.append(model(torch.from_numpy(Xv[i:i+512]).float().to(dev)).argmax(1).cpu().numpy())
        pred = np.concatenate(pred); acc = float((pred == yv).mean())
        print(f"[train] val top-1 acc = {acc:.4f} (n={len(yv)})")
    print(f"[train] saved {CACHE / args.out}")


# ------------------------------- bridge + eval --------------------------------
def wbf_cluster_boxes(dets, iou_thr):
    """For each det, return the score-weighted-average box [x,y,w,h] of its IoU>=thr cluster.
    Used to feed a LOWER-VARIANCE cf/bw to the channelizer WITHOUT altering the box set that the
    mAP metric scores (localization/recall unchanged; only the return-to-IQ crop is cleaned)."""
    n = len(dets)
    assigned = [None] * n
    used = [False] * n
    order = sorted(range(n), key=lambda i: -float(dets[i].get("score", 1.0)))
    for ii in order:
        if used[ii]:
            continue
        x1, y1, w1, h1 = dets[ii]["bbox"]; used[ii] = True
        members = [ii]
        for jj in order:
            if used[jj] or jj == ii:
                continue
            x2, y2, w2, h2 = dets[jj]["bbox"]
            ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy; u = w1 * h1 + w2 * h2 - inter
            if u > 0 and inter / u >= iou_thr:
                members.append(jj); used[jj] = True
        ws = [float(dets[m].get("score", 1.0)) for m in members]; sw = sum(ws) or 1.0
        fb = [sum(dets[m]["bbox"][k] * w for m, w in zip(members, ws)) / sw for k in range(4)]
        for m in members:
            assigned[m] = fb
    return assigned


def cmd_bridge(args):
    import torch
    from iqdet_metrics import class_aware_detection_map
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap()
    nC = len(names)
    fams = [family_of(n) for n in names]
    H = W = 512

    # GT from metadata (true carrier); boxes normalized TF [t0,t1,fc,bw]
    gt_boxes, gt_labels, order = {}, {}, []
    _lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if getattr(args, "limit", 0):
        _lines = _lines[: args.limit]
    for line in _lines:
        if not line.strip():
            continue
        r = json.loads(line); nq, fs = r["num_iq_samples"], r["sample_rate"]; sid = r["sample_id"]
        bx, lb = [], []
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            bx.append([inst["start_in_samples"]/nq, (inst["start_in_samples"]+inst["duration_in_samples"])/nq,
                       inst["center_freq"]/fs, inst["bandwidth"]/fs])
            lb.append(cid2idx[inst["category_id"]])
        gt_boxes[sid] = bx; gt_labels[sid] = lb; order.append(sid)

    # baseline predictions (COCO json): image_id->stem via coco images
    coco = json.loads((MM / "coco_multiclass" / "annotations" / f"instances_{args.split}.json").read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    preds = json.loads(Path(args.baseline_pred).read_text())
    order_set = set(order)
    by_sid = defaultdict(list)
    for d in preds:
        s = id2stem[d["image_id"]]
        if s in order_set and float(d.get("score", 1.0)) >= args.score_thr:
            by_sid[s].append(d)

    # recognizer (flat / hierarchical / e2e-refine)
    use_hier = bool(getattr(args, "hier_model", None))
    use_e2e = bool(getattr(args, "e2e_model", None))
    use_refine = bool(getattr(args, "refine_model", None))
    if use_refine:
        model = make_refine_model(nC, args.L).to(dev)
        model.load_state_dict(torch.load(CACHE / args.refine_model, map_location=dev)); model.eval()
    elif use_e2e:
        model = make_e2e_model(nC, args.L).to(dev)
        model.load_state_dict(torch.load(CACHE / args.e2e_model, map_location=dev)); model.eval()
    elif use_hier:
        is_multi_h, single_cls_h, multi_cls_h = hier_classes(names)
        single_arr_h = np.array(single_cls_h)
        multi_arr_h = np.array(multi_cls_h)
        model = make_hier_model(len(single_cls_h), len(multi_cls_h)).to(dev)
        model.load_state_dict(torch.load(CACHE / args.hier_model, map_location=dev)); model.eval()
    else:
        model = make_model(nC).to(dev); model.load_state_dict(torch.load(CACHE / args.model, map_location=dev)); model.eval()
    fs_default = 10_000_000.0

    base_b, base_s, base_l = {}, {}, {}   # baseline: vision class
    ours_b, ours_s, ours_l = {}, {}, {}   # ours: return-to-IQ class
    fused_l = {}                          # domain-matched fusion: IQ for constellation families, vision else
    # CORRECTED domain-matching (from DEPLOYMENT per-family): amplitude-phase CONSTELLATION families ->
    # return-to-IQ (IQ wins: psk +0.16, ask +0.12, qam +0.05); frequency/analog families (fsk/msk/chirp/
    # fm/am/ofdm) -> vision (spectrogram wins those on real boxes). Override via --iq-families.
    IQ_SET = set((getattr(args, "iq_families", None) or "psk,qam,ask").split(","))
    nb = 0
    for si, sid in enumerate(order):
        dets = by_sid.get(sid, [])
        bb, bs, bl, ob, os_, ol = [], [], [], [], [], []
        iq_t = None
        kept = [d for d in dets if float(d.get("score", 1.0)) >= args.score_thr]
        wbf_boxes = wbf_cluster_boxes(kept, args.wbf_iou) if getattr(args, "wbf_iou", 1.0) < 1.0 else None
        if getattr(args, "nms_iou", 1.0) < 1.0 and len(kept) > 1:
            kept = sorted(kept, key=lambda d: -float(d.get("score", 1.0)))
            keep2 = []
            for d in kept:
                x1, y1, w1, h1 = d["bbox"]
                ok = True
                for kj in keep2:
                    x2, y2, w2, h2 = kj["bbox"]
                    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    inter = ix * iy
                    u = w1 * h1 + w2 * h2 - inter
                    if u > 0 and inter / u >= args.nms_iou:
                        ok = False; break
                if ok:
                    keep2.append(d)
            kept = keep2
        rp = raw_scene_path(args.split, sid)
        if kept and rp.exists():
            try:
                iq = load_raw_iq(rp)
                iq_t = torch.from_numpy(np.ascontiguousarray(iq)).to(dev); nq = iq_t.numel()
            except Exception:  # corrupt npz -> keep vision labels for this scene
                iq_t = None
        snippets, idxmap = [], []
        for _ki, d in enumerate(kept):
            x, y0, w, h = d["bbox"]
            t0, t1 = x / W, (x + w) / W
            fc = (0.5 - (y0 + h / 2) / H)        # normalized freq center (verified rule)
            bw = h / H
            box = [t0, t1, fc, bw]
            sc = float(d.get("score", 1.0))
            lab_vision = cid2idx.get(d["category_id"], 0)
            bb.append(box); bs.append(sc); bl.append(lab_vision)
            ob.append(box); os_.append(sc)
            ol.append(lab_vision)  # placeholder, overwritten below if recognized
            if iq_t is not None:
                cf_u, bw_u, t0u, t1u = fc, bw, t0, t1
                if wbf_boxes is not None:
                    fx, fy, fw, fh = wbf_boxes[_ki]
                    cf_u = (0.5 - (fy + fh / 2) / H)
                    bw_u = fh / H
                ofix = getattr(args, "oracle_fix", "off")
                if ofix != "off" and gt_boxes.get(sid):
                    from iqdet_metrics import time_frequency_iou as _tfi3
                    _gb = gt_boxes[sid]
                    _iv = _tfi3(torch.tensor([box], dtype=torch.float32), torch.tensor(_gb, dtype=torch.float32).reshape(-1, 4))[0]
                    _gj = int(_iv.argmax())
                    if float(_iv[_gj]) > 0:
                        if "bw" in ofix: bw_u = _gb[_gj][3]
                        if "cf" in ofix: cf_u = _gb[_gj][2]
                        if "time" in ofix: t0u, t1u = _gb[_gj][0], _gb[_gj][1]
                s0 = max(0, int(t0u * nq)); s1 = min(nq, int(t1u * nq))
                snip = channelize_torch(iq_t, s0, s1, cf_u * fs_default, max(abs(bw_u) * fs_default, 1.0), fs_default, args.L, calibrate=args.calibrate, refine_cf=getattr(args, "refine_cf", False), power_cal=getattr(args, "power_cal", False), refine_bw=getattr(args, "refine_bw", False))
                if snip is not None:
                    snippets.append(snip); idxmap.append(len(ol) - 1)
        # recognize snippets in batch (GPU) -> overwrite OURS labels (+ optionally OURS scores)
        if snippets:
            with torch.no_grad():
                out = model(to_input_rep(torch.stack(snippets), getattr(args, "input_rep", "iq")))
                if use_e2e or use_refine:
                    logits = out[0]
                elif use_hier:
                    logits = out[1]  # hier single-carrier branch
                else:
                    logits = out
                conf = torch.softmax(logits, dim=1).max(1).values.cpu().numpy()
                if use_hier:
                    if getattr(args, 'hier_coarse_route', False):
                        _cp = out[0].argmax(1).cpu().numpy()
                        _rs = single_arr_h[out[1].argmax(1).cpu().numpy()]
                        _rm = multi_arr_h[np.clip(out[2].argmax(1).cpu().numpy(), 0, len(multi_arr_h) - 1)]
                        rec = np.where(_cp == 1, _rm, _rs)
                        _cs = torch.softmax(out[1], dim=1).max(1).values.cpu().numpy()
                        _cm = torch.softmax(out[2], dim=1).max(1).values.cpu().numpy()
                        conf = np.where(_cp == 1, _cm, _cs)
                    else:
                        rec = single_arr_h[out[1].argmax(1).cpu().numpy()]
                else:
                    rec = logits.argmax(1).cpu().numpy()
            for jj, pos in enumerate(idxmap):
                ol[pos] = int(rec[jj])
                if getattr(args, "ours_score_recog", False):
                    os_[pos] = os_[pos] * float(conf[jj])  # det x recognition confidence
        nb += len(dets)
        base_b[sid], base_s[sid], base_l[sid] = bb, bs, bl
        ours_b[sid], ours_s[sid], ours_l[sid] = ob, os_, ol
        gate = [True] * len(bb)
        gmode = getattr(args, "fidelity_gate_mode", "off")
        if gmode == "gt" and gt_boxes.get(sid) and len(bb):
            from iqdet_metrics import time_frequency_iou as _tfi2
            PBg = torch.tensor(bb, dtype=torch.float32).reshape(-1, 4)
            GBg = torch.tensor(gt_boxes[sid], dtype=torch.float32).reshape(-1, 4)
            if GBg.shape[0] > 0:
                ioug = _tfi2(PBg, GBg)
                glo, ghi = args.fidelity_gate_lo, args.fidelity_gate_hi
                for i in range(len(bb)):
                    gj = int(ioug[i].argmax())
                    gbw = abs(gt_boxes[sid][gj][3]); pbw = abs(bb[i][3])
                    bwr = (pbw / gbw) if gbw > 0 else 99.0
                    gate[i] = (glo <= bwr <= ghi)
        fused_l[sid] = [ol[i] if (fams[bl[i]] in IQ_SET and gate[i]) else bl[i] for i in range(len(bl))]
        if (si + 1) % 2000 == 0:
            print(f"[bridge] {si+1}/{len(order)} samples, {nb} dets", flush=True)

    def to_t(dct, sid, w):
        v = dct.get(sid, [])
        return torch.tensor(v, dtype=torch.float32).reshape(-1, 4) if w == 4 else torch.tensor(v).reshape(-1)

    def evalset(pb, ps, pl, cats=None):
        import torch as T
        from iqdet_metrics import time_frequency_iou as _tfi
        GB = [T.tensor(gt_boxes[s], dtype=T.float32).reshape(-1, 4) for s in order]
        GL = [T.tensor(gt_labels[s], dtype=T.long).reshape(-1) for s in order]
        PB = [T.tensor(pb[s], dtype=T.float32).reshape(-1, 4) for s in order]
        PS = [T.tensor(ps[s], dtype=T.float32).reshape(-1) for s in order]
        PL = [T.tensor(pl[s], dtype=T.long).reshape(-1) for s in order]
        cnms = getattr(args, "class_nms_iou", 1.0)
        if cnms < 1.0:  # per-class NMS on EACH method's own labels (symmetric, fair dedup)
            PB2, PS2, PL2 = [], [], []
            for b, s, l in zip(PB, PS, PL):
                if len(b) <= 1:
                    PB2.append(b); PS2.append(s); PL2.append(l); continue
                ordr = T.argsort(s, descending=True).tolist(); keep = []
                for i in ordr:
                    if all(not (int(l[i]) == int(l[j]) and float(_tfi(b[i:i+1], b[j:j+1])) >= cnms) for j in keep):
                        keep.append(i)
                PB2.append(b[keep]); PS2.append(s[keep]); PL2.append(l[keep])
            PB, PS, PL = PB2, PS2, PL2
        if cats is not None:
            cs = set(cats)
            def filt(boxes, labels, scores=None):
                keep = [i for i, l in enumerate(labels.tolist()) if l in cs]
                b = boxes[keep] if len(keep) else boxes[:0]
                l = labels[keep] if len(keep) else labels[:0]
                if scores is None:
                    return b, l
                s = scores[keep] if len(keep) else scores[:0]
                return b, l, s
            GB2, GL2, PB2, PS2, PL2 = [], [], [], [], []
            for gb, gl, pb_, ps_, pl_ in zip(GB, GL, PB, PS, PL):
                b, l = filt(gb, gl); GB2.append(b); GL2.append(l)
                b, l, s = filt(pb_, pl_, ps_); PB2.append(b); PL2.append(l); PS2.append(s)
            GB, GL, PB, PS, PL = GB2, GL2, PB2, PS2, PL2
        return class_aware_detection_map(PB, PS, PL, GB, GL, num_classes=nC)

    print("\n================ OURS (return-to-IQ) vs BASELINE (vision) -- class-aware COCO mAP ================")
    mb = evalset(base_b, base_s, base_l)
    mo = evalset(ours_b, ours_s, ours_l)
    mf = evalset(base_b, base_s, fused_l)
    print(f"{'':>24} | {'mAP@.5:.95':>10} {'mAP@.5':>8} {'mAP@.75':>8}")
    print(f"{'BASELINE (vision)':>24} | {mb['class_bbox_mAP']:>10.4f} {mb['class_bbox_mAP_50']:>8.4f} {mb['class_bbox_mAP_75']:>8.4f}")
    print(f"{'OURS (pure return-to-IQ)':>24} | {mo['class_bbox_mAP']:>10.4f} {mo['class_bbox_mAP_50']:>8.4f} {mo['class_bbox_mAP_75']:>8.4f}")
    print(f"{'OURS-FUSED (domain-match)':>24} | {mf['class_bbox_mAP']:>10.4f} {mf['class_bbox_mAP_50']:>8.4f} {mf['class_bbox_mAP_75']:>8.4f}")
    print(f"{'fused delta vs baseline':>24} | {mf['class_bbox_mAP']-mb['class_bbox_mAP']:>+10.4f}")
    # per-family breakdown
    fam2cats = defaultdict(list)
    for i, f in enumerate(fams):
        fam2cats[f].append(i)
    print("\n-- per-family class-aware mAP@.5:.95 (where return-to-IQ helps) --")
    print(f"{'family':>10} {'#cls':>4} | {'BASELINE':>9} {'OURS':>9} {'delta':>8}")
    for fam, cats in sorted(fam2cats.items()):
        b = evalset(base_b, base_s, base_l, cats)["class_bbox_mAP"]
        o = evalset(ours_b, ours_s, ours_l, cats)["class_bbox_mAP"]
        print(f"{fam:>10} {len(cats):>4} | {b:>9.4f} {o:>9.4f} {o-b:>+8.4f}")
    print("\n[honest] same boxes/scores for both; only the CLASS source differs (vision vs return-to-IQ).")


def make_refine_model(nC, L):
    """Learned CFO-refine recognizer: a small head predicts the residual CFO, the signal is
    explicitly de-rotated (differentiable), then a ResNet1D recognizes. The crossover-law remedy
    done as a learnable module -- trained on CLEAN snippets with SYNTHETIC cf injection (clean labels)."""
    import torch
    import torch.nn as nn

    class Refine(nn.Module):
        def __init__(self):
            super().__init__()
            self.L = L
            self.cfo = nn.Sequential(
                nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Linear(64, 1))
            self.rec = make_model(nC)

        def forward(self, x):
            eps = self.cfo(x).squeeze(1)  # [B] normalized cf (cycles/sample)
            n = torch.arange(self.L, device=x.device, dtype=torch.float32)
            ang = -2.0 * np.pi * eps[:, None] * n[None, :]
            cos, sin = torch.cos(ang), torch.sin(ang)
            re = x[:, 0] * cos - x[:, 1] * sin
            im = x[:, 0] * sin + x[:, 1] * cos
            return self.rec(torch.stack([re, im], dim=1)), eps

    return Refine()


def cmd_train_refine(args):
    """Train the learned CFO-refine recognizer on the CLEAN cache + synthetic cf injection (clean labels).
    Loss = CE(recog) + lambda * MSE(predicted cf, injected cf). Deploys via bridge --refine-model:
    the head de-rotates the predicted-box residual CFO so the constellation is restored."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, names = load_catmap(); nC = len(names)
    d = np.load(CACHE / args.train_cache); X, y = d["X"], d["y"]
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    L = X.shape[2]; narr = torch.arange(L, dtype=torch.float32)
    model = make_refine_model(nC, L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long(); n = len(Xt)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx]; yb = yt[idx].to(dev)
            b = xb.shape[0]
            eps_true = torch.from_numpy(rng.uniform(-args.inject_cf, args.inject_cf, size=b).astype(np.float32))
            ang = 2.0 * np.pi * eps_true[:, None] * narr[None, :]
            cos, sin = torch.cos(ang), torch.sin(ang)
            re = xb[:, 0] * cos - xb[:, 1] * sin; im = xb[:, 0] * sin + xb[:, 1] * cos
            xb = torch.stack([re, im], dim=1).to(dev); eg = eps_true.to(dev)
            logits, eps_hat = model(xb)
            loss = F.cross_entropy(logits, yb) + args.lam * F.mse_loss(eps_hat, eg)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * b
        print(f"[refine] epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)
    torch.save(model.state_dict(), CACHE / args.out)
    # val: clean acc + acc under injected cf (robustness)
    dv = np.load(CACHE / args.val_cache); Xv, yv = dv["X"], dv["y"]
    model.eval()
    with torch.no_grad():
        p0 = []
        for i in range(0, len(Xv), 512):
            p0.append(model(torch.from_numpy(Xv[i:i+512]).float().to(dev))[0].argmax(1).cpu().numpy())
        p0 = np.concatenate(p0)
    print(f"[refine] val acc (clean) = {(p0==yv).mean():.4f}")
    print(f"[refine] saved {CACHE / args.out}")


def hier_classes(names):
    is_multi = [1 if "ofdm" in n.lower() else 0 for n in names]
    single_cls = [i for i in range(len(names)) if not is_multi[i]]
    multi_cls = [i for i in range(len(names)) if is_multi[i]]
    return is_multi, single_cls, multi_cls


def to_input_rep(x, rep):
    """[B,2,L] (I,Q) -> input representation. rep='diff' uses the differential product
    z[n]=x[n]*conj(x[n-1]) (Re,Im): a constant CFO becomes a CONSTANT rotation of z instead of a
    per-sample ramp, so the recognizer is structurally robust to the residual carrier offset that
    corrupts the channelized constellation. rep='iq' is the identity."""
    import torch
    if rep == "iq":
        return x
    c = torch.complex(x[:, 0], x[:, 1])
    z = c[:, 1:] * torch.conj(c[:, :-1])
    z = torch.nn.functional.pad(z, (1, 0))
    zr = z.real.unsqueeze(1); zi = z.imag.unsqueeze(1)
    if rep == "diff":
        return torch.cat([zr, zi], dim=1)
    if rep == "iqdiff":
        return torch.cat([x, zr, zi], dim=1)
    raise ValueError("unknown input-rep " + str(rep))


def make_hier_model(n_single, n_multi):
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, c, k=5):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, k, padding=k // 2); self.b1 = nn.BatchNorm1d(c)
            self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.b2 = nn.BatchNorm1d(c)

        def forward(self, x):
            h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return F.relu(h + x)

    class HierNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
            layers, cin = [], 64
            for c in [64, 128, 256]:
                if c != cin:
                    layers += [nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU()]; cin = c
                layers += [ResBlock(c), ResBlock(c), nn.MaxPool1d(2)]
            self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.coarse = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
            self.single = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_single))
            self.multi = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, max(n_multi, 1)))

        def forward(self, x):
            f = self.body(self.stem(x))
            return self.coarse(f), self.single(f), self.multi(f)

    return HierNet()


def make_e2e_model(n_cls, L):
    """Differentiable Stage-2: a CFO-estimation head -> differentiable rotation correction -> recognizer.
    Trained end-to-end on the recognition loss so it learns to UNDO the predicted-box center-frequency error
    (the dominant IQ-killer per the crossover law). This is the realizable form of end-to-end joint training."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, c, k=5):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, k, padding=k // 2); self.b1 = nn.BatchNorm1d(c)
            self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.b2 = nn.BatchNorm1d(c)

        def forward(self, x):
            h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return F.relu(h + x)

    class Recog(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
            layers, cin = [], 64
            for c in [64, 128, 256]:
                if c != cin:
                    layers += [nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU()]; cin = c
                layers += [ResBlock(c), ResBlock(c), nn.MaxPool1d(2)]
            self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_cls))

        def forward(self, x):
            return self.fc(self.body(self.stem(x)))

    class E2ERefineIQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.L = L
            self.cfo = nn.Sequential(
                nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.cfo_fc = nn.Linear(64, 1)
            self.recog = Recog()

        def forward(self, x):
            cfo = self.cfo_fc(self.cfo(x)).squeeze(1)           # [B] estimated normalized CFO
            xc = torch.complex(x[:, 0], x[:, 1])                 # [B,L]
            n = torch.arange(self.L, device=x.device).float()
            ang = -2 * np.pi * cfo[:, None] * n[None, :]
            rot = torch.complex(torch.cos(ang), torch.sin(ang))
            xcorr = xc * rot
            xr = torch.stack([xcorr.real, xcorr.imag], dim=1)
            return self.recog(xr), cfo

    return E2ERefineIQ()


def cmd_train_e2e(args):
    """Train the differentiable Stage-2 (CFO-refine + recognizer) on PREDICTED-box snippets end-to-end:
    CE(recog) + lambda*MSE(estimated_cfo, gt_residual_cfo). Cache must have X, y, and cfo (residual target)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, names = load_catmap(); nC = len(names)
    d = np.load(CACHE / args.train_cache)
    X, y = d["X"], d["y"]; gt = d["cfo"] if "cfo" in d else np.zeros(len(y), np.float32)
    print(f"[e2e] train {X.shape}, cfo target std={float(np.std(gt)):.4f}")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    model = make_e2e_model(nC, X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y).long(); gtt = torch.from_numpy(gt).float()
    n = len(Xt)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx].to(dev); yb = yt[idx].to(dev); gb = gtt[idx].to(dev)
            logits, cfo = model(xb)
            loss = F.cross_entropy(logits, yb) + args.lam * F.mse_loss(cfo, gb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        print(f"[e2e] epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)
    torch.save(model.state_dict(), CACHE / args.out)
    if args.val_cache and (CACHE / args.val_cache).exists():
        dv = np.load(CACHE / args.val_cache); Xv, yv = dv["X"], dv["y"]
        model.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(Xv), 512):
                pred.append(model(torch.from_numpy(Xv[i:i+512]).float().to(dev))[0].argmax(1).cpu().numpy())
        print(f"[e2e] val top-1 (clean GT boxes) = {float((np.concatenate(pred)==yv).mean()):.4f}")
    print(f"[e2e] saved {CACHE / args.out}")


def cmd_train_hier(args):
    """HIERARCHICAL recognizer (user's design): Stage1 single-carrier vs multi-carrier, then a single-carrier
    fine head and a multi-carrier (OFDM) fine head. Trains on cached IQ snippets; reports Stage1 acc,
    per-branch acc, and combined fine acc -- vs a flat classifier. (Multi-carrier on narrowband IQ is
    expected weak -> motivates the spectral branch / vision routing for OFDM.)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, names = load_catmap(); nC = len(names)
    is_multi = [1 if "ofdm" in n.lower() else 0 for n in names]
    single_cls = [i for i in range(nC) if not is_multi[i]]
    multi_cls = [i for i in range(nC) if is_multi[i]]
    s_pos = {c: i for i, c in enumerate(single_cls)}; m_pos = {c: i for i, c in enumerate(multi_cls)}
    print(f"[hier] single-carrier {len(single_cls)} cls, multi-carrier(OFDM) {len(multi_cls)} cls")

    class ResBlock(nn.Module):
        def __init__(self, c, k=5):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, k, padding=k // 2); self.b1 = nn.BatchNorm1d(c)
            self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.b2 = nn.BatchNorm1d(c)

        def forward(self, x):
            h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return F.relu(h + x)

    class HierNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
            layers, cin = [], 64
            for c in [64, 128, 256]:
                if c != cin:
                    layers += [nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU()]; cin = c
                layers += [ResBlock(c), ResBlock(c), nn.MaxPool1d(2)]
            self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.coarse = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
            self.single = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(single_cls)))
            self.multi = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, max(len(multi_cls), 1)))

        def forward(self, x):
            f = self.body(self.stem(x))
            return self.coarse(f), self.single(f), self.multi(f)

    d = np.load(CACHE / args.train_cache); X, y = d["X"], d["y"]
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    cy = torch.tensor([is_multi[int(v)] for v in y], dtype=torch.long)
    sy = torch.tensor([s_pos.get(int(v), -1) for v in y], dtype=torch.long)
    my = torch.tensor([m_pos.get(int(v), -1) for v in y], dtype=torch.long)
    Xt = torch.from_numpy(X).float(); n = len(Xt); L = X.shape[2]; narr = np.arange(L, dtype=np.float32)
    model = HierNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = None
    if getattr(args, 'cosine', False):
        import math as _m
        warm = 5
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: min((ep + 1) / warm, 0.5 * (1 + _m.cos(_m.pi * ep / max(args.epochs, 1)))))
    ema_model = None
    if getattr(args, 'ema', 0.0) > 0:
        from torch.optim.swa_utils import AveragedModel
        _d = args.ema
        ema_model = AveragedModel(model, avg_fn=lambda a, c, n: _d * a + (1 - _d) * c)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx].clone()
            if args.aug_cfo > 0:
                b = xb.shape[0]; eps = rng.uniform(-args.aug_cfo, args.aug_cfo, size=b).astype(np.float32)
                ang = 2 * np.pi * eps[:, None] * narr[None, :]
                cos = torch.from_numpy(np.cos(ang)); sin = torch.from_numpy(np.sin(ang))
                xb = torch.stack([xb[:, 0] * cos - xb[:, 1] * sin, xb[:, 0] * sin + xb[:, 1] * cos], dim=1)
            xb = to_input_rep(xb, args.input_rep)
            xb = xb.to(dev); cb = cy[idx].to(dev); sb = sy[idx].to(dev); mb = my[idx].to(dev)
            ls = getattr(args, 'label_smooth', 0.0)
            mx = getattr(args, 'mixup', 0.0)
            def _hier_loss(_cl, _sl, _ml, _cb, _sb, _mb):
                _l = F.cross_entropy(_cl, _cb, label_smoothing=ls)
                if (_sb >= 0).any():
                    _l = _l + F.cross_entropy(_sl[_sb >= 0], _sb[_sb >= 0], label_smoothing=ls)
                if (_mb >= 0).any():
                    _l = _l + F.cross_entropy(_ml[_mb >= 0], _mb[_mb >= 0], label_smoothing=ls)
                return _l
            if mx > 0:
                lam = float(np.random.beta(mx, mx))
                pj = torch.randperm(xb.shape[0], device=xb.device)
                xb = lam * xb + (1 - lam) * xb[pj]
                cl, sl, ml = model(xb)
                loss = lam * _hier_loss(cl, sl, ml, cb, sb, mb) + (1 - lam) * _hier_loss(cl, sl, ml, cb[pj], sb[pj], mb[pj])
            else:
                cl, sl, ml = model(xb)
                loss = _hier_loss(cl, sl, ml, cb, sb, mb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
            if ema_model is not None:
                ema_model.update_parameters(model)
        print(f"[hier] epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)
        if sched is not None:
            sched.step()
    torch.save(model.state_dict(), CACHE / args.out)
    # eval on val cache
    dv = np.load(CACHE / args.val_cache); Xv, yv = dv["X"], dv["y"]
    model.eval(); cP, finalP = [], []
    single_arr = np.array(single_cls); multi_arr = np.array(multi_cls)
    with torch.no_grad():
        for i in range(0, len(Xv), 512):
            cl, sl, ml = model(to_input_rep(torch.from_numpy(Xv[i:i+512]).float().to(dev), args.input_rep))
            cp = cl.argmax(1).cpu().numpy(); sp = sl.argmax(1).cpu().numpy(); mp = ml.argmax(1).cpu().numpy()
            cP.append(cp)
            fp = np.where(cp == 1, multi_arr[np.clip(mp, 0, len(multi_arr)-1)], single_arr[np.clip(sp, 0, len(single_arr)-1)])
            finalP.append(fp)
    cP = np.concatenate(cP); finalP = np.concatenate(finalP)
    cyv = np.array([is_multi[int(v)] for v in yv])
    print(f"\n[hier] Stage1 single/multi acc = {(cP==cyv).mean():.4f}")
    sm = cyv == 0; mm = cyv == 1
    print(f"[hier] Stage2-single acc (true single) = {(finalP[sm]==yv[sm]).mean():.4f} (n={sm.sum()})")
    print(f"[hier] Stage2-multi(OFDM) acc (true multi) = {(finalP[mm]==yv[mm]).mean():.4f} (n={mm.sum()})  <- narrowband-IQ limit")
    print(f"[hier] COMBINED fine acc = {(finalP==yv).mean():.4f}  (flat recognizer was ~0.57)")
    print(f"[hier] saved {CACHE / args.out}")


def cmd_oracle(args):
    """CEILING test (no training needed): use PERFECT GT boxes as detections, labeled+scored by the
    recognizer (score = recognizer softmax confidence). If oracle-OURS class-aware mAP > baseline, the
    idea works and the deployment gap (predicted boxes) is the fixable problem; if not, the recognizer
    itself is the bottleneck. Also reports recognizer top-1 acc and per-family mAP."""
    import torch
    from iqdet_metrics import class_aware_detection_map
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap(); nC = len(names); fams = [family_of(n) for n in names]
    use_hier_o = bool(getattr(args, "hier_model", None))  # oracle hier support
    if use_hier_o:
        _imh, _scl, _mcl = hier_classes(names)
        single_arr_o = np.array(_scl)
        model = make_hier_model(len(_scl), len(_mcl)).to(dev)
        model.load_state_dict(torch.load(CACHE / args.hier_model, map_location=dev)); model.eval()
    else:
        model = make_model(nC).to(dev); model.load_state_dict(torch.load(CACHE / args.model, map_location=dev)); model.eval()
    fs_def = 10_000_000.0
    H = W = 512
    _orc_rng = np.random.default_rng(0)
    GB, GL, PB, PS, PL, PLV = [], [], [], [], [], []
    pred_by_sid = None
    if getattr(args, "baseline_pred", None):
        coco_o = json.loads((MM / "coco_multiclass" / "annotations" / f"instances_{args.split}.json").read_text())
        id2stem_o = {im["id"]: Path(im["file_name"]).stem for im in coco_o["images"]}
        preds_o = json.loads(Path(args.baseline_pred).read_text())
        pred_by_sid = defaultdict(list)
        for d in preds_o:
            pred_by_sid[id2stem_o[d["image_id"]]].append(d)
    n_corr = n_tot = 0
    lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line); nq, fs = r["num_iq_samples"], r["sample_rate"]
        boxes, labs, params = [], [], []
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            boxes.append([inst["start_in_samples"]/nq, (inst["start_in_samples"]+inst["duration_in_samples"])/nq,
                          inst["center_freq"]/fs, inst["bandwidth"]/fs])
            labs.append(cid2idx[inst["category_id"]])
            params.append((int(inst["start_in_samples"]), int(inst["duration_in_samples"]), inst["center_freq"], inst["bandwidth"]))
        if not boxes:
            continue
        rp = raw_scene_path(args.split, r["sample_id"])
        if not rp.exists():
            continue
        iq = load_raw_iq(rp)
        iq_t = torch.from_numpy(np.ascontiguousarray(iq)).to(dev)
        snips, keep = [], []
        for bi, (s, d_, cf, bw) in enumerate(params):
            cf_use = cf + float(_orc_rng.uniform(-args.inject_cf_bins, args.inject_cf_bins)) * (fs / 512.0)
            bw_use = max(bw * (1.0 + float(_orc_rng.uniform(-args.inject_bw, args.inject_bw))), 1.0)
            ds = int(float(_orc_rng.uniform(-args.inject_t, args.inject_t)) * d_)
            dd = int(d_ * (1.0 + float(_orc_rng.uniform(-args.inject_t, args.inject_t))))
            a = max(0, s + ds); b2 = min(len(iq), a + max(dd, 16))
            snip = channelize_torch(iq_t, a, b2, cf_use, bw_use, fs, args.L, calibrate=args.calibrate, refine_bw=getattr(args, "refine_bw", False))
            if snip is not None:
                snips.append(snip); keep.append(bi)
        if not snips:
            continue
        with torch.no_grad():
            if use_hier_o:
                _out_o = model(torch.stack(snips))
                prob = torch.softmax(_out_o[1], dim=1)
                conf, _ps = prob.max(1); conf = conf.cpu().numpy()
                pred = single_arr_o[_ps.cpu().numpy()]
            else:
                prob = torch.softmax(model(torch.stack(snips)), dim=1)
                conf, pred = prob.max(1); conf = conf.cpu().numpy(); pred = pred.cpu().numpy()
        GB.append(torch.tensor(boxes, dtype=torch.float32)); GL.append(torch.tensor(labs, dtype=torch.long))
        pb = [boxes[bi] for bi in keep]
        PB.append(torch.tensor(pb, dtype=torch.float32)); PS.append(torch.tensor(conf, dtype=torch.float32)); PL.append(torch.tensor(pred.tolist(), dtype=torch.long))
        if pred_by_sid is not None:
            from iqdet_metrics import time_frequency_iou as _tfio
            pvb, pvl = [], []
            for d in pred_by_sid.get(r["sample_id"], []):
                x, y0, w, h = d["bbox"]; pvb.append([x/W, (x+w)/W, 0.5-(y0+h/2)/H, h/H]); pvl.append(cid2idx.get(d["category_id"], 0))
            if pvb:
                iouM = _tfio(torch.tensor(pb, dtype=torch.float32), torch.tensor(pvb, dtype=torch.float32))
                vis_keep = [(pvl[int(iouM[rr].argmax())] if float(iouM[rr].max()) > 0 else 0) for rr in range(len(pb))]
            else:
                vis_keep = [0] * len(pb)
            PLV.append(torch.tensor(vis_keep, dtype=torch.long))
        for jj, bi in enumerate(keep):
            n_tot += 1; n_corr += int(pred[jj] == labs[bi])
    m = class_aware_detection_map(PB, PS, PL, GB, GL, num_classes=nC)
    print(f"\n[oracle] PERFECT GT boxes + recognizer (model={args.model}, calibrate={args.calibrate})")
    print(f"[oracle] recognizer top-1 acc on GT boxes = {n_corr/max(n_tot,1):.4f} (n={n_tot})")
    print(f"[oracle] ORACLE-OURS(pure-IQ) class-aware mAP@.5:.95 = {m['class_bbox_mAP']:.4f}  @.5={m['class_bbox_mAP_50']:.4f}  @.75={m['class_bbox_mAP_75']:.4f}")
    if pred_by_sid is not None and PLV:
        mvis = class_aware_detection_map(PB, PS, PLV, GB, GL, num_classes=nC)
        print(f"[oracle] GTbox+VISION class-aware mAP@.5:.95 = {mvis['class_bbox_mAP']:.4f}  (SAME GT boxes & scores, only label source = vision)")
        print(f"[CLARIFY] perfect boxes: pure-IQ {m['class_bbox_mAP']:.4f} vs vision {mvis['class_bbox_mAP']:.4f} -> IQ NET gain under perfect localization = {m['class_bbox_mAP']-mvis['class_bbox_mAP']:+.4f}")
        print(f"[CLARIFY] vision localization gain = GTbox+vision {mvis['class_bbox_mAP']:.4f} - deploy-baseline ~0.524 = {mvis['class_bbox_mAP']-0.524:+.4f}  (if LARGE, oracle delta is mostly LOCALIZATION, not IQ)")
    print(f"[compare] deploy baseline (predicted boxes) ~0.52.")
    # per-family
    fam2 = defaultdict(list)
    for i, f in enumerate(fams):
        fam2[f].append(i)
    print(f"{'family':>8} {'#cls':>4} | {'oracleOURS':>10}")
    for fam, cats in sorted(fam2.items()):
        cs = set(cats)
        def filt(B, S, L_, byL):
            out = ([], [], [])
            for b, s, l in zip(B, S, L_):
                keep = [i for i, x in enumerate((byL if byL is not None else l).tolist()) if x in cs]
                out[0].append(b[keep] if len(keep) else b[:0]); out[1].append(s[keep] if len(keep) else s[:0]); out[2].append(l[keep] if len(keep) else l[:0])
            return out
        pbf, psf, plf = filt(PB, PS, PL, None)
        gbf, _, glf = filt(GB, [torch.ones(len(g)) for g in GL], GL, None)
        mm = class_aware_detection_map(pbf, psf, plf, gbf, glf, num_classes=nC)
        print(f"{fam:>8} {len(cats):>4} | {mm['class_bbox_mAP']:>10.4f}")


def cmd_build_jitter(args):
    """Build a training set from CLEAN GT boxes but with INJECTED realistic box errors
    (cf/bw/time jitter matching predicted-box deployment noise). Labels stay CLEAN (GT class)
    -> robustness to predicted-box channelization WITHOUT the label noise that sinks training
    directly on predicted boxes. GPU channelize. The honest fix for the deployment gap."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap()
    X, y = [], []
    rng = np.random.default_rng(args.seed)
    lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        r = json.loads(line); nq, fs = r["num_iq_samples"], r["sample_rate"]
        bin_hz = fs / 512.0
        rp = raw_scene_path(args.split, r["sample_id"])
        if not rp.exists():
            continue
        iq = load_raw_iq(rp)
        iq_t = torch.from_numpy(np.ascontiguousarray(iq)).to(dev)
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            lab = cid2idx[inst["category_id"]]
            s0 = int(inst["start_in_samples"]); d0 = int(inst["duration_in_samples"])
            cf0 = inst["center_freq"]; bw0 = inst["bandwidth"]
            for _ in range(args.reps):
                cf = cf0 + float(rng.uniform(-args.jitter_cf, args.jitter_cf)) * bin_hz
                bw = bw0 * float(rng.uniform(1 - args.jitter_bw, 1 + args.jitter_bw))
                ds = int(float(rng.uniform(-args.jitter_t, args.jitter_t)) * d0)
                dd = int(d0 * float(rng.uniform(1 - args.jitter_t, 1 + args.jitter_t)))
                a = max(0, s0 + ds); b = min(len(iq), a + max(dd, 16))
                snip = channelize_torch(iq_t, a, b, cf, max(bw, 1.0), fs, args.L, calibrate=args.calibrate)
                if snip is not None:
                    X.append(snip.cpu().numpy()); y.append(lab)
        if (k + 1) % 1000 == 0:
            print(f"[build-jitter] {k+1} samples, {len(y)} snippets", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    outp = CACHE / args.out
    np.savez(outp, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64))
    print(f"[build-jitter] saved {outp}  X={np.asarray(X).shape}")


def cmd_build_pred(args):
    """Build a recognizer training set from the BASELINE detector's PREDICTED boxes (not clean GT):
    channelize each predicted box (GPU, capped) and label it by the matched GT (IoU>=0.5). This makes
    the recognizer's training distribution match deployment (the end-to-end fix for the train/test mismatch)."""
    import torch
    from iqdet_metrics import time_frequency_iou
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap()
    H = W = 512; fs_def = 10_000_000.0
    coco = json.loads((MM / "coco_multiclass" / "annotations" / f"instances_{args.split}.json").read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    preds = json.loads(Path(args.baseline_pred).read_text())
    by_sid = defaultdict(list)
    for d in preds:
        if float(d.get("score", 1.0)) >= args.score_thr:
            by_sid[id2stem[d["image_id"]]].append(d)
    X, y, CFO = [], [], []
    nseen = 0
    for line in (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line); sid, nq, fs = r["sample_id"], r["num_iq_samples"], r["sample_rate"]
        dets = by_sid.get(sid, [])
        if not dets:
            continue
        gboxes, glab = [], []
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            gboxes.append([inst["start_in_samples"]/nq, (inst["start_in_samples"]+inst["duration_in_samples"])/nq,
                           inst["center_freq"]/fs, inst["bandwidth"]/fs])
            glab.append(cid2idx[inst["category_id"]])
        if not gboxes:
            continue
        rp = raw_scene_path(args.split, sid)
        if not rp.exists():
            continue
        iq = load_raw_iq(rp)
        iq_t = torch.from_numpy(np.ascontiguousarray(iq)).to(dev)
        pboxes = [[d["bbox"][0]/W, (d["bbox"][0]+d["bbox"][2])/W, 0.5-(d["bbox"][1]+d["bbox"][3]/2)/H, d["bbox"][3]/H] for d in dets]
        iou = time_frequency_iou(torch.tensor(pboxes, dtype=torch.float32), torch.tensor(gboxes, dtype=torch.float32))
        for pi, d in enumerate(dets):
            gi = int(iou[pi].argmax())
            if float(iou[pi, gi]) < args.min_iou:
                continue
            t0, t1 = pboxes[pi][0], pboxes[pi][1]
            fc = pboxes[pi][2] * fs_def; bw = max(pboxes[pi][3] * fs_def, 1.0)
            s0 = max(0, int(t0 * len(iq))); s1 = min(len(iq), int(t1 * len(iq)))
            snip = channelize_torch(iq_t, s0, s1, fc, bw, fs_def, args.L, calibrate=args.calibrate)
            if snip is not None:
                Nb = min(s1 - s0, 16384); D = int(max(1, round(fs_def / max(2.5 * bw, fs_def / max(Nb, 1)))))
                cfo_t = (gboxes[gi][2] - pboxes[pi][2]) * D  # residual CFO at channel rate (cycles/sample)
                X.append(snip.cpu().numpy()); y.append(glab[gi]); CFO.append(cfo_t)
        nseen += 1
        if nseen % 1000 == 0:
            print(f"[build-pred] {nseen} samples, {len(y)} matched boxes", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    outp = CACHE / args.out
    np.savez(outp, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64), cfo=np.asarray(CFO, dtype=np.float32))
    print(f"[build-pred] saved {outp}  X={np.asarray(X).shape}  cfo std={float(np.std(CFO)) if CFO else 0:.4f}")


def cmd_diag(args):
    """Pinpoint why OURS loses: (1) recognizer acc on CLEAN GT boxes (ceiling) vs (2) on
    PREDICTED boxes matched to GT, head-to-head VISION vs RECOGNIZER per detection + per family.
    If recog >> vision on matched dets but OURS mAP < baseline -> labeling/mAP bug.
    If recog << vision -> recognizer genuinely weaker (and check clean-vs-predicted gap = loc-error effect)."""
    import torch
    from iqdet_metrics import time_frequency_iou
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap(); nC = len(names); fams = [family_of(n) for n in names]
    H = W = 512; fs_def = 10_000_000.0
    model = make_model(nC).to(dev); model.load_state_dict(torch.load(CACHE / args.model, map_location=dev)); model.eval()

    coco = json.loads((MM / "coco_multiclass" / "annotations" / f"instances_{args.split}.json").read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    preds = json.loads(Path(args.baseline_pred).read_text())
    by_sid = defaultdict(list)
    for d in preds:
        by_sid[id2stem[d["image_id"]]].append(d)

    @torch.no_grad()
    def predict(snips):
        if not snips:
            return np.array([])
        out = []
        for i in range(0, len(snips), 512):
            xb = torch.from_numpy(np.asarray(snips[i:i+512], dtype=np.float32)).to(dev)
            out.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)

    gt_snips, gt_y = [], []                       # clean GT-box recognition
    pm_snips, pm_vis, pm_y = [], [], []           # matched predicted-box: snippet, vision label, gt label
    lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        r = json.loads(line); sid, nq, fs = r["sample_id"], r["num_iq_samples"], r["sample_rate"]
        rp = raw_scene_path(args.split, sid)
        if not rp.exists():
            continue
        iq = load_raw_iq(rp)
        gboxes, glab, gparams = [], [], []
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            gboxes.append([inst["start_in_samples"]/nq, (inst["start_in_samples"]+inst["duration_in_samples"])/nq, inst["center_freq"]/fs, inst["bandwidth"]/fs])
            glab.append(cid2idx[inst["category_id"]])
            gparams.append((int(inst["start_in_samples"]), int(inst["duration_in_samples"]), inst["center_freq"], inst["bandwidth"]))
        # (1) clean GT-box recognition
        for (s, d_, cf, bw), lab in zip(gparams, glab):
            s1 = min(len(iq), s + d_)
            if s1 - s >= 16:
                gt_snips.append(channelize(iq[s:s1], cf, bw, fs, args.L, calibrate=args.calibrate)); gt_y.append(lab)
        # (2) matched predicted boxes
        dets = by_sid.get(sid, [])
        if dets and gboxes:
            pboxes, plab = [], []
            for d in dets:
                x, y0, w, h = d["bbox"]
                pboxes.append([x/W, (x+w)/W, 0.5-(y0+h/2)/H, h/H]); plab.append(cid2idx.get(d["category_id"], 0))
            iou = time_frequency_iou(torch.tensor(pboxes, dtype=torch.float32), torch.tensor(gboxes, dtype=torch.float32))
            for pi in range(len(pboxes)):
                gi = int(iou[pi].argmax()); v = float(iou[pi, gi])
                if v >= 0.5:
                    x, y0, w, h = dets[pi]["bbox"]
                    t0, t1 = x/W, (x+w)/W; cf = (0.5-(y0+h/2)/H)*fs_def; bw = max((h/H)*fs_def, 1.0)
                    s0 = max(0, int(t0*len(iq))); s1 = min(len(iq), int(t1*len(iq)))
                    if s1 - s0 >= 16:
                        pm_snips.append(channelize(iq[s0:s1], cf, bw, fs_def, args.L, calibrate=args.calibrate))
                        pm_vis.append(plab[pi]); pm_y.append(glab[gi])
        if (k+1) % 2000 == 0:
            print(f"[diag] {k+1}/{len(lines)} samples", flush=True)

    gt_y = np.array(gt_y); pm_vis = np.array(pm_vis); pm_y = np.array(pm_y)
    gt_pred = predict(gt_snips); pm_rec = predict(pm_snips)
    fa = np.array([fams[i] for i in gt_y]) if len(gt_y) else np.array([])
    fap = np.array([fams[i] for i in pm_y]) if len(pm_y) else np.array([])

    print(f"\n[diag] split={args.split} calibrate={args.calibrate}")
    print(f"(1) recognizer acc on CLEAN GT boxes : {float((gt_pred==gt_y).mean()):.4f}  (n={len(gt_y)})")
    print(f"(2) matched predicted boxes (IoU>=.5, n={len(pm_y)}):")
    print(f"    VISION    label acc = {float((pm_vis==pm_y).mean()):.4f}")
    print(f"    RECOGNIZER label acc = {float((pm_rec==pm_y).mean()):.4f}  (return-to-IQ on predicted boxes)")
    print(f"\n{'family':>8} | {'recog@cleanGT':>13} {'vision@pred':>11} {'recog@pred':>10}")
    for fam in sorted(set(fams)):
        mg = fa == fam; mp = fap == fam
        rg = float((gt_pred[mg]==gt_y[mg]).mean()) if mg.sum() else float('nan')
        vv = float((pm_vis[mp]==pm_y[mp]).mean()) if mp.sum() else float('nan')
        rr = float((pm_rec[mp]==pm_y[mp]).mean()) if mp.sum() else float('nan')
        print(f"{fam:>8} | {rg:>13.3f} {vv:>11.3f} {rr:>10.3f}")
    print("\n[read] if recog@cleanGT >> recog@pred -> localization-error/train-test mismatch (fix: train on predicted-like boxes / e2e).")
    print("       if recog@cleanGT ~ vision@pred -> recognizer competitive, predicted-box channelize is the leak.")
    print("       if recog@pred >> vision@pred but OURS mAP < baseline -> labeling/mAP bug.")


def _energy_quality(iq_np, s0, s1, gfc, gbw, pfc, pbw, all_bands, gi, gtime, nq):
    """Signal-fidelity energy metrics on the channelizer's exact pred-window slice (capped MAXLEN).
    All frequencies NORMALIZED (cycles/sample, fftfreq). GT band = center_freq +- bw/2 (the channelize/
    render convention; _lower/_upper metadata are sign-flipped, NOT used). Returns
    (energy_coverage, energy_contamination, energy_in_window, neighbor_overlap)."""
    MAXLEN = 16384
    sl = iq_np[s0:s1]
    Nb = len(sl)
    if Nb < 16:
        return float("nan"), float("nan"), float("nan"), 0
    if Nb > MAXLEN:
        st = (Nb - MAXLEN) // 2
        sl = sl[st:st + MAXLEN]; Nb = MAXLEN
    P = np.abs(np.fft.fft(sl)) ** 2
    f = np.fft.fftfreq(Nb)  # normalized [-0.5,0.5)
    inband = np.zeros(Nb, dtype=bool)
    for (lo, hi) in all_bands:
        inband |= (f >= lo) & (f <= hi)
    outm = ~inband
    N0 = float(np.median(P[outm])) if outm.any() else 0.0
    Ps = np.clip(P - N0, 0.0, None)
    g0, g1 = gfc - abs(gbw) / 2.0, gfc + abs(gbw) / 2.0
    bw_eff = max(abs(pbw), 1.0 / Nb)
    p0, p1 = pfc - bw_eff / 2.0, pfc + bw_eff / 2.0
    mg = (f >= g0) & (f <= g1)
    mp = (f >= p0) & (f <= p1)
    e_gt = float(Ps[mg].sum()); e_pred = float(Ps[mp].sum())
    ecov = float(Ps[mg & mp].sum() / e_gt) if e_gt > 0 else float("nan")
    econt = float(Ps[mp & (~mg)].sum() / e_pred) if e_pred > 0 else float("nan")
    nover = 0
    for j, (lo, hi) in enumerate(all_bands):
        if j != gi and not (hi < g0 or lo > g1):
            nover = 1; break
    # time-truncation: GT-band energy in the pred window vs the full GT window
    ewin = float("nan")
    gs, ge = gtime; gs = max(0, gs); ge = min(int(nq), ge)
    if ge - gs >= 16:
        gsl = iq_np[gs:ge]
        if len(gsl) > MAXLEN:
            st = (len(gsl) - MAXLEN) // 2; gsl = gsl[st:st + MAXLEN]
        Pg = np.abs(np.fft.fft(gsl)) ** 2; fg = np.fft.fftfreq(len(gsl))
        Pg = np.clip(Pg - N0, 0.0, None)
        egf = float(Pg[(fg >= g0) & (fg <= g1)].sum())
        ewin = float(min(1.0, e_gt / egf)) if egf > 0 else float("nan")
    return ecov, econt, ewin, nover


def cmd_diag_quality(args):
    """OBSERVATIONAL DIAGNOSTIC: per matched predicted box, dump SIGNAL-fidelity quality metrics
    (cf-error, containment, coverage, energy coverage/contamination) ALONGSIDE the recognizer's
    per-box correctness. Tests the hypothesis that IoU (a vision metric) predicts return-to-IQ
    recognition success POORLY vs the signal-fidelity metrics. Recognizer path mirrors cmd_bridge
    for deployment parity. Output = one JSONL record per matched (IoU>=match-iou) predicted box."""
    import torch
    from iqdet_metrics import time_frequency_iou
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cid2idx, names = load_catmap(); nC = len(names); fams = [family_of(n) for n in names]
    H = W = 512; fs_def = 10_000_000.0

    use_hier = bool(getattr(args, "hier_model", None))
    if use_hier:
        is_multi_h, single_cls_h, multi_cls_h = hier_classes(names)
        single_arr_h = np.array(single_cls_h)
        model = make_hier_model(len(single_cls_h), len(multi_cls_h)).to(dev)
        model.load_state_dict(torch.load(CACHE / args.hier_model, map_location=dev)); model.eval()
    else:
        model = make_model(nC).to(dev); model.load_state_dict(torch.load(CACHE / args.model, map_location=dev)); model.eval()

    coco = json.loads((MM / "coco_multiclass" / "annotations" / f"instances_{args.split}.json").read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    preds = json.loads(Path(args.baseline_pred).read_text())
    by_sid = defaultdict(list)
    for d in preds:
        s = id2stem[d["image_id"]]
        if float(d.get("score", 1.0)) >= args.score_thr:
            by_sid[s].append(d)

    outf = open(args.out, "w", encoding="utf-8")
    nrec = 0; ncorr = 0; ntot = 0; sc_max = 0.0
    lines = (RAWDS / "metadata" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        r = json.loads(line); sid, nq, fs = r["sample_id"], r["num_iq_samples"], r["sample_rate"]
        dets = by_sid.get(sid, [])
        if not dets:
            continue
        gboxes, glab, gband, gtime, gsnr = [], [], [], [], []
        for inst in r["instances"]:
            if inst["category_id"] not in cid2idx:
                continue
            cfn = inst["center_freq"] / fs; bwn = inst["bandwidth"] / fs
            s_, d_ = int(inst["start_in_samples"]), int(inst["duration_in_samples"])
            gboxes.append([s_ / nq, (s_ + d_) / nq, cfn, bwn])
            glab.append(cid2idx[inst["category_id"]])
            gband.append((cfn - abs(bwn) / 2.0, cfn + abs(bwn) / 2.0))
            gtime.append((s_, s_ + d_))
            gsnr.append(float(inst.get("snr_db", float("nan"))))
        if not gboxes:
            continue
        rp = raw_scene_path(args.split, sid)
        if not rp.exists():
            continue
        iq = load_raw_iq(rp)
        iq = np.ascontiguousarray(iq)
        iq_t = torch.from_numpy(iq).to(dev); nqs = iq_t.numel()
        pboxes, plab = [], []
        for d in dets:
            x, y0, w, h = d["bbox"]
            pboxes.append([x / W, (x + w) / W, 0.5 - (y0 + h / 2) / H, h / H])
            plab.append(cid2idx.get(d["category_id"], 0))
        iou = time_frequency_iou(torch.tensor(pboxes, dtype=torch.float32), torch.tensor(gboxes, dtype=torch.float32))
        oracle_lab = {}
        if getattr(args, "with_oracle", False):
            osnips, ogi = [], []
            for gj in range(len(gboxes)):
                gfc_, gbw_ = gboxes[gj][2], gboxes[gj][3]
                gs = max(0, int(gtime[gj][0])); ge = min(nqs, int(gtime[gj][1]))
                osnip = channelize_torch(iq_t, gs, ge, gfc_ * fs_def, max(abs(gbw_) * fs_def, 1.0), fs_def, args.L,
                                         calibrate=args.calibrate, refine_cf=args.refine_cf,
                                         power_cal=args.power_cal, refine_bw=args.refine_bw)
                if osnip is not None:
                    osnips.append(osnip); ogi.append(gj)
            if osnips:
                with torch.no_grad():
                    oout = model(torch.stack(osnips)); ologits = oout[1] if use_hier else oout
                    oconf = torch.softmax(ologits, dim=1).max(1).values.cpu().numpy()
                    opred = single_arr_h[ologits.argmax(1).cpu().numpy()] if use_hier else ologits.argmax(1).cpu().numpy()
                for jj, gj in enumerate(ogi):
                    oracle_lab[gj] = (int(opred[jj]), int(int(opred[jj]) == glab[gj]), round(float(oconf[jj]), 5))
        recs, snips, snip_idx = [], [], []
        for pi in range(len(pboxes)):
            gi = int(iou[pi].argmax()); v = float(iou[pi, gi])
            if v < args.match_iou and not args.include_unmatched:
                continue
            t0, t1, pfc, pbw = pboxes[pi]
            gt0, gt1, gfc, gbw = gboxes[gi]
            a_t0, a_t1 = min(t0, t1), max(t0, t1)
            a_f0, a_f1 = pfc - 0.5 * abs(pbw), pfc + 0.5 * abs(pbw)
            b_t0, b_t1 = min(gt0, gt1), max(gt0, gt1)
            b_f0, b_f1 = gfc - 0.5 * abs(gbw), gfc + 0.5 * abs(gbw)
            it = max(0.0, min(a_t1, b_t1) - max(a_t0, b_t0))
            ifq = max(0.0, min(a_f1, b_f1) - max(a_f0, b_f0))
            inter = it * ifq
            area_p = max(0.0, a_t1 - a_t0) * max(0.0, a_f1 - a_f0)
            area_g = max(0.0, b_t1 - b_t0) * max(0.0, b_f1 - b_f0)
            gt_cont = inter / area_g if area_g > 0 else 0.0
            pr_cont = inter / area_p if area_p > 0 else 0.0
            fcov = ifq / (b_f1 - b_f0) if (b_f1 - b_f0) > 0 else 0.0
            tcov = it / (b_t1 - b_t0) if (b_t1 - b_t0) > 0 else 0.0
            bw_ratio = (a_f1 - a_f0) / (b_f1 - b_f0) if (b_f1 - b_f0) > 0 else 0.0
            cf_err_bins = (pfc - gfc) * 512.0
            s0 = max(0, int(t0 * nqs)); s1 = min(nqs, int(t1 * nqs)); Nb = max(s1 - s0, 1)
            Dd = int(max(1, round(fs_def / max(2.5 * max(abs(pbw) * fs_def, 1.0), fs_def / Nb))))
            cf_err_cyc = (gfc - pfc) * Dd
            if args.selfcheck:
                union = area_p + area_g - inter
                my_iou = inter / union if union > 0 else 0.0
                assert abs(my_iou - v) < 1e-3, f"IoU mismatch {my_iou} vs {v}"
            ecov, econt, ewin, nover = _energy_quality(iq, s0, s1, gfc, gbw, pfc, pbw, gband, gi, gtime[gi], nqs)
            snip = channelize_torch(iq_t, s0, s1, pfc * fs_def, max(abs(pbw) * fs_def, 1.0), fs_def, args.L,
                                    calibrate=args.calibrate, refine_cf=args.refine_cf,
                                    power_cal=args.power_cal, refine_bw=args.refine_bw)
            rec = {
                "sid": sid, "pi": pi, "gi": gi, "iou": round(v, 5),
                "gt_containment": round(gt_cont, 5), "pred_containment": round(pr_cont, 5),
                "freq_coverage": round(fcov, 5), "time_coverage": round(tcov, 5), "bw_ratio": round(bw_ratio, 5),
                "cf_err_bins": round(cf_err_bins, 5), "cf_err_bins_abs": round(abs(cf_err_bins), 5),
                "cf_err_cyc": round(cf_err_cyc, 6), "cf_err_cyc_abs": round(abs(cf_err_cyc), 6),
                "energy_coverage": (round(ecov, 5) if ecov == ecov else None),
                "energy_contamination": (round(econt, 5) if econt == econt else None),
                "energy_in_window": (round(ewin, 5) if ewin == ewin else None),
                "neighbor_overlap": int(nover),
                "gt_label": int(glab[gi]), "gt_name": names[glab[gi]],
                "vision_label": int(plab[pi]), "vision_correct": int(plab[pi] == glab[gi]),
                "iq_label": None, "iq_conf": None, "recog_correct": None,
                "family": fams[glab[gi]], "snr_db": gsnr[gi], "det_score": round(float(dets[pi].get("score", 1.0)), 5),
                "nb_samples": int(Nb), "decim_D": int(Dd),
                "oracle_label": (oracle_lab[gi][0] if gi in oracle_lab else None),
                "oracle_correct": (oracle_lab[gi][1] if gi in oracle_lab else None),
                "oracle_conf": (oracle_lab[gi][2] if gi in oracle_lab else None),
            }
            if snip is not None:
                snips.append(snip); snip_idx.append(len(recs))
            recs.append(rec)
        if snips:
            with torch.no_grad():
                out = model(torch.stack(snips))
                logits = out[1] if use_hier else out
                conf = torch.softmax(logits, dim=1).max(1).values.cpu().numpy()
                pred = single_arr_h[logits.argmax(1).cpu().numpy()] if use_hier else logits.argmax(1).cpu().numpy()
            for jj, ri in enumerate(snip_idx):
                recs[ri]["iq_label"] = int(pred[jj]); recs[ri]["iq_conf"] = round(float(conf[jj]), 5)
                recs[ri]["recog_correct"] = int(int(pred[jj]) == recs[ri]["gt_label"])
                ntot += 1; ncorr += recs[ri]["recog_correct"]
        for rec in recs:
            outf.write(json.dumps(rec) + "\n"); nrec += 1
        if (k + 1) % 2000 == 0:
            print(f"[diag-quality] {k+1}/{len(lines)} samples, {nrec} boxes", flush=True)
    outf.close()
    print(f"[diag-quality] wrote {nrec} matched-box records -> {args.out}")
    if ntot:
        print(f"[diag-quality] recog_correct mean = {ncorr/ntot:.4f} (n={ntot})  <- should match cmd_diag recog@pred")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--split", required=True); b.add_argument("--L", type=int, default=1024); b.add_argument("--limit", type=int, default=0); b.add_argument("--max-missing-frac", type=float, default=0.0, help="Fail if more than this fraction of scenes have no readable raw IQ."); b.set_defaults(func=cmd_build)
    t = sub.add_parser("train"); t.add_argument("--train-cache", required=True); t.add_argument("--val-cache", default=""); t.add_argument("--out", default="recognizer.pth"); t.add_argument("--epochs", type=int, default=40); t.add_argument("--bs", type=int, default=256); t.add_argument("--aug-cfo", type=float, default=0.02); t.add_argument("--seed", type=int, default=20260619); t.set_defaults(func=cmd_train)
    g = sub.add_parser("bridge"); g.add_argument("--hier-coarse-route", action="store_true", help="use coarse head to pick single/multi branch per crop (enables OFDM routing)"); g.add_argument("--input-rep", default="iq", choices=["iq","diff","iqdiff"]); g.add_argument("--split", default="test"); g.add_argument("--baseline-pred", required=True); g.add_argument("--model", default="recognizer.pth"); g.add_argument("--L", type=int, default=1024); g.add_argument("--calibrate", action="store_true"); g.add_argument("--score-thr", type=float, default=0.0); g.add_argument("--limit", type=int, default=0); g.add_argument("--hier-model", default=None, help="use hierarchical recognizer single-branch (e.g. recognizer_hier.pth)"); g.add_argument("--e2e-model", default=None, help="use e2e CFO-refine recognizer (e.g. recognizer_e2e.pth)"); g.add_argument("--refine-model", default=None, help="use learned-CFO-refine recognizer trained on clean+injection (recognizer_refine.pth)"); g.add_argument("--refine-cf", action="store_true", help="sub-bin CF refinement (fine-FFT centroid) before channelize"); g.add_argument("--power-cal", action="store_true", help="4th-power blind CFO calibration (for PSK/QAM)"); g.add_argument("--refine-bw", action="store_true", help="estimate occupied bandwidth from PSD (fixes the dominant deployment-gap cause)"); g.add_argument("--ours-score-recog", action="store_true", help="rank OURS detections by det x recognition confidence (not det score alone)"); g.add_argument("--wbf-iou", type=float, default=1.0, help="WBF: feed channelizer the score-weighted-avg cf/bw of each det IoU-cluster (mAP box set unchanged); cuts cf variance ~1/sqrt(K)"); g.add_argument("--nms-iou", type=float, default=1.0, help="class-agnostic NMS IoU on predicted boxes before recog (dedup redundant detections)"); g.add_argument("--iq-families", default=None, help="comma-separated families routed to return-to-IQ (default psk,qam,ask)"); g.add_argument("--class-nms-iou", type=float, default=1.0, help="per-class NMS IoU applied to EACH method's own labels (symmetric fair dedup)"); g.add_argument("--fidelity-gate-mode", default="off", choices=["off", "gt"], help="route IQ_SET boxes to IQ only if size-fit (bw_ratio) in [lo,hi]; gt=GT bw_ratio (upper-bound, not deployable)"); g.add_argument("--fidelity-gate-lo", type=float, default=0.85); g.add_argument("--fidelity-gate-hi", type=float, default=1.2); g.add_argument("--oracle-fix", default="off", help="UPPER-BOUND box-fix: replace predicted box param(s) with matched GT before channelize. one of off/bw/cf/time/bwcf/bwcftime (substring match)"); g.set_defaults(func=cmd_bridge)
    tr = sub.add_parser("train-refine"); tr.add_argument("--train-cache", required=True); tr.add_argument("--val-cache", required=True); tr.add_argument("--out", default="recognizer_refine.pth"); tr.add_argument("--epochs", type=int, default=45); tr.add_argument("--bs", type=int, default=256); tr.add_argument("--inject-cf", type=float, default=0.05, help="synthetic cf injection range (cycles/sample)"); tr.add_argument("--lam", type=float, default=2.0); tr.add_argument("--seed", type=int, default=20260619); tr.set_defaults(func=cmd_train_refine)
    th = sub.add_parser("train-hier"); th.add_argument("--label-smooth", type=float, default=0.0); th.add_argument("--cosine", action="store_true"); th.add_argument("--ema", type=float, default=0.0); th.add_argument("--mixup", type=float, default=0.0); th.add_argument("--input-rep", default="iq", choices=["iq","diff","iqdiff"]); th.add_argument("--train-cache", required=True); th.add_argument("--val-cache", required=True); th.add_argument("--out", default="recognizer_hier.pth"); th.add_argument("--epochs", type=int, default=40); th.add_argument("--bs", type=int, default=256); th.add_argument("--aug-cfo", type=float, default=0.02); th.add_argument("--seed", type=int, default=20260619); th.set_defaults(func=cmd_train_hier)
    orc = sub.add_parser("oracle"); orc.add_argument("--hier-model", default=None); orc.add_argument("--split", default="test"); orc.add_argument("--model", default="recognizer_L1024.pth"); orc.add_argument("--L", type=int, default=1024); orc.add_argument("--calibrate", action="store_true"); orc.add_argument("--limit", type=int, default=2000); orc.add_argument("--inject-cf-bins", type=float, default=0.0, help="inject +-J STFT-bins of cf error to GT boxes (precision-threshold sweep)"); orc.add_argument("--inject-bw", type=float, default=0.0, help="inject +-J fractional bandwidth error"); orc.add_argument("--inject-t", type=float, default=0.0, help="inject +-J fractional time error"); orc.add_argument("--refine-bw", action="store_true", help="estimate occupied bw from PSD (test the bw-refinement fix)"); orc.add_argument("--baseline-pred", default=None, help="if set, also compute GTbox+vision mAP (clarify localization vs IQ gain)"); orc.set_defaults(func=cmd_oracle)
    bj = sub.add_parser("build-jitter"); bj.add_argument("--split", default="train"); bj.add_argument("--out", default="trainjit_L1024.npz"); bj.add_argument("--L", type=int, default=1024); bj.add_argument("--reps", type=int, default=2); bj.add_argument("--jitter-cf", type=float, default=1.5, help="cf jitter in STFT bins"); bj.add_argument("--jitter-bw", type=float, default=0.25); bj.add_argument("--jitter-t", type=float, default=0.06); bj.add_argument("--calibrate", action="store_true"); bj.add_argument("--limit", type=int, default=0); bj.add_argument("--seed", type=int, default=20260619); bj.set_defaults(func=cmd_build_jitter)
    bp = sub.add_parser("build-pred"); bp.add_argument("--split", default="train"); bp.add_argument("--baseline-pred", required=True); bp.add_argument("--out", default="trainpred_L1024.npz"); bp.add_argument("--L", type=int, default=1024); bp.add_argument("--score-thr", type=float, default=0.1); bp.add_argument("--min-iou", type=float, default=0.5); bp.add_argument("--calibrate", action="store_true"); bp.set_defaults(func=cmd_build_pred)
    te = sub.add_parser("train-e2e"); te.add_argument("--train-cache", required=True); te.add_argument("--val-cache", default=""); te.add_argument("--out", default="recognizer_e2e.pth"); te.add_argument("--epochs", type=int, default=50); te.add_argument("--bs", type=int, default=256); te.add_argument("--lam", type=float, default=3.0); te.add_argument("--seed", type=int, default=20260619); te.set_defaults(func=cmd_train_e2e)
    dg = sub.add_parser("diag"); dg.add_argument("--split", default="test"); dg.add_argument("--baseline-pred", required=True); dg.add_argument("--model", default="recognizer.pth"); dg.add_argument("--L", type=int, default=1024); dg.add_argument("--calibrate", action="store_true"); dg.add_argument("--limit", type=int, default=0); dg.set_defaults(func=cmd_diag)
    dq = sub.add_parser("diag-quality"); dq.add_argument("--split", default="test"); dq.add_argument("--baseline-pred", required=True); dq.add_argument("--out", required=True); dq.add_argument("--model", default="recognizer.pth"); dq.add_argument("--hier-model", default=None); dq.add_argument("--L", type=int, default=1024); dq.add_argument("--match-iou", type=float, default=0.5); dq.add_argument("--score-thr", type=float, default=0.0); dq.add_argument("--calibrate", action="store_true"); dq.add_argument("--refine-cf", action="store_true"); dq.add_argument("--power-cal", action="store_true"); dq.add_argument("--refine-bw", action="store_true"); dq.add_argument("--include-unmatched", action="store_true"); dq.add_argument("--limit", type=int, default=0); dq.add_argument("--selfcheck", action="store_true"); dq.add_argument("--with-oracle", action="store_true", help="also channelize+recognize each GT box (perfect frame) to isolate box-quality net effect"); dq.set_defaults(func=cmd_diag_quality)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
