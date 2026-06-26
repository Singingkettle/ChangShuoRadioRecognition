"""CPU build+instantiate smoke test for AMR-Benchmark configs.

Validates (without any GPU) that a config can:
  1. be parsed by mmengine.Config,
  2. build the model via MODELS.build,
  3. build the train dataset and pull one sample through the pipeline,
  4. collate a tiny batch and run a single CPU forward (loss + predict).

Run:
    CUDA_VISIBLE_DEVICES="" python tools/amr_benchmark/_smoke_test.py \
        configs/mcldnn/mcldnn_iq-deepsig-201801A.py ...
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from mmengine.config import Config
from mmengine.registry import init_default_scope

import csrr  # noqa: F401  (register modules)
from csrr.registry import MODELS, DATASETS


def smoke(cfg_path: str) -> bool:
    print(f"\n{'='*70}\n[SMOKE] {cfg_path}\n{'='*70}")
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get("default_scope", "csrr"))

    # ---- 1. build model ----
    model = MODELS.build(cfg.model)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] model built: {type(model).__name__}  params={n_params/1e6:.3f}M")

    # ---- 2. build train dataset + one sample ----
    # cache=True np.loads every sample at init; with ~1M+ samples that is
    # both slow and an IO/RAM contention risk against live training. Build a
    # tiny temp annotation (first few + last few entries) so we still exercise
    # the real pipeline/model without scanning the whole corpus.
    import json
    import tempfile

    ds_cfg = dict(cfg.train_dataloader["dataset"])
    data_root = ds_cfg["data_root"]
    ann_path = Path(data_root) / ds_cfg["ann_file"]
    ann = json.loads(ann_path.read_text())
    full_n = len(ann["data_list"])
    subset = ann["data_list"][:6] + ann["data_list"][-6:]
    tiny = dict(metainfo=ann["metainfo"], data_list=subset)
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(tiny, tf)
    tf.close()
    ds_cfg["ann_file"] = tf.name
    ds_cfg["data_root"] = data_root
    print(f"[info] full train size={full_n}; smoke subset={len(subset)}")
    dataset = DATASETS.build(ds_cfg)
    print(f"[ok] train dataset built: {type(dataset).__name__}  len={len(dataset)}")
    sample = dataset[0]
    inputs = sample["inputs"]
    data_sample = sample["data_samples"]
    print(f"[ok] sample[0] inputs shape={tuple(inputs.shape)} dtype={inputs.dtype} "
          f"label={int(data_sample.gt_label)}")

    # ---- 3. collate a tiny batch via the dataloader collate_fn ----
    from mmengine.dataset import default_collate, pseudo_collate
    from torch.utils.data import DataLoader
    collate = cfg.train_dataloader.get("collate_fn", dict(type="pseudo_collate"))
    collate_type = collate.get("type", "pseudo_collate")
    collate_fn = default_collate if collate_type == "default_collate" else pseudo_collate
    batch = collate_fn([dataset[i] for i in range(min(4, len(dataset)))])

    # data_preprocessor lives on the model
    dp = model.data_preprocessor
    data = dp(batch, training=True)

    # ---- 4. forward: loss (train mode) + predict (eval mode) ----
    # Some backbones (e.g. DAE) branch on self.training and only return their
    # auxiliary reconstruction outputs in train mode, so the loss path must be
    # exercised with model.train() to match the real training loop.
    with torch.no_grad():
        model.train()
        losses = model(data["inputs"], data["data_samples"], mode="loss")
        print(f"[ok] loss forward: { {k: float(v) for k,v in losses.items()} }")
        model.eval()
        preds = model(data["inputs"], data["data_samples"], mode="predict")
        scores = preds[0].pred_score
        print(f"[ok] predict forward: batch={len(preds)} num_classes={scores.numel()} "
              f"argmax={int(scores.argmax())}")

    # sanity: head dim must match dataset classes
    num_classes = scores.numel()
    meta_classes = getattr(dataset, "metainfo", {}).get("classes", None)
    if meta_classes is not None:
        print(f"[ok] dataset classes={len(meta_classes)} head_dim={num_classes}")
        assert len(meta_classes) == num_classes, (
            f"HEAD MISMATCH: head={num_classes} vs classes={len(meta_classes)}")
    print(f"[PASS] {cfg_path}")
    return True


if __name__ == "__main__":
    paths = sys.argv[1:]
    results = {}
    for p in paths:
        try:
            results[p] = smoke(p)
        except Exception:
            results[p] = False
            traceback.print_exc()
    print(f"\n{'='*70}\nSUMMARY")
    for p, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {p}")
    sys.exit(0 if all(results.values()) else 1)
