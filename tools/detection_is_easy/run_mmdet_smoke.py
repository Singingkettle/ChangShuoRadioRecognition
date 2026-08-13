# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

MMCV_EXT_STUB_TAG = "<iqdet-mmcv-lite-stub>"


def category_names(coco_root: Path) -> tuple[str, ...]:
    ann_dir = coco_root / "annotations"
    candidate_names = [
        "instances_train.json",
        "instances_wideband_clean_train_all.json",
        "instances_wideband_impaired_train_all.json",
        "instances_val.json",
        "instances_wideband_clean_val_all.json",
        "instances_wideband_impaired_val_all.json",
    ]
    categories = []
    for name in candidate_names:
        ann_path = ann_dir / name
        if ann_path.exists():
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            categories = ann.get("categories", [])
            if categories:
                break
    if not categories and ann_dir.exists():
        for ann_path in sorted(ann_dir.glob("instances_*.json")):
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            categories = ann.get("categories", [])
            if categories:
                break
    if not categories:
        summary = coco_root.parent / "summary.json"
        categories = json.loads(summary.read_text(encoding="utf-8")).get("categories", []) if summary.exists() else []
    categories = sorted(categories, key=lambda cat: cat["id"])
    return tuple(cat["name"] for cat in categories) or ("signal",)


def data_subdir(coco_root: Path, split: str) -> str:
    if (coco_root / split / "tensors").exists():
        return "tensors"
    return "images"


def apply_tensor_stats(cfg, coco_root: Path, *, require: bool = False) -> None:
    """Inject the dataset's per-channel mean/std into the data preprocessor.

    The STFT3 configs deliberately ship identity statistics (mean 0, std 1) and rely on
    this function to replace them with the values recorded in the dataset's summary.json.
    The real values have std ~12.8, so falling through silently means training on
    unnormalised data -- which looks like a reproduction failure but is a plumbing failure.
    Every path that declines to apply the statistics now says so out loud, and
    ``require=True`` turns "config still has identity stats" into a hard error.
    """

    def _identity(mean, std) -> bool:
        return (mean is not None and std is not None
                and all(float(v) == 0.0 for v in mean) and all(float(v) == 1.0 for v in std))

    def _bail(reason: str) -> None:
        prep = cfg.model.get("data_preprocessor", {}) if "model" in cfg else {}
        mean, std = prep.get("mean"), prep.get("std")
        if _identity(mean, std):
            msg = (f"[tensor-stats] NOT applied ({reason}); the config still carries identity "
                   f"mean/std, so training would see unnormalised data.")
            if require:
                raise SystemExit(msg + " Re-run without --require-tensor-stats to override.")
            print(msg + " Pass --require-tensor-stats to make this fatal.", flush=True)
        else:
            print(f"[tensor-stats] not applied ({reason}); using the config's own mean/std.",
                  flush=True)

    summary_path = coco_root.parent / "summary.json"
    if not summary_path.exists():
        return _bail(f"no summary.json at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = summary.get("stft_tensor_stats")
    if not stats:
        return _bail(f"summary.json has no 'stft_tensor_stats' key")
    if "data_preprocessor" in cfg.model:
        preprocessor_type = str(cfg.model.data_preprocessor.get("type", ""))
        if preprocessor_type == "RawIQFilterbankDetDataPreprocessor":
            # Online raw-IQ front ends have their own statistics after the
            # learnable/Fourier filterbank. Do not overwrite them with offline
            # STFT tensor statistics from the shared COCO annotation root.
            print("[tensor-stats] skipped for RawIQFilterbankDetDataPreprocessor "
                  "(front end has its own constants).", flush=True)
            return
        stat_mean = [float(v) for v in stats["mean"]]
        stat_std = [float(max(float(v), 1e-6)) for v in stats["std"]]
        configured_mean = cfg.model.data_preprocessor.get("mean", None)
        if configured_mean is not None and len(configured_mean) != len(stat_mean):
            return _bail(
                f"summary has {len(stat_mean)} channels but config has {len(configured_mean)}"
            )
        cfg.model.data_preprocessor.mean = stat_mean
        cfg.model.data_preprocessor.std = stat_std
        print(f"[tensor-stats] applied from {summary_path}: "
              f"mean={[round(v, 6) for v in stat_mean]} std={[round(v, 6) for v in stat_std]}",
              flush=True)


def set_num_classes(node, num_classes: int) -> None:
    if isinstance(node, dict):
        if "num_classes" in node:
            node["num_classes"] = num_classes
        for value in node.values():
            set_num_classes(value, num_classes)
    elif isinstance(node, (list, tuple)):
        for value in node:
            set_num_classes(value, num_classes)


def configure_test_cfg(test_cfg, *, score_thr: float, nms_pre: int, max_per_img: int) -> None:
    if not isinstance(test_cfg, dict):
        return
    if "score_thr" in test_cfg:
        test_cfg.score_thr = score_thr
    if "nms_pre" in test_cfg:
        test_cfg.nms_pre = nms_pre
    if "max_per_img" in test_cfg:
        test_cfg.max_per_img = max_per_img
    for value in test_cfg.values():
        configure_test_cfg(value, score_thr=score_thr, nms_pre=nms_pre, max_per_img=max_per_img)


def maybe_stub_mmcv_ext() -> bool:
    """Allow registry import with mmcv-lite when unused CUDA ops are imported.

    MMDetection imports mask/roi/nms modules while registering datasets and hooks.
    The RTMDet smoke train path below does not need those extensions, but importing
    mmdet on Windows with mmcv-lite otherwise fails before the model can build.
    If any missing op is actually called, the stub raises a clear error.
    """

    try:
        import mmcv._ext  # type: ignore  # noqa: F401

        return False
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("mmcv._ext")
    module.__file__ = MMCV_EXT_STUB_TAG
    module.__package__ = "mmcv"
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)

    def nms(bboxes, scores, iou_threshold: float, offset: int = 0):
        """Pure PyTorch NMS fallback for smoke validation.

        This is intentionally simple and only covers the standard axis-aligned
        NMS signature used by RTMDet inference. Full experiments should use
        the OpenMMLab CUDA extension.
        """

        import torch

        if bboxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=bboxes.device)

        x1, y1, x2, y2 = bboxes.unbind(dim=1)
        widths = (x2 - x1 + offset).clamp(min=0)
        heights = (y2 - y1 + offset).clamp(min=0)
        areas = widths * heights
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[current], x1[rest])
            yy1 = torch.maximum(y1[current], y1[rest])
            xx2 = torch.minimum(x2[current], x2[rest])
            yy2 = torch.minimum(y2[current], y2[rest])
            inter_w = (xx2 - xx1 + offset).clamp(min=0)
            inter_h = (yy2 - yy1 + offset).clamp(min=0)
            inter = inter_w * inter_h
            union = (areas[current] + areas[rest] - inter).clamp(min=torch.finfo(bboxes.dtype).eps)
            iou = inter / union
            order = rest[iou <= iou_threshold]

        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=bboxes.device)

    def __getattr__(name: str):
        def missing_op(*args, **kwargs):
            raise NotImplementedError(
                f"mmcv._ext.{name} was called, but this smoke environment uses mmcv-lite. "
                "Install full mmcv with CUDA extensions in a Linux/OpenMMLab environment for full training."
            )

        return missing_op

    module.nms = nms  # type: ignore[attr-defined]
    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules["mmcv._ext"] = module
    return True


def mmcv_ext_is_stubbed() -> bool:
    """True when `mmcv._ext` is our stand-in rather than a compiled extension."""
    module = sys.modules.get("mmcv._ext")
    if module is not None:
        return getattr(module, "__file__", "") == MMCV_EXT_STUB_TAG
    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return True
    return False


def patch_focal_loss_for_mmcv_lite() -> bool:
    """Route mmdet's `FocalLoss` to its pure-PyTorch kernel when mmcv has no CUDA ops.

    RTMDet trains under mmcv-lite because its classification loss (QualityFocalLoss) is
    pure PyTorch. FCOS, ATSS, RetinaNet -- anything whose head uses `FocalLoss` -- does
    not: for a CUDA tensor mmdet dispatches to `mmcv.ops.sigmoid_focal_loss`, which calls
    the compiled `mmcv._ext` kernel and dies in the first backward pass.

    mmdet already ships the identical computation as `py_sigmoid_focal_loss`; the CUDA op
    is a speed optimisation, not a different loss. Pointing the dispatch at the Python
    kernel therefore reproduces the same numbers on an mmcv-lite install, and mirrors
    exactly what the branch below it already does on CPU.

    Returns True when the patch was applied.
    """
    if not mmcv_ext_is_stubbed():
        return False

    import torch.nn.functional as F
    from mmdet.models.losses import focal_loss as mmdet_focal_loss

    def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                           reduction="mean", avg_factor=None):
        # The CUDA op takes class indices; the Python one takes one-hot targets. This is
        # the same conversion mmdet performs in its own CPU branch.
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]
        return mmdet_focal_loss.py_sigmoid_focal_loss(
            pred, target, weight, gamma=gamma, alpha=alpha,
            reduction=reduction, avg_factor=avg_factor)

    mmdet_focal_loss.sigmoid_focal_loss = sigmoid_focal_loss
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/torchsig_mini/coco")
    parser.add_argument("--config", default="configs/detection_is_easy/rtmdet_tiny_torchsig_smoke.py")
    parser.add_argument("--work-dir", default="work_dirs/torchsig_mmdet_smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    used_stub = maybe_stub_mmcv_ext()

    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    patch_focal_loss_for_mmcv_lite()

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    coco_root = root / args.root
    classes = category_names(coco_root)

    cfg = Config.fromfile(root / args.config)
    cfg.data_root = str(coco_root).replace("\\", "/") + "/"
    cfg.classes = classes
    cfg.num_classes = len(classes)
    cfg.work_dir = str((root / args.work_dir).resolve()).replace("\\", "/")
    cfg.load_from = None
    set_num_classes(cfg.model, len(classes))
    if "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    configure_test_cfg(cfg.model.get("test_cfg", {}), score_thr=0.01, nms_pre=300, max_per_img=50)
    apply_tensor_stats(cfg, coco_root)

    for loader_name, split in (("train_dataloader", "train"), ("val_dataloader", "val"), ("test_dataloader", "test")):
        loader = cfg[loader_name]
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.dataset.data_root = cfg.data_root
        loader.dataset.metainfo = dict(classes=classes)
        loader.dataset.ann_file = f"annotations/instances_{split}.json"
        loader.dataset.data_prefix = dict(img=f"{split}/{data_subdir(coco_root, split)}/")

    cfg.val_evaluator.ann_file = cfg.data_root + "annotations/instances_val.json"
    cfg.test_evaluator.ann_file = cfg.data_root + "annotations/instances_test.json"
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 999
    cfg.default_hooks.logger.interval = 1

    if args.device.startswith("cuda") and torch.cuda.is_available():
        cfg.device = args.device
    else:
        cfg.device = "cpu"

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    generated = Path(cfg.work_dir) / "generated_rtmdet_torchsig_smoke.py"
    cfg.dump(str(generated))
    print(f"[mmdet] generated config: {generated}")
    print(f"[mmdet] classes={classes} device={cfg.device}")
    if used_stub:
        print("[mmdet] warning: using mmcv-lite import stub for unused extension ops")

    runner = Runner.from_cfg(cfg)
    runner.train()
    print("[mmdet] ok")


if __name__ == "__main__":
    main()
