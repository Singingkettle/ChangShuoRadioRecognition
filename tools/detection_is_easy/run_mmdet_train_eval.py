# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from export_mmdet_summary import write_outputs  # noqa: E402
from run_mmdet_smoke import apply_tensor_stats, category_names, data_subdir, maybe_stub_mmcv_ext  # noqa: E402


def set_num_classes(node, num_classes: int) -> None:
    if isinstance(node, Mapping):
        if "num_classes" in node:
            node["num_classes"] = num_classes
        for value in node.values():
            set_num_classes(value, num_classes)
    elif isinstance(node, (list, tuple)):
        for value in node:
            set_num_classes(value, num_classes)


def configure_test_cfg(test_cfg, *, score_thr: float, nms_pre: int, max_per_img: int, nms_iou: float | None) -> None:
    if not isinstance(test_cfg, dict):
        return
    if "score_thr" in test_cfg:
        test_cfg.score_thr = score_thr
    if "nms_pre" in test_cfg:
        test_cfg.nms_pre = nms_pre
    if "max_per_img" in test_cfg:
        test_cfg.max_per_img = max_per_img
    nms_cfg = test_cfg.get("nms", None)
    if nms_iou is not None and isinstance(nms_cfg, dict) and "iou_threshold" in nms_cfg:
        nms_cfg.iou_threshold = float(nms_iou)
    for value in test_cfg.values():
        configure_test_cfg(value, score_thr=score_thr, nms_pre=nms_pre, max_per_img=max_per_img, nms_iou=nms_iou)


def stft_tensor_channels(coco_root: Path) -> int | None:
    summary_path = coco_root.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stats = summary.get("stft_tensor_stats") or {}
    mean = stats.get("mean")
    if isinstance(mean, list) and mean:
        return int(len(mean))
    channels = summary.get("channels")
    if isinstance(channels, int):
        return int(channels)
    return None


def adapt_stft_channels(cfg, channels: int | None) -> None:
    if channels is None:
        return
    if "data_preprocessor" in cfg.model:
        ptype = str(cfg.model.data_preprocessor.get("type", ""))
        if ptype in {"RawIQFilterbankDetDataPreprocessor", "RawIQPassThroughDetDataPreprocessor"}:
            # Raw-IQ front ends define their own input channels via channel_mode
            # (realimag=2, realimag_logmag=3, logmag2ch=2). Do NOT force the offline
            # STFT3 channel count (3) from the shared coco_multiclass summary.
            return
    if "data_preprocessor" in cfg.model:
        mean = cfg.model.data_preprocessor.get("mean", None)
        std = cfg.model.data_preprocessor.get("std", None)
        if mean is not None and len(mean) != channels:
            cfg.model.data_preprocessor.mean = [0.0] * channels
        if std is not None and len(std) != channels:
            cfg.model.data_preprocessor.std = [1.0] * channels
    backbone = cfg.model.get("backbone", None)
    if isinstance(backbone, Mapping):
        if "input_channels" in backbone:
            backbone.input_channels = channels
        if "in_channels" in backbone:
            backbone.in_channels = channels


def adapt_loader_expected_channels(node, channels: int | None) -> None:
    if channels is None:
        return
    if isinstance(node, Mapping):
        if node.get("type") in {"LoadComplexStftFromFile", "LoadTensorMemmapFromCOCOStem"}:
            node["expected_channels"] = channels
        for value in node.values():
            adapt_loader_expected_channels(value, channels)
    elif isinstance(node, (list, tuple)):
        for value in node:
            adapt_loader_expected_channels(value, channels)


def override_pipeline_data_roots(node, memmap_root: str | None, raw_root: str | None) -> None:
    """Repoint the heavy data roots baked into a config's Load* transforms.

    ``--root`` only moves the COCO annotations. The packed STFT memmap and the raw IQ
    scenes are named separately inside each config (``memmap_root`` / ``raw_root``), so
    a dataset living outside the repository needs these overrides as well.
    """
    if memmap_root is None and raw_root is None:
        return
    if isinstance(node, Mapping):
        if memmap_root is not None and "memmap_root" in node:
            node["memmap_root"] = memmap_root
        if raw_root is not None and "raw_root" in node:
            node["raw_root"] = raw_root
        for value in node.values():
            override_pipeline_data_roots(value, memmap_root, raw_root)
    elif isinstance(node, (list, tuple)):
        for value in node:
            override_pipeline_data_roots(value, memmap_root, raw_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/torchsig_mini/coco")
    parser.add_argument(
        "--memmap-root",
        default=None,
        help="Override the packed-STFT memmap directory named in the config "
             "(default: the config's own repo-relative path).",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Override the raw-IQ scene directory named in the config "
             "(used by the raw-IQ input-representation cells).",
    )
    parser.add_argument("--config", default="configs/detection_is_easy/rtmdet_tiny_torchsig_smoke.py")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for validation/test dataloaders. Defaults to --batch-size for throughput-fair screening.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--val-sample-limit", type=int, default=None)
    parser.add_argument("--test-sample-limit", type=int, default=None)
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split used by the training dataloader. Latest TorchSig official configs use names such as wideband_clean_train_all.",
    )
    parser.add_argument(
        "--train-ann-split",
        default=None,
        help=(
            "Annotation split name for training. Defaults to --train-split. "
            "Use this for teacher-refined labels while keeping --train-data-split on the original data directory."
        ),
    )
    parser.add_argument(
        "--train-data-split",
        default=None,
        help="Data-prefix split name for training tensors/raw files. Defaults to --train-split.",
    )
    parser.add_argument(
        "--eval-split",
        default="val",
        help="Split used by the validation loop during training. Use train for overfit diagnostics.",
    )
    parser.add_argument(
        "--eval-ann-split",
        default=None,
        help="Annotation split name for validation. Defaults to --eval-split.",
    )
    parser.add_argument(
        "--eval-data-split",
        default=None,
        help="Data-prefix split name for validation tensors/raw files. Defaults to --eval-split.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Split used by the test loop in eval-only mode.",
    )
    parser.add_argument(
        "--test-ann-split",
        default=None,
        help="Annotation split name for test/eval-only. Defaults to --test-split.",
    )
    parser.add_argument(
        "--test-data-split",
        default=None,
        help="Data-prefix split name for test/eval-only tensors/raw files. Defaults to --test-split.",
    )
    parser.add_argument("--logger-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--optimizer", choices=["config", "SGD", "Adam", "AdamW"], default="config")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay.")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Optional max-norm gradient clipping injected into cfg.optim_wrapper.",
    )
    parser.add_argument("--warmup-iters", type=int, default=None, help="Override iter-based LinearLR warmup length.")
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--max-per-img", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--resume-from", default=None, help="Resume MMEngine training from a checkpoint path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for eval-only mode. Defaults to --resume-from when omitted.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Run test loop only with a loaded checkpoint.")
    parser.add_argument(
        "--dump-results",
        action="store_true",
        help="Write COCO-format test predictions through CocoMetric outfile_prefix.",
    )
    parser.add_argument("--experiment-name", default="stft_rtmdet_formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_batch_size = args.eval_batch_size or args.batch_size
    used_stub = maybe_stub_mmcv_ext()

    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    coco_root = ROOT / args.root
    classes = category_names(coco_root)

    cfg = Config.fromfile(ROOT / args.config)
    cfg.data_root = str(coco_root).replace("\\", "/") + "/"
    cfg.classes = classes
    cfg.num_classes = len(classes)
    cfg.work_dir = str((ROOT / args.work_dir).resolve()).replace("\\", "/")
    checkpoint_arg = args.checkpoint or args.resume_from
    if args.eval_only and checkpoint_arg is None:
        raise ValueError("--eval-only requires --checkpoint or --resume-from")

    if args.resume_from and not args.eval_only:
        # MMEngine checkpoints include runner state objects. PyTorch >=2.6 defaults
        # torch.load to weights_only=True, which blocks those trusted local objects.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        resume_path = (ROOT / args.resume_from).resolve() if not Path(args.resume_from).is_absolute() else Path(args.resume_from)
        cfg.load_from = str(resume_path).replace("\\", "/")
        cfg.resume = True
    elif checkpoint_arg:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        checkpoint_path = (ROOT / checkpoint_arg).resolve() if not Path(checkpoint_arg).is_absolute() else Path(checkpoint_arg)
        cfg.load_from = str(checkpoint_path).replace("\\", "/")
        cfg.resume = False
    else:
        cfg.load_from = None
        cfg.resume = False
    set_num_classes(cfg.model, len(classes))
    if "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    if isinstance(cfg.model.get("test_cfg", None), dict):
        cfg.model.test_cfg.score_thr = args.score_thr
        cfg.model.test_cfg.nms_pre = args.nms_pre
        cfg.model.test_cfg.max_per_img = args.max_per_img
        if args.nms_iou is not None and isinstance(cfg.model.test_cfg.get("nms", None), dict):
            cfg.model.test_cfg.nms.iou_threshold = float(args.nms_iou)
    configure_test_cfg(
        cfg.model.get("test_cfg", {}),
        score_thr=args.score_thr,
        nms_pre=args.nms_pre,
        max_per_img=args.max_per_img,
        nms_iou=args.nms_iou,
    )
    tensor_channels = stft_tensor_channels(coco_root)
    adapt_stft_channels(cfg, tensor_channels)
    adapt_loader_expected_channels(getattr(cfg, "_cfg_dict", cfg), tensor_channels)
    apply_tensor_stats(cfg, coco_root)

    train_ann_split = args.train_ann_split or args.train_split
    train_data_split = args.train_data_split or args.train_split
    eval_ann_split = args.eval_ann_split or args.eval_split
    eval_data_split = args.eval_data_split or args.eval_split
    test_ann_split = args.test_ann_split or args.test_split
    test_data_split = args.test_data_split or args.test_split

    loader_splits = (
        ("train_dataloader", train_ann_split, train_data_split),
        ("val_dataloader", eval_ann_split, eval_data_split),
        ("test_dataloader", test_ann_split, test_data_split),
    )
    for loader_name, ann_split, data_split in loader_splits:
        loader = cfg[loader_name]
        loader.num_workers = args.num_workers
        loader.persistent_workers = args.num_workers > 0
        if loader_name == "train_dataloader":
            loader.batch_size = args.batch_size
        else:
            loader.batch_size = eval_batch_size
        loader.dataset.data_root = cfg.data_root
        loader.dataset.metainfo = dict(classes=classes)
        loader.dataset.ann_file = f"annotations/instances_{ann_split}.json"
        loader.dataset.data_prefix = dict(img=f"{data_split}/{data_subdir(coco_root, data_split)}/")
        adapt_loader_expected_channels(loader.dataset.pipeline, tensor_channels)
        override_pipeline_data_roots(loader.dataset.pipeline, args.memmap_root, args.raw_root)
        if loader_name == "train_dataloader" and args.train_sample_limit is not None:
            loader.dataset.indices = args.train_sample_limit
        if loader_name == "val_dataloader" and args.val_sample_limit is not None:
            loader.dataset.indices = args.val_sample_limit
        if loader_name == "val_dataloader" and ann_split == train_ann_split and args.train_sample_limit is not None:
            loader.dataset.indices = args.train_sample_limit
        if loader_name == "test_dataloader" and args.test_sample_limit is not None:
            loader.dataset.indices = args.test_sample_limit

    cfg.val_evaluator.ann_file = cfg.data_root + f"annotations/instances_{eval_ann_split}.json"
    cfg.test_evaluator.ann_file = cfg.data_root + f"annotations/instances_{test_ann_split}.json"
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.dump_results:
        source_dir = work_dir / "source_data"
        source_dir.mkdir(parents=True, exist_ok=True)
        prefix = (source_dir / "test_predictions").resolve()
        cfg.test_evaluator.outfile_prefix = str(prefix).replace("\\", "/")
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.val_interval
    cfg.default_hooks.logger.interval = args.logger_interval
    cfg.default_hooks.checkpoint.interval = args.checkpoint_interval
    if args.optimizer != "config":
        base_lr = args.lr if args.lr is not None else float(cfg.optim_wrapper.optimizer.get("lr", 1e-3))
        weight_decay = args.weight_decay
        if weight_decay is None:
            weight_decay = float(cfg.optim_wrapper.optimizer.get("weight_decay", 0.0))
        opt = dict(type=args.optimizer, lr=base_lr, weight_decay=weight_decay)
        if args.optimizer in {"Adam", "AdamW"}:
            opt["betas"] = (0.9, 0.999)
        cfg.optim_wrapper.optimizer = opt
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    if args.weight_decay is not None:
        cfg.optim_wrapper.optimizer.weight_decay = args.weight_decay
    if args.clip_grad_norm is not None:
        cfg.optim_wrapper.clip_grad = dict(max_norm=float(args.clip_grad_norm), norm_type=2)
    if args.warmup_iters is not None:
        for scheduler in cfg.get("param_scheduler", []):
            if isinstance(scheduler, dict) and scheduler.get("type") == "LinearLR" and not scheduler.get("by_epoch", False):
                scheduler.end = args.warmup_iters
    cfg.randomness = dict(seed=args.seed, deterministic=False)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        cfg.device = args.device
    else:
        cfg.device = "cpu"

    generated = work_dir / "generated_rtmdet_torchsig_train_eval.py"
    cfg.dump(str(generated))
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,          # the literal command, for copy-paste reproduction
        "args": vars(args),
        "classes": classes,
        "device": cfg.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "used_mmcv_lite_stub": used_stub,
        "seed": args.seed,
        "resume_from": args.resume_from,
        "checkpoint": args.checkpoint,
        "eval_only": args.eval_only,
        "generated_config": str(generated),
    }
    (work_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    mode = "eval" if args.eval_only else "train"
    print(f"[mmdet-{mode}] generated config: {generated}")
    print(f"[mmdet-{mode}] classes={classes} device={cfg.device} epochs={args.epochs} batch_size={args.batch_size}")
    if used_stub:
        print(f"[mmdet-{mode}] warning: using mmcv-lite import stub; full CUDA mmcv is recommended for long runs")

    runner = Runner.from_cfg(cfg)
    if args.eval_only:
        runner.test()
        write_outputs(work_dir, args.experiment_name, split_label="test")
    else:
        runner.train()
        write_outputs(work_dir, args.experiment_name)
    print(f"[mmdet-{mode}] ok work_dir={work_dir}")


if __name__ == "__main__":
    main()
