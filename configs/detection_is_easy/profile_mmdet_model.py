# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    _p = Path(__file__).resolve()
    for _up in [_p, *_p.parents]:
        if (_up / "tools" / "train.py").exists() and (_up / "csrr").is_dir():
            return _up
    raise RuntimeError("CSRR repo root not found above " + str(_p))


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from run_mmdet_smoke import apply_tensor_stats, category_names, maybe_stub_mmcv_ext  # noqa: E402
from run_mmdet_train_eval import set_num_classes  # noqa: E402


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be C,H,W, for example 3,512,512")
    return parts[0], parts[1], parts[2]


class DetectorForwardWrapper:
    def __init__(self, detector):
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                feats = self.model.extract_feat(x)
                return self.model.bbox_head.forward(feats)

        self.module = _Wrapper(detector)


def build_model(config: Path, root: Path | None, device: str):
    import torch
    from mmengine.config import Config
    from mmdet.registry import MODELS

    maybe_stub_mmcv_ext()
    cfg = Config.fromfile(str(config))
    if root is not None:
        coco_root = root
        classes = category_names(coco_root)
        cfg.data_root = str(coco_root).replace("\\", "/") + "/"
        cfg.classes = classes
        cfg.num_classes = len(classes)
        apply_tensor_stats(cfg, coco_root)
        set_num_classes(cfg.model, len(classes))
    if "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    detector = MODELS.build(cfg.model)
    detector.eval()
    detector.to(torch.device(device))
    return detector, cfg


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}


def measure_latency(module, shape: tuple[int, int, int], device: str, warmup: int, iters: int) -> dict[str, float]:
    import torch

    x = torch.randn((1, *shape), device=device)
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            module(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "latency_ms_forward_mean": 1000.0 * elapsed / max(iters, 1),
        "latency_iters": float(iters),
        "latency_warmup": float(warmup),
    }


def complexity(module, shape: tuple[int, int, int]) -> dict[str, object]:
    try:
        from mmengine.analysis import get_model_complexity_info

        info = get_model_complexity_info(
            module,
            input_shape=shape,
            show_table=False,
            show_arch=False,
        )
        return {
            "flops": info.get("flops"),
            "flops_str": info.get("flops_str"),
            "activations": info.get("activations"),
            "activations_str": info.get("activations_str"),
            "complexity_error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on third-party analyzer coverage.
        return {
            "flops": None,
            "flops_str": None,
            "activations": None,
            "activations_str": None,
            "complexity_error": repr(exc),
        }


def write_outputs(record: dict[str, object], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)
    print(f"[profile] wrote {out}")
    print(f"[profile] wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=None, help="Optional COCO root for class count and tensor statistics.")
    parser.add_argument("--shape", type=parse_shape, default=(3, 512, 512))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-latency", action="store_true", help="Only report params/complexity; skip timed forward passes.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model, cfg = build_model(ROOT / args.config, ROOT / args.root if args.root else None, device)
    wrapper = DetectorForwardWrapper(model).module.to(device)
    record: dict[str, object] = {
        "config": args.config,
        "root": args.root,
        "shape": "x".join(str(v) for v in args.shape),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_type": cfg.model.get("type", "unknown"),
        "backbone_type": cfg.model.get("backbone", {}).get("type", "unknown"),
    }
    record.update(count_parameters(model))
    record.update(complexity(wrapper, args.shape))
    if args.skip_latency:
        record.update(
            {
                "latency_ms_forward_mean": None,
                "latency_iters": 0.0,
                "latency_warmup": 0.0,
            }
        )
    else:
        record.update(measure_latency(wrapper, args.shape, device, args.warmup, args.iters))
    write_outputs(record, ROOT / args.out)


if __name__ == "__main__":
    main()
