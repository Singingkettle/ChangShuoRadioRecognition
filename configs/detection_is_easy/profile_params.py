# -*- coding: utf-8 -*-
# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Count parameters of RTMDet tiny/s/m/l on STFT3 (57-class) for the complexity table.
Run: python configs/detection_is_easy/profile_params.py"""
import sys
sys.path.insert(0, ".")
from run_mmdet_smoke import maybe_stub_mmcv_ext
maybe_stub_mmcv_ext()
sys.path.insert(0, "configs/detection_is_easy")
import mmdet_plugins  # noqa: F401
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

init_default_scope("mmdet")
CFG = "configs/detection_is_easy/rtmdet_{}_stft3_tensor_memmap_resize512.py"
for name in ["tiny", "s", "m", "l"]:
    try:
        c = Config.fromfile(CFG.format(name), import_custom_modules=False)
        if "bbox_head" in c.model:
            c.model.bbox_head.num_classes = 57
        m = MODELS.build(c.model)
        n = sum(p.numel() for p in m.parameters())
        nb = sum(p.numel() for p in m.backbone.parameters())
        print(f"{name}: total={n/1e6:.2f}M backbone={nb/1e6:.2f}M")
    except Exception as e:
        print(f"{name}: ERROR {type(e).__name__}: {e}")
print("PARAMS_DONE")
