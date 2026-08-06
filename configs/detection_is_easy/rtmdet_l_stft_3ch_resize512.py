# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_stft_3ch_resize512.py'

model = dict(
    backbone=dict(
        deepen_factor=1.0,
        widen_factor=1.0,
    ),
    neck=dict(
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
    ),
    bbox_head=dict(
        in_channels=256,
        feat_channels=256,
    ),
)

work_dir = 'work_dirs/torchsig_mmdet_stft_3ch_resize512_rtmdet_l'
