# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_complex_stft_resize512.py'

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5,
    ),
    neck=dict(
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        in_channels=128,
        feat_channels=128,
    ),
)

work_dir = 'work_dirs/torchsig_mmdet_complex_stft_resize512_rtmdet_s'
