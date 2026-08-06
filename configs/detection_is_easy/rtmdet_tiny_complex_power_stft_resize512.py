# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_complex_stft_resize512.py'

model = dict(
    backbone=dict(
        type='ComplexPowerCSPNeXt',
        input_channels=2,
        init_cfg=None,
    ),
)

work_dir = 'work_dirs/torchsig_mmdet_complex_power_stft_resize512'
