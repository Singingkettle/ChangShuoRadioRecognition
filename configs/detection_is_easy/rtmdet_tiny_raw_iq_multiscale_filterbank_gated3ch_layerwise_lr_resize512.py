# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_raw_iq_filterbank_3ch_layerwise_lr_resize512.py'

# Multi-resolution raw-IQ front end for the small/short-signal AP50 bottleneck.
# The gated sum keeps the detector input at [real, imag, log|.|] = 3 channels,
# so the RTMDet detector checkpoint remains compatible. Scale logits are biased
# toward the long-support branch, then allowed to adapt with the filterbank.
model = dict(
    data_preprocessor=dict(
        filterbank_kernel_sizes=(257, 513, 1025),
        filterbank_strides=(256, 512, 1024),
        filterbank_fusion='gated_sum',
        filterbank_scale_logits_init=(-2.0, 0.0, 2.0),
    ),
    backbone=dict(input_channels=3),
)

work_dir = 'work_dirs/iqdet_raw_iq_multiscale_filterbank_gated3ch_layerwise_lr_smoke'
