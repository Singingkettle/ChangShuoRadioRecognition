# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_raw_iq_filterbank_3ch_resize512.py'

# Multi-resolution raw-IQ front end that keeps all scale evidence instead of
# collapsing the 257/513/1025-sample branches with one global gate. The detector
# sees [real, imag, log|.|] for each scale, i.e. 9 channels in total.
model = dict(
    data_preprocessor=dict(
        mean=[
            0.06780968703714052,
            0.07019388975766105,
            0.06729979692022425,
            0.05032386277821388,
            0.02831820611720559,
            -0.013639966185337471,
            0.3339245264651254,
            0.4103365376358852,
            0.37677047634497285,
        ],
        std=[
            136.52209587329588,
            191.79330285154202,
            151.868261446986,
            136.09692482830428,
            195.11386177584643,
            153.37538396728573,
            0.5914754543652666,
            0.6579132360945008,
            0.6673284868639882,
        ],
        filterbank_kernel_sizes=(257, 513, 1025),
        filterbank_strides=(256, 512, 1024),
        filterbank_fusion='stack',
    ),
    backbone=dict(input_channels=9),
)

work_dir = 'work_dirs/iqdet_raw_iq_multiscale_filterbank_stack9_resize512_smoke'
