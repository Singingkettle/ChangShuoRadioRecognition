# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_raw_iq_filterbank_hardshort_resize512.py'

# Stage-A phase-ablation OUT cell: magnitude-only (phase DISCARDED). SAME frozen-Fourier front-end as the
# phase-IN cell (rtmdet_m_raw_iq_filterbank_hardshort = realimag_logmag); only channel_mode + channel count
# differ -> clean isolation of phase utility for class-aware detect+recognize. The harness was patched so
# adapt_stft_channels/apply_tensor_stats skip RawIQFilterbank preprocessors, so these 2ch stats stand and
# input_channels is NOT forced to 3.
model = dict(
    data_preprocessor=dict(
        channel_mode='logmag2ch',
        mean=[0.221728, 0.221728],
        std=[0.330376, 0.330376],
    ),
    backbone=dict(input_channels=2),
)

work_dir = 'work_dirs/torchsig_mmdet_rawiq_fourier_logmag2ch_rtmdet_m'
