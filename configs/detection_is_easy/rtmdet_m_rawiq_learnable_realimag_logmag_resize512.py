# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_raw_iq_filterbank_hardshort_resize512.py'

# Axis-A A4 cell: raw-IQ -> LEARNABLE filterbank (warm-started from Fourier, then trained end-to-end),
# channel_mode='realimag_logmag' (3ch) -- identical to the A5 frozen-Fourier phase-IN cell EXCEPT the
# filterbank is trainable. Direct frozen-vs-learnable front-end comparison on class-aware detect+recognize.
# Stats are the Fourier-init output stats (correct at init; drift during training is absorbed by BN/norm).
model = dict(
    data_preprocessor=dict(
        trainable_filterbank=True,
    ),
)

work_dir = 'work_dirs/torchsig_mmdet_rawiq_learnable_realimag_logmag_rtmdet_m'
