# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_raw_iq_filterbank_hardshort_resize512.py'

# Axis-A cell: raw-IQ -> frozen-Fourier filterbank, channel_mode='realimag' (2ch, real+imag only, NO explicit
# log-mag channel). Phase IS present (carried by real+imag), but the magnitude is not handed to the net as a
# separate channel. Compared to the phase-IN cell (realimag_logmag 3ch) this isolates the VALUE OF THE EXPLICIT
# LOG-MAG CHANNEL; compared to the phase-OUT cell (logmag2ch) it shares only the 2ch budget. Harness patched so
# input_channels=2 stands. Stats = real/imag of the Fourier output.
model = dict(
    data_preprocessor=dict(
        channel_mode='realimag',
        mean=[-0.00005, -0.000306],
        std=[0.96207, 0.962656],
    ),
    backbone=dict(input_channels=2),
)

work_dir = 'work_dirs/torchsig_mmdet_rawiq_fourier_realimag_rtmdet_m'
