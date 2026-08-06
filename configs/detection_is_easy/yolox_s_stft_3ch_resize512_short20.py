# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './yolox_s_complex_stft_resize512_short20.py'

model = dict(
    data_preprocessor=dict(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    ),
    backbone=dict(input_channels=3),
)
