# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './yolox_s_complex_stft_resize512.py'

# Paired short-schedule control for 20-epoch RF experiments. The default
# MMDetection YOLOX config inherits a 300e COCO schedule with a 5e warmup; this
# schedule keeps the detector family fixed while making a short 20e run less
# warmup-dominated. Use this only as a paired raw/STFT ablation.
max_epochs = 20
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=999)
param_scheduler = [
    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end=1, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=1,
        end=max_epochs,
        T_max=max_epochs - 1,
        eta_min=5e-6,
        convert_to_iter_based=True,
    ),
]
