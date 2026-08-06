# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_raw_iq_filterbank_3ch_2048_resize512.py'

# Keep RTMDet stable while letting the native-complex front end adapt faster.
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            'data_preprocessor.filterbank': dict(lr_mult=10.0, decay_mult=0.0),
        },
    )
)

work_dir = 'work_dirs/iqdet_raw_iq_filterbank_rtmdet_tiny_2048_layerwise_lr_smoke'
