# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_raw_iq_multiscale_filterbank_gated3ch_layerwise_lr_resize512.py'

custom_hooks = [
    dict(type='FreezeDetectorExceptFilterbankHook', trainable_prefix='data_preprocessor.filterbank'),
]

work_dir = 'work_dirs/iqdet_raw_iq_multiscale_filterbank_gated3ch_freeze_detector_smoke'
