# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_raw_iq_filterbank_3ch_resize512.py'

# Stabilization setting for the in-graph raw-IQ filterbank experiment.
# The global LR is intentionally tiny for the mature detector; the custom key
# gives the native-complex filterbank a 10x multiplier so it can adapt without
# disturbing RTMDet's ranking and box heads.
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

work_dir = 'work_dirs/iqdet_raw_iq_filterbank_rtmdet_tiny_layerwise_lr_smoke'
