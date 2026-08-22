# Copyright (c) Shuo Chang and contributors. Licensed under the Apache License, Version 2.0.
# Input rep: diff (first-order phase difference z[n] = c[n] c*[n-1], Re/Im). Same
# recipe as the iq base; only the front-end representation changes.
# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition", IEEE Transactions on Wireless Communications (in preparation).
_base_ = ['./returniq_resnet1d_iq_120e_wideband.py']

model = dict(backbone=dict(input_rep='diff'))
