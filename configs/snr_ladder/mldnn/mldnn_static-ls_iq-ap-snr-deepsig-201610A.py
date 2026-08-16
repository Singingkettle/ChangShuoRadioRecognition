# Static label smoothing route of the SNR-aware audit spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/mldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = 'work_dirs/amc/deepsig201610A/mldnn_static-ls'

model = dict(head=dict(loss_amc_merge=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1), loss_amc_ap=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1), loss_amc_iq=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1)))
