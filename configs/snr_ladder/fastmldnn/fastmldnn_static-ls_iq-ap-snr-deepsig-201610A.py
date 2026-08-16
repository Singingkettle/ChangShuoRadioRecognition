# Static label smoothing route of the SNR-aware audit spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = 'work_dirs/amc/deepsig201610A/fastmldnn_static-ls'
model = dict(head=dict(loss=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1.0)))
