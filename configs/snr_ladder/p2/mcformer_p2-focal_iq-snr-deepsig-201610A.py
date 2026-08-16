# Softmax focal loss route of the SNR-aware supervision spectrum (P2).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/mcformer_iq-snr-deepsig-201610A.py']

model = dict(
    head=dict(
        loss=dict(type='SoftmaxFocalLoss', gamma=2.0, loss_weight=1.0)))
