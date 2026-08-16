# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ["../_base_/models/mcformer_iq-snr-deepsig-201610B.py"]

model = dict(
    head=dict(
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0)))
