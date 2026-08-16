# SNR curriculum weighting route of the SNR-aware supervision spectrum (P2).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610B.py']

model = dict(
    head=dict(
        loss=dict(
            type='SNRCurriculumCELoss',
            snr_min=-20.0, snr_max=18.0, tau=2.0,
            warmup_iters=60000, min_weight=0.0, loss_weight=1.0)))
