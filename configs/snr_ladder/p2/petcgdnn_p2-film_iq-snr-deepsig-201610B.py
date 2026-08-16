# SNR-FiLM conditioning route (P2): the backbone consumes the SNR through FiLM modulation; the only audited route that changes phi.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610B.py']

model = dict(
    type='SNRFiLMClassifier',
    reliability_key='snr',
    backbone=dict(
        type='PETCGDNNFiLM',
        num_classes=10,
        snr_min=-20.0, snr_max=18.0,
        film_scale=0.1),
    head=dict(loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
