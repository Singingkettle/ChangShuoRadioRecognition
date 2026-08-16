# Confidence-penalty route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = [
    '../_base_/models/cnn2_iq-snr-deepsig-201610A.py',
]

work_dir = 'work_dirs/amc/deepsig201610A/cnn2_confidence-penalty'

method_name = 'confidence_penalty_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='ConfidencePenaltyLoss',
            beta=0.1,
            loss_weight=1.0),
    ),
)
