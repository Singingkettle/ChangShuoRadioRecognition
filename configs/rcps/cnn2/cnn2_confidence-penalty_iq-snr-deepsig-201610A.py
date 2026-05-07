_base_ = [
    '../_base_/models/cnn2_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/cnn2_confidence-penalty'

method_name = 'confidence_penalty_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='ConfidencePenaltyLoss',
            beta=0.1,
            loss_weight=1.0),
    ),
)
