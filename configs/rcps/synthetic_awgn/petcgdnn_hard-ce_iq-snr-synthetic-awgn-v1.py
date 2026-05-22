_base_ = [
    '../_base_/models/petcgdnn_iq-snr-synthetic-awgn-v1.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/petcgdnn_hard-ce'
method_name = 'hard_ce'

model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
