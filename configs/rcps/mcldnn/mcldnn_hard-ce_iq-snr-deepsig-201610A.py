_base_ = [
    '../_base_/models/mcldnn_iq-snr-deepsig-201610A.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mcldnn_hard-ce'

method_name = 'hard_ce'

model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
