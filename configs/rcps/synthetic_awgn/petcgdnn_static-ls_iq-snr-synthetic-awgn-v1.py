_base_ = [
    '../_base_/models/petcgdnn_iq-snr-synthetic-awgn-v1.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/petcgdnn_static-ls'
method_name = 'static_ls_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothingCrossEntropyLoss',
            smoothing=0.1,
            loss_weight=1.0),
    ),
)
