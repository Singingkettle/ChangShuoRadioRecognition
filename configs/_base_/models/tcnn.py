model = dict(
    type='TCNN',
    backbone=dict(
        type='TanhNet',
        signal_length=128,  # this valu is consistent with the split_length in datasets
    ),
    filter_head=dict(
        type='SeparatorHead',
        loss_reg=dict(
            type='MSELoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()
