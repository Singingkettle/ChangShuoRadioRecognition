method_name = 'static_ls_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothingCrossEntropyLoss',
            smoothing=0.1,
            loss_weight=1.0),
    ),
)
