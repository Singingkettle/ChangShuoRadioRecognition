method_name = 'hard_ce'

model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
