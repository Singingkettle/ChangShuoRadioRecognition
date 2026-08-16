method_name = 'confidence_penalty_0p1'

model = dict(
    head=dict(
        loss=dict(
            type='ConfidencePenaltyLoss',
            beta=0.1,
            loss_weight=1.0),
    ),
)
