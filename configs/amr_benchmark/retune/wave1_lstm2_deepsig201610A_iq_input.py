"""Wave-1 retune: LSTM2 @ RML2016.10A — raw I/Q + L2 norm (Keras path)."""

_base_ = [
    '../../_base_/datasets/deepsig/iq-shape-L-F-deepsig201610A.py',
    '../../_base_/schedules/amc.py',
    '../../_base_/runtimes/amc.py',
]

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='LSTM2',
        num_classes=11,
        init_cfg=[
            dict(type='Xavier', layer='Linear', distribution='uniform'),
            dict(type='LSTM', layer='LSTM', gain=1),
        ],
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
