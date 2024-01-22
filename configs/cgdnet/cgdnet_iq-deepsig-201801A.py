_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201801A.py',
    '../_base_/schedules/amc.py',
    '../_base_/runtimes/amc.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CGDNet',
        frame_length=128,
        num_classes=24,
        init_cfg=[
            dict(type='Kaiming', layer='Linear', mode='fan_in'),
            dict(type='RNN', layer='GRU', gain=1),
            # keras vs. pytorch about different init methods https://zhuanlan.zhihu.com/p/336005430?utm_id=0
            dict(
                type='Xavier',
                layer='Conv2d',
                distribution='uniform',
                override=dict(type='Uniform', name='cnn1', a=-0.108253, b=0.108253)),
        ],
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
