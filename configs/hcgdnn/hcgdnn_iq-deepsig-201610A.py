# Hierarchical classification-head gated DNN (own method) on I/Q.
# Paper: "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification", IEEE WCL (2022).
_base_ = [
    './iq-deepsig201610A.py',
    './runtimes.py',
    './schedules.py'
]

# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='HCGDNN',
        num_classes=11,
    ),
    head=dict(
        type='HCGDNNHead',
        loss=dict(
            cnn=dict(type='CrossEntropyLoss', loss_weight=1),
            gru1=dict(type='CrossEntropyLoss', loss_weight=1),
            gru2=dict(type='CrossEntropyLoss', loss_weight=1)
        )
    )
)
