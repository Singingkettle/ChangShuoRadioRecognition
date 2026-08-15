# Multitask-learning DNN (own method) on I/Q + A/P; paper-native 50/10/40.
# Paper: "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification", IEEE TVT (2021).
_base_ = [
    './iq-ap-deepsig201610A.py',
    './schedules.py',
    '../_base_/runtimes/amc.py'
]

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MLDNN',
        dropout_rate=0.5,
        use_GRU=True,
        is_BIGRU=True,
        fusion_method='safn',
        gradient_truncation=True,
        num_classes=11,
        init_cfg=dict(type='Xavier', layer='Conv2d')
    ),
    head=dict(
        type='MLDNNHead',
        loss_amc_merge=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        ),
        loss_amc_ap=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        ),
        loss_amc_iq=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        ),
        loss_snr=dict(
            type='CrossEntropyLoss',
            loss_weight=1
        ),
    ),
)
