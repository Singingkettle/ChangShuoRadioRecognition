_base_ = [
    './iq-ap-deepsig201604C.py',
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
