_base_ = [
    '../_base_/default_runtime.py',
    './schedule.py',
    './data_iq-ap-deepsig201610A.py'
]

model = dict(
    type='MLDNN',
    backbone=dict(
        type='MLNet',
        dropout_rate=0.5,
        use_GRU=True,
        is_BIGRU=True,
        fusion_method='attention',
        gradient_truncation=True,
    ),
    classifier_head=dict(
        type='MLDNNHead',
        heads=dict(
            # Snr Head
            snr=dict(
                type='ClassificationHead',
                num_classes=2,
                in_size=100,
                out_size=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Low Head
            low=dict(
                type='ClassificationHead',
                num_classes=11,
                in_size=100,
                out_size=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # High Head
            high=dict(
                type='ClassificationHead',
                num_classes=11,
                in_size=100,
                out_size=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Merge Head
            merge=dict(
                type='MergeAMCHead',
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
        )
    ),
)
