_base_ = [
    '../_base_/default_runtime.py',
    './schedule.py',
    './data_iq-ap-deepsig201801A.py'
]

model = dict(
    type='MLDNN',
    backbone=dict(
        type='MLNet',
        avg_pool=(1, 8),
        dropout_rate=0.5,
        use_GRU=True,
        is_BIGRU=True,
        fusion_method='safn',
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
                init_cfg=[
                    dict(type='Kaiming', layer='Linear', mode='fan_in', nonlinearity='relu', bias=0., distribution='normal'),
                ],
            ),
            # Low Head
            low=dict(
                type='ClassificationHead',
                num_classes=24,
                in_size=100,
                out_size=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
                init_cfg=[
                    dict(type='Kaiming', layer='Linear', mode='fan_in', nonlinearity='relu', bias=0., distribution='normal'),
                ],
            ),
            # High Head
            high=dict(
                type='ClassificationHead',
                num_classes=24,
                in_size=100,
                out_size=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
                init_cfg=[
                    dict(type='Kaiming', layer='Linear', mode='fan_in', nonlinearity='relu', bias=0., distribution='normal'),
                ],
            ),
            # Merge Head
            merge=dict(
                type='MergeAMCHead',
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
        ),
    ),
)
optimizer = dict(type='Adam', lr=0.0001)