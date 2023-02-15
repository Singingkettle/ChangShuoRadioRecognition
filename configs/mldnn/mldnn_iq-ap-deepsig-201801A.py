_base_ = [
    '../_base_/default_runtime.py',
    './schedule.py',
    './data_iq-ap-deepsig-201801A.py'
]

model = dict(
    type='DNN',
    method_name='MLDNN',
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
        heads=[
            # Snr Head
            dict(
                type='ClassificationHead',
                num_classes=2,
                in_features=100,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Low Head
            dict(
                type='ClassificationHead',
                num_classes=24,
                in_features=100,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # High Head
            dict(
                type='ClassificationHead',
                num_classes=24,
                in_features=100,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # Merge Head
            dict(
                type='MergeAMCHead',
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
        ]
    ),
)
