_base_ = [
    '../_base_/default_runtime.py',
    './schedule.py',
    './data_iq-ap-deepsig201610A.py'
]

model = dict(
    type='SingleHeadClassifier',

    backbone=dict(
        type='MLNet',
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
                type='ACMHead',
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
                type='ACMHead',
                num_classes=11,
                in_features=100,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # High Head
            dict(
                type='ACMHead',
                num_classes=11,
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
                    type='NLLLoss',
                    loss_weight=1,
                ),
            ),
        ]
    ),
)

# for flops calculation
input_shape = [(1, 2, 128), (1, 2, 128), (1, 128, 128)]
