_base_ = [
    '../_base_/default_runtime.py',
    './schedule.py',
    './data_iq-ap-deepsig-201610A.py'
]

model = dict(
    type='DNN',
    method_name='MLDNN-V9',
    backbone=dict(
        type='MLNet',
        dropout_rate=0.5,
        use_GRU=True,
        is_BIGRU=True,
        fusion_method='add',
        gradient_truncation=True,
    ),
    classifier_head=dict(
        type='MLDNNHead',
        heads=[
            # Snr Head
            dict(
                type='AMCHead',
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
                type='AMCHead',
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
                type='AMCHead',
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
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
        ]
    ),
)
