_base_ = [
    './schedule.py',
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig201610A.py'
]

in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='SingleHeadClassifier',
    method_name='Fast MLDNN-V6',
    backbone=dict(
        type='FMLNet',
        in_features=4,
        channel_mode=True,
        skip_connection=True,
    ),
    classifier_head=dict(
        type='FAMCAUXHead',
        in_features=in_features,
        out_features=out_features,
        num_classes=num_classes,
        batch_size=640,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
        # Intra Head
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=out_features,
            batch_size=640,
            num_classes=num_classes,
            mm='inner_product',

            loss_aux=dict(
                type='LogisticLoss',
                loss_weight=0.004,
                temperature=800,
            ),
        ),
    ),
)

# for flops calculation
input_shape = [(2, 1, 128), (2, 1, 128), (1, 128, 128)]
