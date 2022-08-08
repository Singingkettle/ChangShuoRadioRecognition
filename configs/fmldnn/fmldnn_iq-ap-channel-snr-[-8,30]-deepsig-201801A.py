_base_ = [
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-snr-[-8,30]-deepsig-201801A.py'
]

in_features = 100
out_features = 256
num_classes = 24
model = dict(
    type='DNN',
    method_name='Fast MLDNN',
    backbone=dict(
        type='FMLNet',
        in_features=4,
        channel_mode=True,
        skip_connection=True,
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='FMLDNNHead',
        in_features=100,
        out_features=256,
        num_classes=num_classes,
        batch_size=80,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5
        ),
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=256,
            batch_size=80,
            num_classes=num_classes,
            mm='cosine',
            is_abs=False,
            loss_aux=dict(
                type='InfoNCELoss',
                loss_weight=0.4,
                temperature=10,
            )
        )
    )
)


total_epochs = 1600
# optimizer
optimizer = dict(type='AdamW', lr=0.00015)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
# learning policy
lr_config = dict(policy='step', gamma=0.3, step=[700, 1200])

