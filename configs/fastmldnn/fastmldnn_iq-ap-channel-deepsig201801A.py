_base_ = [
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig201801A.py'
]

in_size = 100
out_size = 256
num_classes = 24
model = dict(
    type='FastMLDNN',
    backbone=dict(
        type='FMLNet',
        avg_pool=(1, 8),
    ),
    classifier_head=dict(
        type='FastMLDNNHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=num_classes,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5
        ),
        shrinkage_head=dict(
            type='ShrinkageHead',
            num_classes=num_classes,
            mm='cosine',
            loss_shrinkage=dict(
                type='ShrinkageLoss',
                loss_weight=0.4,
                temperature=10,
            )
        )
    )
)

total_epochs = 800
# optimizer
optimizer = dict(type='AdamW', lr=0.00015)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
# learning policy
lr_config = dict(policy='step', gamma=0.3, step=[700, 1200])
