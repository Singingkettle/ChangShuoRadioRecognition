_base_ = [
    '../_base_/default_runtime.py',
    './data_iq-ap-channel-deepsig-201610A.py'
]

in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='DNN',
    method_name='Fast MLDNN',
    backbone=dict(
        type='FMLNet',
        in_features=4,
        channel_mode=True,
        skip_connection=True
    ),
    classifier_head=dict(
        type='FAMCAUXHead',
        in_features=100,
        out_features=256,
        num_classes=11,
        batch_size=640,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1,
            alpha=0.5
        ),
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=256,
            batch_size=640,
            num_classes=11,
            mm='inner_product',
            is_abs=False,
            loss_aux=dict(
                type='LogisticLoss',
                loss_weight=0.004,
                temperature=800,
            )
        )
    )
)

train_cfg = dict()
test_cfg = dict(vis_fea=True)

total_epochs = 600
# optimizer
optimizer = dict(type='Adam', lr=0.00069)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[300, 500])

# for flops calculation
input_shape = [(2, 1, 128), (2, 1, 128), (1, 128, 128)]
