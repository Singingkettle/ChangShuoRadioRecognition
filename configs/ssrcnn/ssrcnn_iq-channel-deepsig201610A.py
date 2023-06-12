_base_ = [
    './data_iq-channel-deepsig201610A.py',
]

in_size = 448
out_size = 100
# Model
model = dict(
    type='SSRCNN',
    backbone=dict(
        type='SSRNet',
    ),
    classifier_head=dict(
        type='SSRCNNHead',
        in_size=in_size,
        out_size=out_size,
        num_classes=11,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
        center_head=dict(
            type='CenterHead',
            in_size=out_size,  # keep the same as snr head
            num_classes=11,
            loss_center=dict(
                type='CenterLoss',
                loss_weight=0.003,
            ),
        )
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=350)
# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', add_graph=False)
    ])
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = None
auto_resume = None
workflow = [('train', 1)]
evaluation = dict(interval=1)
dropout_alive = False