_base_ = [
    './data_csrr2023.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='BaseDetector',
    method_name='RRDNN',
    backbone=dict(
        type='DetCNN',
    ),
    detector_head=dict(
        type='SignalDetectionHead',
        cfg=dict(
            nms_pre=1000,
            score_thr=0.05,
            nms=dict(iou_threshold=0.45),
            max_per_sequence=27,
        )
    ),
)

is_det = True
runner = dict(type='IterBasedRunner', max_iters=200000)
# Optimizer
# for flops calculation
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed',)
evaluation = dict(interval=100)
checkpoint_config = dict(interval=100)
