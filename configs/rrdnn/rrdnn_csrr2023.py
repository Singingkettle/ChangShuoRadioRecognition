_base_ = [
    './data_csrr2023.py',
    '../_base_/schedules/schedule.py',
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
            max_per_sequence=9,
        )
    ),
)

is_det = True
runner = dict(type='EpochBasedRunner', max_epochs=600)
# Optimizer
# for flops calculation
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='fixed')
evaluation = dict(interval=10)
checkpoint_config = dict(interval=10)
