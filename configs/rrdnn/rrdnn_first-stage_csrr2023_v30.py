_base_ = [
    '../_base_/default_runtime.py'
]

dataset_type = 'CSRRDataset'
data_root = './data/ChangShuo/v30'
target_name = 'modulation'
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        # ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            # dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='LoadCSRRTrainAnnotations'),
            dict(type='Collect', keys=['inputs', 'targets', 'input_metas', 'file_name', 'image_id'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs', 'input_metas', 'file_name', 'image_id'])
        ],
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs', 'input_metas', 'file_name', 'image_id'])
        ],
        data_root=data_root,
    ),
)

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
runner = dict(type='EpochBasedRunner', max_epochs=60)
# Optimizer
# for flops calculation
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
# seed = 711981417