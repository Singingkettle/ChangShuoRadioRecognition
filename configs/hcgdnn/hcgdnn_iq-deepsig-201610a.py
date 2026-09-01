# HCGDNN calibration on the RadioML.2016.10A 50%/10% development split.
# Paper: "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification", IEEE Transactions on Wireless Communications (2022).

data_root = 'data/ModulationClassification/DeepSig/RadioML.2016.10A'
dataset_type = 'AMCDataset'

pipeline = [
    dict(type='Reshape', shapes=dict(iq=[2, 1, 128])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(
    batch_size=640,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        pipeline=pipeline,
        cache=True,
        test_mode=False))
val_dataloader = dict(
    batch_size=640,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='validation.json',
        pipeline=pipeline,
        cache=True,
        test_mode=True))
test_dataloader = None

val_evaluator = dict(
    type='HCGDNNWeightsAccuracy',
    topk=(1,),
    optimization_temperature=745.0,
    optimization_disagreement_only=False)
test_evaluator = dict(type='Accuracy', topk=(1,))

model = dict(
    type='SignalClassifier',
    backbone=dict(type='HCGDNN', num_classes=11),
    head=dict(
        type='HCGDNNHead',
        loss=dict(
            cnn=dict(type='CrossEntropyLoss', loss_weight=1),
            gru1=dict(type='CrossEntropyLoss', loss_weight=1),
            gru2=dict(type='CrossEntropyLoss', loss_weight=1))))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=4.4e-4, weight_decay=1e-5),
    clip_grad=dict(max_norm=5.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0,
         end=5, convert_to_iter_based=True),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=1600,
         milestones=[800], gamma=0.3),
]
train_cfg = dict(by_epoch=True, max_epochs=1600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_scope = 'csrr'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='accuracy/top1', rule='greater',
                    max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))
custom_hooks = [dict(type='HCGDNNHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=31, deterministic=False)
visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='TensorboardVisBackend')])
