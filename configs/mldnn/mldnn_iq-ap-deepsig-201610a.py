# MLDNN calibration on the RadioML.2016.10A 50%/10% development split.
# Paper: "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification", IEEE Internet of Things Journal (2021).

data_root = 'data/ModulationClassification/DeepSig/RadioML.2016.10A'
dataset_type = 'AMCDataset'
sample_length = 128

train_pipeline = [
    dict(type='MLDNNSNRLabel'),
    dict(type='MLDNNIQToAP', phase_order='real_over_imag'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, sample_length])),
    dict(type='Reshape', shapes=dict(ap=[1, 2, sample_length])),
    dict(
        type='PackMultiTaskInputs',
        multi_task_fields=['gt_label'],
        input_key=['iq', 'ap'],
        task_handlers=dict(
            amc=dict(type='PackInputs'),
            snr=dict(type='PackInputs'))),
]

test_pipeline = [
    dict(type='MLDNNIQToAP', phase_order='real_over_imag'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, sample_length])),
    dict(type='Reshape', shapes=dict(ap=[1, 2, sample_length])),
    dict(type='PackInputs', input_key=['iq', 'ap']),
]

train_dataloader = dict(
    batch_size=640,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        pipeline=train_pipeline,
        cache=True,
        cache_file='auto',
        test_mode=False))
val_dataloader = dict(
    batch_size=640,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='validation.json',
        pipeline=test_pipeline,
        cache=True,
        cache_file='auto',
        test_mode=True))
test_dataloader = None

val_evaluator = [
    dict(type='StreamingAccuracy', expected_samples=22000,
         expected_world_size=1),
    dict(type='StreamingLoss', task='classification',
         expected_samples=22000, expected_world_size=1),
]
test_evaluator = dict(type='Accuracy', topk=(1,))

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MLDNN',
        dropout_rate=0.5,
        use_GRU=True,
        is_BIGRU=True,
        fusion_method='safn',
        gradient_truncation=True,
        merge_log_probability=True,
        num_classes=11,
        init_cfg=dict(type='Xavier', layer='Conv2d')),
    head=dict(
        type='MLDNNHead',
        loss_amc_merge=dict(type='CrossEntropyLoss', loss_weight=1),
        loss_amc_ap=dict(type='CrossEntropyLoss', loss_weight=1),
        loss_amc_iq=dict(type='CrossEntropyLoss', loss_weight=1),
        loss_snr=dict(type='CrossEntropyLoss', loss_weight=1)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=4e-4, weight_decay=1e-5),
    clip_grad=dict(max_norm=5.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0,
         end=5, convert_to_iter_based=True),
    dict(type='ConstantLR', factor=1, by_epoch=True, begin=5, end=400),
]
train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)
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
custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage',
         momentum=1e-4, update_buffers=True, priority=49),
]
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
