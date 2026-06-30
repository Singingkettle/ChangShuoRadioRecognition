# runtime settings
default_scope = 'csrr'

default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy/top1', rule='greater', max_keep_ckpts=1),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Early stopping on the fused validation top-1 accuracy (0-100 scale). The old
# ``min_delta=0, patience=100`` let any sub-0.01pp wiggle reset the window, so
# early stopping effectively never fired and training drifted to epoch 200+.
# Requiring a >=0.1pp gain over a 15-epoch window makes it fire once accuracy
# genuinely plateaus (matches the baseline recipe in _base_/runtimes/amc.py).
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0.1, patience=15, rule='greater'),
    dict(type='HCGDNNHook',)
]

env_cfg = dict(
    # disable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume the training of the checkpoint
resume_from = None

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=0, deterministic=False)

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])
