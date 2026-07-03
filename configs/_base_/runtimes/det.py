# runtime settings for signal-detection training (JDM detection module)
default_scope = 'csrr'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # track the best model by detection mAP (1-D IoU)
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='detection/mAP', rule='greater',
                    max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume_from = None
randomness = dict(seed=0, deterministic=False)

visualizer = dict(type='Visualizer',
                  vis_backends=[dict(type='TensorboardVisBackend')])
