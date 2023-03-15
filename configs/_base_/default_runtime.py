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
