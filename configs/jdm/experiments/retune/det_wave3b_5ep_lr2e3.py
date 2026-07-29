# Wave 3 Track B — fresh 5-ep train, lr 2e-3, ES off.
_base_ = '../jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-3),
    clip_grad=dict(max_norm=35, norm_type=2),
)

custom_hooks = []

work_dir = 'work_dirs/jdm/retune/det_wave3b_5ep_lr2e3'
