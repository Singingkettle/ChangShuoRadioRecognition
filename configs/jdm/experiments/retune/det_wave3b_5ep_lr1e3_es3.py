# Wave 3 Track B — fresh 5-ep train, lr 1e-3, ES patience 3.
_base_ = '../jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='detection/mAP',
         min_delta=0.001, patience=3, rule='greater'),
]

work_dir = 'work_dirs/jdm/retune/det_wave3b_5ep_lr1e3_es3'
