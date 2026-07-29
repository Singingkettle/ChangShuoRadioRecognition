# Wave 3 Track B — fresh 5-ep train, lr 1e-3 (winning recipe), ES off.
_base_ = '../jdm-det_fft-csrd_anchor096146_bw20_5ep.py'

custom_hooks = []

work_dir = 'work_dirs/jdm/retune/det_wave3b_5ep_lr1e3'
