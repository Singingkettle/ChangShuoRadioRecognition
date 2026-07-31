"""Wave-28: ICAMCNET@Hisar polish ratchet — FT the 98.58-peak ckpt at low LR.

Peak stuck 98.52-98.58 across three runs (pass 99.0). Apply the ratchet that
carried HCGDNN over its line: warm start from the best ckpt, LR 3.3e-5.
"""
_base_ = ['./wave21_icamcnet_hisar2019_plateau_peak.py']
load_from = (
    'work_dirs/amr_benchmark_retune/icamcnet/hisar2019/'
    'plateau_peak_w21/best_accuracy_top1_epoch_185.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3.3e-5))
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=25, rule='greater')]
randomness = dict(seed=0)
