"""Wave-30: LSTM2@Hisar polish r2 — FT the 69.88/97.23 ckpt at LR 3.3e-5.

Peak already clears pass line (97.23 >= 97.0); overall still 69.88 vs 71.5.
"""
_base_ = ['./wave25_lstm2_hisar2019_plateau_peak.py']
load_from = (
    'work_dirs/amr_benchmark_retune/lstm2/hisar2019/'
    'polish_lr1e4_w29/best_accuracy_top1_epoch_3.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3.3e-5))
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=25, rule='greater')]
randomness = dict(seed=0)
