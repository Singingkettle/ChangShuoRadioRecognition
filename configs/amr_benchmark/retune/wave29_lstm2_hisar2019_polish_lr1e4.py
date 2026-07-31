"""Wave-29: LSTM2@Hisar polish — FT seed0 champion (peak 96.92, pass 97.0)."""
_base_ = ['./wave25_lstm2_hisar2019_plateau_peak.py']
load_from = (
    'work_dirs/amr_benchmark_retune/lstm2/hisar2019/'
    'plateau_peak_w25/best_accuracy_top1_epoch_144.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1',
                     min_delta=0, patience=25, rule='greater')]
randomness = dict(seed=0)
