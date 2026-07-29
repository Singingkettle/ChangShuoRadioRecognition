"""Wave-1 retune: LSTM2 @ RML2016.10A — relaxed early stopping (peak-only fail)."""

_base_ = ['../../lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
