"""Wave-4 marginal retune: LSTM2 @ RML2016.10A — longer ES patience (peak fail)."""

_base_ = ['../../lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=35, rule='greater'),
]
