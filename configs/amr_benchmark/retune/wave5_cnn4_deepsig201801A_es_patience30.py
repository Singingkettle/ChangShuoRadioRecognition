"""Auto marginal retune: cnn4 @ deepsig201801A — relaxed early stopping."""

_base_ = ['../../cnn4/cnn4_iq-deepsig-201801A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
