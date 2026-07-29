"""Auto marginal retune: gru2 @ hisar2019 — relaxed early stopping."""

_base_ = ['../../gru2/gru2_iq-shape-L-F-hisar-2019.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
