"""Auto marginal retune: mcldnn @ hisar2019 — relaxed early stopping."""

_base_ = ['../../mcldnn/mcldnn_iq-hisar-2019.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
