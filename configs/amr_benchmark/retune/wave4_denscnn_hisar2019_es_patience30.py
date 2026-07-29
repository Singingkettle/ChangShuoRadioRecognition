"""Auto marginal retune: denscnn @ hisar2019 — relaxed early stopping."""

_base_ = ['../../denscnn/denscnn_iq-hisar-2019.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
