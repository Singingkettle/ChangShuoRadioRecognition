"""Auto marginal retune: denscnn @ deepsig201610A — relaxed early stopping."""

_base_ = ['../../denscnn/denscnn_iq-deepsig-201610A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
