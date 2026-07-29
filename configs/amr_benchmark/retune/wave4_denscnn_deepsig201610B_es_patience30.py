"""Auto marginal retune: denscnn @ deepsig201610B — relaxed early stopping."""

_base_ = ['../../denscnn/denscnn_iq-deepsig-201610B.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
