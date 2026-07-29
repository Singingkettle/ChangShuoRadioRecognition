"""Wave-4 marginal retune: ResNetAMR @ RML2018.01A — relaxed early stopping (−0.26 pp overall)."""

_base_ = ['../../resnetamr/resnetamr_iq-deepsig-201801A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
