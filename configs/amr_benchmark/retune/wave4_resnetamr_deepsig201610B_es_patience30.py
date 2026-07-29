"""Wave-4 marginal retune: ResNetAMR @ RML2016.10B — relaxed early stopping."""

_base_ = ['../../resnetamr/resnetamr_iq-deepsig-201610B.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
