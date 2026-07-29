"""Wave-1 retune: CGDNet @ RML2018.01A — relaxed early stopping."""

_base_ = ['../../cgdnet/cgdnet_iq-deepsig-201801A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
