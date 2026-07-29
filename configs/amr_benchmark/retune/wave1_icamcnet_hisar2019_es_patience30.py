"""Wave-1 retune: ICAMCNet @ HisarMod — relaxed early stopping (peak-only fail)."""

_base_ = ['../../icamcnet/icamcnet_iq-hisar-2019.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
