"""Wave-6 Tier-B: ResNetAMR @ 10B — longer ES patience (gap 0.13pp overall)."""

_base_ = ['../../resnetamr/resnetamr_iq-deepsig-201610B.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=40, rule='greater'),
]
