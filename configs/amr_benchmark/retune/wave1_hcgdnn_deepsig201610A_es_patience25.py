"""Wave-1 retune: HCGDNN @ RML2016.10A — relaxed early stopping."""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=25, rule='greater'),
]
