"""Wave-1 retune: CLDNNW @ RML2018.01A — Xavier + relaxed early stopping."""

_base_ = ['../../cldnnw/cldnnw_iq-deepsig-201801A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
        ],
    ),
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]
