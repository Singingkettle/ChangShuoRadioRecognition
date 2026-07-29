"""Wave-1 retune: CLDNNW @ RML2018.01A — Xavier init only."""

_base_ = ['../../cldnnw/cldnnw_iq-deepsig-201801A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform'),
        ],
    ),
)
