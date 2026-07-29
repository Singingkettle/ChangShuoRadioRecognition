"""Wave-2 retune: FastMLDNN @ RML2016.10A — P0 + paper channel-mode dropout.

Same as beta05_xavier_esoff150 but backbone dp=0.07 (paper channel pretrain
value; default AMR uses dp=0.5).
"""

_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

model = dict(
    backbone=dict(
        dp=0.07,
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
    head=dict(beta=0.5),
)

custom_hooks = []
