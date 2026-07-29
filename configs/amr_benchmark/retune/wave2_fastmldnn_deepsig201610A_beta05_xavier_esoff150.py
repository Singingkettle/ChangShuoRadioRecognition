"""Wave-2 retune: FastMLDNN @ RML2016.10A — P0 gap fixes (fc5c869c).

Restore paper multi-loss (beta=0.5), Xavier/TruncNormal init on backbone,
disable early stopping (full 150-epoch cosine). Inherits 10A base lr=4.4e-4,
batch=640.
"""

_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

model = dict(
    backbone=dict(
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
    head=dict(beta=0.5),
)

custom_hooks = []
