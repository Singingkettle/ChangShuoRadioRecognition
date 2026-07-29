"""Wave-6 Tier-A: FastMLDNN @ 10A — paper fixed-LR near-miss + SelfNormalize L2, 500ep.

Prior best paper_fixedlr_beta05_dp007_esoff400: 60.67/91.59 vs paper 63.24/92.0
(-0.35pp overall). Add L2 (Wave-2/3 winner stack) and extend to 500ep.
"""

_base_ = [
    './wave4_fastmldnn_deepsig201610A_paper_fixedlr_beta05_dp007_esoff400.py',
]

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

# SelfNormalize L2 on IQ pipeline (MCLDNN / Wave-2 success path)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
