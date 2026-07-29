"""Wave-9 Tier-A: FastMLDNN — paper fixed-LR WITHOUT IQ L2, 700ep (paper-path contrast).

Prior L2 stack plateaued ~60.89; paper alignment flags L2 as non-paper. Try longer
no-L2 fixed-LR (W4 no-L2 fixedlr was 400ep only).
"""

_base_ = ['./wave4_fastmldnn_deepsig201610A_paper_fixedlr_beta05_dp007_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='ConstantLR',
    factor=1.0,
)
