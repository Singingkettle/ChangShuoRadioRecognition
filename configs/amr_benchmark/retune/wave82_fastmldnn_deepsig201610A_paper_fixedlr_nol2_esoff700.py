"""Wave-82 auto: FastMLDNN paper fixed-LR no-L2 700ep."""
_base_ = ['./wave4_fastmldnn_deepsig201610A_paper_fixedlr_beta05_dp007_esoff400.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=1)
param_scheduler = dict(_delete_=True, type='ConstantLR', factor=1.0)
