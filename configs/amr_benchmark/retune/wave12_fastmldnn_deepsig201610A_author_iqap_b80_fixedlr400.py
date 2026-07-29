"""Wave-12 Tier-A: FastMLDNN author per-dataset paper config (batch 80).

From configs/fastmldnn/paper/fastmldnn_iq-ap-deepsig201610A.py: batch 80,
Adam 4.4e-4, MultiStep milestones [800,1200] with max_epochs=400 — i.e. the LR
NEVER steps, so effectively constant 4.4e-4 for 400 epochs. CE only (beta=0),
dp=0.5 default, TruncNormal Linear init, no ES, no L2. All waves 1-11 used
batch 640; batch 80 gives 8x more updates/epoch with more gradient noise
(stronger implicit regularisation) and is the paper's own setting.
Val/test loaders stay at 640 (eval batch does not affect the metric).
"""
_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

train_dataloader = dict(batch_size=80)

param_scheduler = dict(_delete_=True, type='ConstantLR', factor=1)

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)

custom_hooks = []
