"""Wave-12 Tier-A: FastMLDNN author-exact STAGE-1 (upstream published recipe).

Recovered from origin/main (the paper's official release; abstract links this
repo). Author stage-1: beta=0, dp=0.5 default, Adam 4.4e-4, MultiStepLR
[800,1200] gamma=0.3, 3200 epochs, NO early stopping, batch 640, no IQ L2.
The author's own best-val landed at epoch 648. save_best keeps the best-val
checkpoint for the stage-2 fine-tune (see wave12 ..._author_stage2_* configs).
"""
_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[800, 1200],
    gamma=0.3,
)

train_cfg = dict(by_epoch=True, max_epochs=3200, val_interval=1)

# Author's published runtime has no EarlyStoppingHook.
custom_hooks = []
