# Extended AMC fine-tune on detector-proposal crops (20 epochs).
#
# Continues from the 5-epoch proposal-crop run (best epoch 2) with a full
# cosine schedule over 20 epochs at the same domain-adaptation LR.
_base_ = 'jdm-amc_iq-csrd_detprops_5ep.py'

load_from = 'work_dirs/jdm/exp_amc_detprops_5ep/best_accuracy_top1_epoch_2.pth'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=20,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
work_dir = 'work_dirs/jdm/exp_amc_detprops_20ep'
