"""Wave-10 Tier-A: FastMLDNN gentle FT 120ep @ lr=5e-5 from W9 FT80 best (val 61.26 / test 60.78). Arch freeze."""
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'paper_fixedlr_l2_ft80_from_w8best/best_accuracy_top1_epoch_37.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(
    _delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
