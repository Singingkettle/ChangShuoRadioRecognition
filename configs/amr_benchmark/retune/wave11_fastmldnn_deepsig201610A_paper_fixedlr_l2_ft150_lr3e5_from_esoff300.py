"""Wave-11 Tier-A: FastMLDNN FT 150ep @ lr=3e-5 from historical best esoff300 (61.02). Arch freeze."""
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'beta05_xavier_l2_dp007_esoff300/best_accuracy_top1_epoch_214.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
param_scheduler = dict(
    _delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=150, eta_min=1e-6)
work_dir = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'paper_fixedlr_l2_ft150_lr3e5_from_esoff300_w11')
