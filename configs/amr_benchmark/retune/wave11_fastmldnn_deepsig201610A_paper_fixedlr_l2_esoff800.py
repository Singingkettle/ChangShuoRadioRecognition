"""Wave-11 Tier-A: FastMLDNN paper fixed-LR + L2, 800ep (longer than W6 500). Arch freeze."""
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)
param_scheduler = dict(
    _delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=800, eta_min=1e-6)
work_dir = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'paper_fixedlr_l2_esoff800_w11')
