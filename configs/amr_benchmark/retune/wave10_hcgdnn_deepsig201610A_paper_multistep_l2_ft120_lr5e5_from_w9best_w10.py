"""Wave-10 Tier-A: HCGDNN gentle FT 120ep @ lr=5e-5 from W9 best 63.30 @ep968. Arch freeze."""
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'paper_multistep_exact800_esoff1600/best_accuracy_top1_epoch_968.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(
    _delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
