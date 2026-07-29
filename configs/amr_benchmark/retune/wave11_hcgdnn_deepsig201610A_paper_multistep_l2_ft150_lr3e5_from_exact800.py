"""Wave-11 Tier-A: HCGDNN FT 150ep @ lr=3e-5 from exact800 best (63.30). Arch freeze."""
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'paper_multistep_exact800_esoff1600/best_accuracy_top1_epoch_968.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
param_scheduler = dict(
    _delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=150, eta_min=1e-6)
work_dir = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'paper_multistep_l2_ft150_lr3e5_from_exact800_w11')
