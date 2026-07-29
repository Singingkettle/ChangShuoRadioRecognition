"""Wave-82 auto: HCGDNN FT 80ep from best."""
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
load_from = 'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/paper_multistep_l2_esoff800/best_accuracy_top1_epoch_571.pth'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=80, eta_min=1e-6)
