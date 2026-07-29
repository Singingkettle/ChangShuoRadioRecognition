"""Wave-82 auto: FastMLDNN FT 80ep from best ckpt (no arch change)."""
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
load_from = 'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/paper_fixedlr_l2_esoff600/best_accuracy_top1_epoch_565.pth'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=80, eta_min=1e-6)
