"""Wave-30: cldnnw@10A polish — FT amrb 55.26 ckpt at LR 1e-4 (pass 55.5)."""
_base_ = ['./wave15_cldnnw_deepsig201610A_amrb_plateau.py']
load_from = (
    'work_dirs/amr_benchmark_retune/cldnnw/deepsig201610A/'
    'amrb_seed1_w29/best_accuracy_top1_epoch_76.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))
randomness = dict(seed=0)
