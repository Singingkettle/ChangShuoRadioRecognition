"""Wave-27: HCGDNN polish ratchet r2 — FT the 63.393 ckpt at quarter LR."""
_base_ = ['./wave16_hcgdnn_deepsig201610A_polish_lr66e6_seed10.py']
load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'polish_lr66e6_seed10_w16/best_accuracy_top1_epoch_116.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3.3e-5))
randomness = dict(seed=20)
