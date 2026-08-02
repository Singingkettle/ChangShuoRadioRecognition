"""Wave-30: FastMLDNN polish r4 — FT 61.31 ckpt at LR 6.5e-6 (pass 61.74)."""
_base_ = ['./wave27_fastmldnn_polish_lr5e5_seed0.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'polish_r3_seed41_w29/best_accuracy_top1_epoch_41.pth')
optim_wrapper = dict(optimizer=dict(lr=6.5e-6))
randomness = dict(seed=50)
