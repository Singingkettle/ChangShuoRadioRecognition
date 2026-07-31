"""Wave-28: FastMLDNN polish ratchet r2 — FT the 61.309 ckpt at quarter LR."""
_base_ = ['./wave27_fastmldnn_polish_lr5e5_seed0.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'polish_lr5e5_seed0_w27/best_accuracy_top1_epoch_16.pth')
optim_wrapper = dict(optimizer=dict(lr=2.6e-5))
randomness = dict(seed=30)
