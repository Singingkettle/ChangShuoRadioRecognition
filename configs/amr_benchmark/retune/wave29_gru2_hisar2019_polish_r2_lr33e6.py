"""Wave-29: GRU2@Hisar polish r2 — FT the 68.59 ckpt at lower LR."""
_base_ = ['./wave28_gru2_hisar2019_polish_lr1e4.py']
load_from = (
    'work_dirs/amr_benchmark_retune/gru2/hisar2019/'
    'polish_lr1e4_w28/best_accuracy_top1_epoch_2.pth')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=3.3e-5))
randomness = dict(seed=1)
