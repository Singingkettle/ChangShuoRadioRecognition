"""Wave-12 Tier-A: HCGDNN author-plateau fine-tune from exact800 best (63.30).

Same author-exact plateau/ES recipe as wave12 author_plateau_es100, but warm
started from the best checkpoint of paper_multistep_exact800_esoff1600
(val best at epoch 968, test 63.30) at one gamma-step-down LR (0.3 x 4.4e-4),
letting ReduceOnPlateau adaptively anneal from there.
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_es100.py']

load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'paper_multistep_exact800_esoff1600/best_accuracy_top1_epoch_968.pth')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1.32e-4))

train_cfg = dict(by_epoch=True, max_epochs=1200, val_interval=1)
