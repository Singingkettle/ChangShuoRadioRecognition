"""Wave-16b: HCGDNN fine-polish FT from 63.39 at HALVED init LR (seed=10).

The 1.32e-4 plateau-FT ratchet saturated at 63.35-63.39 across 4 seeds; try
finer steps (6.6e-5) from the same warm start to squeeze the last ~1.5pp
(paper 64.9).
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_ft_from_exact800.py']

load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'author_plateau_ft_seed1_w13/best_accuracy_top1_epoch_41.pth')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=6.6e-5))

randomness = dict(seed=10)
