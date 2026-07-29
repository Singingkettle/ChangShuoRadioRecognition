"""Wave-14 (local box): ROUND-2 plateau FT from the wave-13 NEW BEST (63.39), seed=8.

Complements seeds 5/6/7 on the H100.
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_ft_from_exact800.py']

load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'author_plateau_ft_seed1_w13/best_accuracy_top1_epoch_41.pth')

randomness = dict(seed=8)
