"""Wave-14: ROUND-2 plateau FT from the wave-13 NEW BEST (63.39), seed=6.

The FT-from-best ratchet has worked twice (63.31 -> 63.314 -> 63.39).
Same author plateau recipe, warm start moved up to the newest best.
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_ft_from_exact800.py']

load_from = (
    'work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/'
    'author_plateau_ft_seed1_w13/best_accuracy_top1_epoch_41.pth')

randomness = dict(seed=6)
