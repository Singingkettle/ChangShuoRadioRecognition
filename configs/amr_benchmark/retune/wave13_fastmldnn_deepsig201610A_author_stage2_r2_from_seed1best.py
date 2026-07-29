"""Wave-13: author stage-2 ROUND 2 — fine-tune again from the wave-12 new best.

Warm start from author_stage2_from_esoff300best_seed1_w12 best (test 61.05),
same author stage-2 recipe (const 1.054e-4, dp0.07, beta0.5, ES pat100).
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_esoff300best.py']

load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'author_stage2_from_esoff300best_seed1_w12/best_accuracy_top1_epoch_85.pth')

randomness = dict(seed=7)
