"""Wave-16: ROUND-2 of the pipeline stage-2 — FT from the 61.29 new best.

Same recipe (dp0.07 beta0.5 const 1.054e-4, no-L2 pipeline), warm start moved
up to author_stage2_from_stage1_w12's best checkpoint.
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']

load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'author_stage2_from_stage1_w12/best_accuracy_top1_epoch_148.pth')

randomness = dict(seed=9)
