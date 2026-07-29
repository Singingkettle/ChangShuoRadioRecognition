"""Wave-14: author-exact STAGE-2 from the wave-12 stage-1 best (epoch 758), seed=0.

Stage-1 (beta0 dp0.5 MS[800,1200] 3200ep) peaked at epoch 758 and never improved
for ~2000 epochs, so it was cut short. This is the paper's true two-stage
pipeline: stage-1 backbone -> dp=0.07 + beta=0.5 at constant lr 1.054e-4.
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_esoff300best.py']

load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'author_stage1_ms3200_w12/best_accuracy_top1_epoch_758.pth')

randomness = dict(seed=0)
