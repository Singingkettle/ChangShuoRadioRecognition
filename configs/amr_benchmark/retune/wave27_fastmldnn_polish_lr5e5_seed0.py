"""Wave-27: FastMLDNN polish ratchet — FT the 61.286 stage-2 best at half LR.

Stage-2 seed lottery saturated at 61.0-61.29 (pass line 61.74). Apply the
HCGDNN-style fine-polish: warm start from the champion, halve the author
constant LR (1.054e-4 -> 5.27e-5), same dp=0.07 beta=0.5 pipeline.
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'author_stage2_from_stage1_w12/best_accuracy_top1_epoch_148.pth')
optim_wrapper = dict(optimizer=dict(lr=5.27e-5))
randomness = dict(seed=0)
