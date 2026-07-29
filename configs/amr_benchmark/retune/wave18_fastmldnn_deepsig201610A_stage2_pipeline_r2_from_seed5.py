"""Wave-18: round-2 FT from FastMLDNN seed5 NEW BEST 61.301."""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']
load_from = (
    'work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/'
    'stage2_pipeline_seed5_w16/best_accuracy_top1_epoch_144.pth')
randomness = dict(seed=12)
