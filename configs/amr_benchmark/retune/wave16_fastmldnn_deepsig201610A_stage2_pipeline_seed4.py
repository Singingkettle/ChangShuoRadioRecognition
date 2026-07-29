"""Wave-16 (local box): pipeline-matched stage-2 seed=4 (complements H100 seeds 1-3)."""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']

randomness = dict(seed=4)
