"""Wave-16b: pipeline stage-2 extra seed (seed=6)."""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']

randomness = dict(seed=6)
