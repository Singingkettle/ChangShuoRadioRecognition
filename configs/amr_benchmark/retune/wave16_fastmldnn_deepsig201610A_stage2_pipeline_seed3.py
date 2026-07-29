"""Wave-16: seed sweep of the TRUE two-stage pipeline stage-2 (seed=3).

The steward-chained author_stage2_from_stage1_w12 hit 61.29 (NEW BEST, prev
61.05; paper 63.24). NOTE: wave-14's hand-written stage2-from-stage1 configs
scored only ~53.9 because they inherited the esoff300 lineage's per-sample L2
pipeline — mismatched with the no-L2 stage-1 backbone. This config inherits
the stage-1 config directly (pipeline-matched), like the 61.29 run.
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_stage1.py']

randomness = dict(seed=3)
