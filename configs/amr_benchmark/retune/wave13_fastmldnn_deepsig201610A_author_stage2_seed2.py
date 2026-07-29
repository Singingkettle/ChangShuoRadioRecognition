"""Wave-13: seed sweep of the author stage-2 fine-tune (seed=2).

wave-12 found seed variance moves this recipe past the historical best
(seed0 61.01, seed1 61.05 vs prev 61.02; paper 63.24). Same recipe, new seed.
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_esoff300best.py']

randomness = dict(seed=2)
