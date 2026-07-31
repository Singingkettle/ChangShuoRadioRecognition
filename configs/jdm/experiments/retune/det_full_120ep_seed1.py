"""Seed re-roll of the champion 120-epoch detector recipe.

Every schedule/regularization variant (60/90/200ep, bw40, EMA, SWA) lost to
det120; the remaining narrative-safe lever on the high-IoU box-tightness gap
is seed variance of the same recipe.
"""
_base_ = './det_full_120ep_lr1e3.py'
randomness = dict(seed=1)
work_dir = 'work_dirs/jdm/retune/det_full_120ep_seed1'
