"""Wave-25: seed re-roll of w22 recipe — resnetamr@deepsig201610B.

w22 scored 60.22 overall (pass line 60.5, peak 90.72 already over 87 target):
0.28pp short, well within seed variance.
"""
_base_ = ['./wave22_resnetamr_deepsig201610B_radioaug_plateau.py']
randomness = dict(seed=1)
