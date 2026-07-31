"""Wave-26: seed re-roll of w25 augment-free recipe — lstm2@hisar2019.

w25 seed0: overall 69.69 (pass 71.5), peak 96.92 (pass 97.0, only 0.08 short).
"""
_base_ = ['./wave25_lstm2_hisar2019_plateau_peak.py']
randomness = dict(seed=1)
