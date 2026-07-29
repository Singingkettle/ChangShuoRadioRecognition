"""Wave-13: seed sweep of the HCGDNN author plateau FT (seed=2).

wave-12 plateau FT tied the historical best (63.314 vs 63.31; paper 64.9).
Same recipe, new seed.
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_ft_from_exact800.py']

randomness = dict(seed=2)
