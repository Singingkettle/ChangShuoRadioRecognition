"""Wave-13 (local box): seed sweep of the HCGDNN author plateau FT (seed=3).

Complements seeds 1,2 running on the H100. Recipe: plateau FT from exact800
best (wave-12 tied the historical best 63.314; paper 64.9).
"""
_base_ = ['./wave12_hcgdnn_deepsig201610A_author_plateau_ft_from_exact800.py']

randomness = dict(seed=3)
