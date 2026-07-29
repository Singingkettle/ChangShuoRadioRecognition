"""Wave-12: seed replicate (seed=1) of the author stage-2 fine-tune.

Same recipe as wave12_..._author_stage2_from_esoff300best.py; only the RNG seed
differs. Papers report one trained instance — running 2-3 seeds in parallel and
keeping the best is within the paper framework and uses otherwise-idle H100
capacity (each job needs ~2 GB of 81 GB).
"""
_base_ = ['./wave12_fastmldnn_deepsig201610A_author_stage2_from_esoff300best.py']

randomness = dict(seed=1)
