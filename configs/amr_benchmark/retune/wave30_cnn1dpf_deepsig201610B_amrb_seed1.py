"""Wave-30: cnn1dpf@10B amrb (best 58.88, pass 60.5) — no augment (augment collapsed)."""
_base_ = ['./wave15_cnn1dpf_deepsig201610B_amrb_plateau.py']
randomness = dict(seed=1)
