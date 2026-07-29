"""Wave-82 auto Tier-B: CNN1DPF @ 10B lr5e-5."""
_base_ = ['./wave4_cnn1dpf_deepsig201610B_lr2e4_warmup.py']
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
