"""Wave-1 retune: HCGDNN @ RML2016.10A — paper LR with longer patience."""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='accuracy/top1',
         min_delta=0.05, patience=30, rule='greater'),
]

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
