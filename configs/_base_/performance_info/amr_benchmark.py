# Optional 15-method comparison subset (no MLDNN/HCGDNN/FastMLDNN/DSCLDNN/
# MCformer/TRN). Same nested ``<model>/<dataset>/res/paper.pkl`` layout as
# ``amc.py``; point ``work_dir`` at the tree you collected.

# Display name -> orchestrator model key (== matrix key == work_dir subdir).
_MODEL_KEYS = dict(
    CNN1='cnn2',
    CNN2='cnn4',
    MCNET='mcnet',
    ICAMCNet='icamcnet',
    ResNet='resnetamr',
    DenseNet='denscnn',
    GRU='gru2',
    LSTM='lstm2',
    DAE='dae',
    MCLDNN='mcldnn',
    CLDNNW='cldnnw',
    CLDNN2='cldnnl',
    CGDNet='cgdnet',
    PETCGDNN='petcgdnn',
    CNN1DPF='cnn1dpf',
)

_DATASETS = ['deepsig201610A', 'deepsig201610B', 'deepsig201801A', 'hisar2019']

info = dict(
    work_dir='work_dirs/amr_benchmark',
    save_dir='work_dirs/performance/amr_benchmark',
    methods={name: idx for idx, name in enumerate(_MODEL_KEYS)},
    publish={
        ds: {disp: f'{key}/{ds}' for disp, key in _MODEL_KEYS.items()}
        for ds in _DATASETS
    },
)

del _MODEL_KEYS, _DATASETS
