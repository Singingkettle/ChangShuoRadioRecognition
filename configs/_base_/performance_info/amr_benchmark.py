# Canonical comparison registry for the AMR-Benchmark reproduction: keeps only
# the 15 baseline methods (no MLDNN/HCGDNN/FastMLDNN/DSCLDNN/MCformer/TRN).
#
# Path convention (IMPORTANT): the Phase 2 orchestrator
# (``tools/amr_benchmark/run_migration.py``) writes every run to the NESTED
# layout ``work_dirs/amr_benchmark/<model>/<dataset>/res/paper.pkl`` (model =
# the matrix key / backbone dir, dataset = short label). The performance
# framework builds the pickle path as ``<work_dir>/<publish_subdir>/res/
# paper.pkl``, so we set ``work_dir='work_dirs/amr_benchmark'`` and make every
# publish entry the ``<model>/<dataset>`` subdir. (The previous flat
# ``<config_name>`` convention resolved to non-existent
# ``work_dirs/<config_name>/res/paper.pkl`` and silently produced empty
# figures.)

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
