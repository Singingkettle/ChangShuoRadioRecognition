# AMC method registry: maps display names used in figures to (a) a stable
# colour/legend index and (b) the work-dir subdirectory holding that run.
#
# Path convention (IMPORTANT): the Phase 2 orchestrator
# (``tools/amr_benchmark/run_migration.py``) writes every run to the NESTED
# layout ``work_dirs/amr_benchmark/<model>/<dataset>/res/paper.pkl`` (model ==
# the matrix key / backbone dir, dataset == short label). The performance
# framework loads ``<work_dir>/<publish_subdir>/res/paper.pkl``, so we set
# ``work_dir='work_dirs/amr_benchmark'`` and each publish entry is the
# ``<model>/<dataset>`` subdir. (The previous flat ``<config_name>``
# convention resolved to non-existent ``work_dirs/<config_name>/res/
# paper.pkl`` and silently produced empty figures for every method.)
#
# Dataset keys:
#   deepsig201610A -> RML2016.10A ; deepsig201610B -> RML2016.10B
#   deepsig201801A -> RML2018.01A ; hisar2019       -> HisarMod2019.1

# Display name -> orchestrator model key (== matrix key == work_dir subdir).
# Order defines the stable colour/legend index used across all plots.
_MODEL_KEYS = dict(
    # AMR-Benchmark methods (15)
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
    # Project-own methods
    MLDNN='mldnn',
    HCGDNN='hcgdnn',
    FastMLDNN='fastmldnn',
    DSCLDNN='dscldnn',
    MCformer='mcformer',
    TRN='trn',
)

_DATASETS = ['deepsig201610A', 'deepsig201610B', 'deepsig201801A', 'hisar2019']

# Methods without a run on a given dataset (TRN only has a RML2016.10A run).
_EXCLUDE = dict(
    deepsig201610B=['TRN'],
    deepsig201801A=['TRN'],
    hisar2019=['TRN'],
)

info = dict(
    work_dir='work_dirs/amr_benchmark',
    save_dir='work_dirs/performance',
    methods={name: idx for idx, name in enumerate(_MODEL_KEYS)},
    publish={
        ds: {
            disp: f'{key}/{ds}'
            for disp, key in _MODEL_KEYS.items()
            if disp not in _EXCLUDE.get(ds, [])
        }
        for ds in _DATASETS
    },
)

del _MODEL_KEYS, _DATASETS, _EXCLUDE
