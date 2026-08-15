# AMC method registry: maps display names used in figures to (a) a stable
# colour/legend index and (b) the work-dir subdirectory holding that run.
#
# Path convention: ``tools/test.py`` writes ``<work_dir>/res/paper.pkl``.
# Comparison plots that collect many methods can use a nested tree
# ``<work_dir>/<model>/<dataset>/res/paper.pkl``; publish entries below are
# the ``<model>/<dataset>`` subdir. Point ``work_dir`` at that tree.
#
# Dataset keys:
#   deepsig201610A -> RML2016.10A ; deepsig201610B -> RML2016.10B
#   deepsig201801A -> RML2018.01A ; hisar2019       -> HisarMod2019.1

# Display name -> orchestrator model key (== matrix key == work_dir subdir).
# Order defines the stable colour/legend index used across all plots.
_MODEL_KEYS = dict(
    # Published AMC baselines (15)
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
