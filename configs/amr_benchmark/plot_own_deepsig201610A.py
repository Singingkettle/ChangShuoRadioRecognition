# Plot config for the project's own methods (MLDNN / HCGDNN / FastMLDNN) on
# RML2016.10A. Based on the 21-method registry ``amc.py`` so every method's
# colour / linestyle / marker is fixed by its registry index everywhere it
# appears (globally consistent styling; own methods are indices 15/16/17).
#
# Figures degrade gracefully: any method whose
# ``work_dirs/amr_benchmark/<model>/<dataset>/res/paper.pkl`` is not present
# yet is skipped (no crash), so this config can be run *now* against the
# already-finished baselines and will auto-include the own methods the moment
# their training produces paper.pkl.
_base_ = ['../_base_/performance_info/amc.py']

DATASET = 'deepsig201610A'

# The 15 AMR-Benchmark baselines + the 3 project-own methods (the methods that
# actually have a planned run; DSCLDNN/MCformer/TRN have no run and are left to
# the registry only, for colour stability).
BASELINES = [
    'CNN1', 'CNN2', 'MCNET', 'ICAMCNet', 'ResNet', 'DenseNet',
    'GRU', 'LSTM', 'DAE', 'MCLDNN', 'CLDNNW', 'CLDNN2',
    'CGDNet', 'PETCGDNN', 'CNN1DPF',
]
OWN = ['MLDNN', 'HCGDNN', 'FastMLDNN']
ALL_METHODS = BASELINES + OWN

# Strong baselines for head-to-head experiment curves against each own method.
STRONG = ['MCLDNN', 'PETCGDNN']

# SNR-vs-accuracy comparison groups: the three own methods together, each own
# method head-to-head vs strong baselines, and the full field for context.
SNR_GROUPS = dict(
    all_methods=ALL_METHODS,
    own_methods=OWN,
    mldnn_vs_baselines=['MLDNN'] + STRONG,
    hcgdnn_vs_baselines=['HCGDNN'] + STRONG,
    fastmldnn_vs_baselines=['FastMLDNN'] + STRONG,
)

# F1 / ROC / PR groups: own methods together + per-method head-to-heads
# (the full 18-way radar is too dense to read, so it is omitted here).
COMPARE_GROUPS = dict(
    own_methods=OWN,
    mldnn_vs_baselines=['MLDNN'] + STRONG,
    hcgdnn_vs_baselines=['HCGDNN'] + STRONG,
    fastmldnn_vs_baselines=['FastMLDNN'] + STRONG,
)

# Confusion matrices for each own method individually (per SNR).
CONFUSION_METHODS = OWN

# Training curves: own methods overlaid with the strong baselines.
TRAIN_METHODS = OWN + STRONG

performance = dict(
    type='Classification',
    Figures=[
        dict(
            type='SNRVsAccuracy',
            dataset={DATASET: SNR_GROUPS},
        ),
        dict(
            type='ClassVsF1ScoreWithSNR',
            dataset={DATASET: COMPARE_GROUPS},
        ),
        dict(
            type='ConfusionMap',
            dataset={DATASET: CONFUSION_METHODS},
        ),
        dict(
            type='TrainPlot',
            dataset={DATASET: TRAIN_METHODS},
            loss_metric='loss',
            val_metric='accuracy/top1',
        ),
        dict(
            type='ROCCurve',
            dataset={DATASET: COMPARE_GROUPS},
            snr_groups=['micro'],
            average='macro',
        ),
        dict(
            type='PRCurve',
            dataset={DATASET: COMPARE_GROUPS},
            snr_groups=['micro'],
            average='micro',
        ),
    ],
    Tables=[
        dict(
            type='ModulationSummary',
            dataset={DATASET: ALL_METHODS},
        ),
        dict(
            type='Flops',
            dataset={DATASET: ALL_METHODS},
            measure_time=False,
        ),
    ],
)
