_base_ = ['../_base_/performance_info/amr_benchmark.py']

DATASET = 'deepsig201610A'

ALL_METHODS = [
    'CNN1', 'CNN2', 'MCNET', 'ICAMCNet', 'ResNet', 'DenseNet',
    'GRU', 'LSTM', 'DAE', 'MCLDNN', 'CLDNNW', 'CLDNN2',
    'CGDNet', 'PETCGDNN', 'CNN1DPF',
]

# A small subset used for confusion-map output (otherwise we emit one PDF per
# SNR per method which is too many).
CONFUSION_METHODS = ['MCLDNN', 'PETCGDNN', 'MCNET']

performance = dict(
    type='Classification',
    Figures=[
        dict(
            type='SNRVsAccuracy',
            dataset={DATASET: dict(comparison=ALL_METHODS)},
        ),
        dict(
            type='ClassVsF1ScoreWithSNR',
            dataset={DATASET: dict(comparison=ALL_METHODS)},
        ),
        dict(
            type='ConfusionMap',
            dataset={DATASET: CONFUSION_METHODS},
        ),
        dict(
            type='TrainPlot',
            dataset={DATASET: ALL_METHODS},
            loss_metric='loss',
            val_metric='accuracy/top1',
        ),
        dict(
            type='ROCCurve',
            dataset={DATASET: dict(comparison=ALL_METHODS)},
            snr_groups=['micro'],
            average='macro',
        ),
        dict(
            type='PRCurve',
            dataset={DATASET: dict(comparison=ALL_METHODS)},
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
