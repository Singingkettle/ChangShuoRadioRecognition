_base_ = [
    '../_base_/performance_info/amc.py',
]

performance = dict(
    type='Classification',
    Figures=[
        dict(
            type='SNRVsAccuracy',
            dataset=dict(
                deepsig201610A=dict(
                    comparison=['CNN2', 'CNN4', 'DensCNN'],
                ),
            ),
        ),
        dict(
            type='ClassVsF1ScoreWithSNR',
            dataset=dict(
                deepsig201610A=dict(
                    comparison=['CNN2', 'CNN4', 'DensCNN'],
                ),
            ),
        ),
        dict(
            type='ConfusionMap',
            dataset=dict(
                deepsig201610A=['CNN2'],
            ),
        ),
    ],
)
