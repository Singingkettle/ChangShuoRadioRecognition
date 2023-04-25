_base_ = [
    '../_base_/performance/amc.py',
]

performance = dict(
    type='Classification',
    # Set the figure_configs about confusion maps
    Figures=[
        dict(
            type='ConfusionMap',
            dataset=dict(
                deepsig201610A=[
                    'MLDNN'
                ],
                has_SNR=True
            ),
        ),
        dict(
            type='Accuracy',
            dataset=dict(
                deepsig201610A=[
                    dict(
                        iq_ap=[
                            'MLDNN', 'DSCLDNN', 'ResCNN', 'CLDNN', 'CNN4', 'DensCNN',
                        ],
                        co=[
                            'MLDNN', 'AlexNet', 'GoogleNet', 'ResNet', 'VGGNet', 'SVM-FB', 'DecisionTree-FB',
                        ]
                    )
                ],
                deepsig201801A=[
                    dict(
                        iq_ap=[
                            'MLDNN', 'DSCLDNN', 'ResCNN', 'CLDNN', 'CNN4', 'DensCNN',
                        ],
                    )
                ]
            ),
        ),
        dict(
            type='FScore',
            dataset=dict(
                deepsig201610A=[
                    dict(
                        iq_ap=[
                            'MLDNN', 'DSCLDNN', 'ResCNN', 'CLDNN', 'CNN4', 'DensCNN',
                        ],
                        co=[
                            'MLDNN', 'AlexNet', 'GoogleNet', 'ResNet', 'VGGNet', 'SVM-FB', 'DecisionTree-FB',
                        ]
                    )
                ],
                deepsig201801A=[
                    dict(
                        iq_ap=[
                            'MLDNN', 'DSCLDNN', 'ResCNN', 'CLDNN', 'CNN4', 'DensCNN',
                        ],
                    )
                ]
            ),
        )
    ]
)
