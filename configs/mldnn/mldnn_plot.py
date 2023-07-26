_base_ = [
    '../_base_/performance_info/amc.py',
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
                        abl=[
                            'CNN3', 'CNN3_ap', 'MLDNN', 'MLDNN_V3', 'MLDNN_V4',
                            'MLDNN_V5', 'MLDNN_V6', 'MLDNN_V7', 'MLDNN_V8', 'MLDNN_V9'
                        ],
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
    ],
    Tables=[
        dict(
            type='GetFlops',
            dataset=dict(
                deepsig201610A=[
                    'MLDNN',
                ],
            )
        ),
        dict(
            type='Summary',
            dataset=dict(
                deepsig201610A=[
                    'MLDNN'
                ],
            )
        )
    ]
)
