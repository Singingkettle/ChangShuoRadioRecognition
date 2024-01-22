_base_ = [
    '../_base_/performance_info/amc.py',
]

performance = dict(
    type='Classification',
    save_dir='/home/citybuster/Data/WirelessRadio/work_dir/fastmldnn_iq-ap-channel-deepsig201610A/paper',
    info=None,
    # Set the figure_configs about confusion maps
    Figures=[
        # dict(
        #     type='ConfusionMap',
        #     dataset=dict(
        #         deepsig201610A=[
        #             'FastMLDNN'
        #         ],
        #     ),
        # ),
        dict(
            type='SNRVsAccuracy',
            dataset=dict(
                deepsig201610A=dict(
                    # control_experiment_data=[
                    #     ['FastMLDNN', r'InputShape $[N\times 4\times 1\times 128]$'],
                    #     ['FastMLDNN_V1', r'InputShape $[N\times 1\times 4\times 128]$']
                    # ],
                    # control_experiment_conv=[
                    #     ['FastMLDNN', 'Group Conv.'], ['FastMLDNN_V2', 'Conv.']
                    # ],
                    # control_experiment_temporal=[
                    #     ['FastMLDNN', 'Transformer'], ['FastMLDNN_V3', 'RNN'],
                    #     ['FastMLDNN_V4', 'lstm2'], ['FastMLDNN_V5', 'BiGRU']
                    # ],
                    # control_experiment_merge=[
                    #     ['FastMLDNN', 'Sum'], ['FastMLDNN_V6', 'Last'], ['FastMLDNN_V7', 'LogSumExp'],
                    #     ['FastMLDNN_V8', 'Max'], ['FastMLDNN_V9', 'Mean'], ['FastMLDNN_V10', 'Median'],
                    #     ['FastMLDNN_V11', 'Min'], ['FastMLDNN_V12', 'Quantile'], ['FastMLDNN_V13', 'Std']
                    # ],
                    # control_experiment_loss=[
                    #     ['FastMLDNN', r'$L_{ce}+\beta L_{de}$'], ['FastMLDNN_V14', r'$L_{ce}$']
                    # ],
                    # control_experiment_balance=[
                    #     ['FastMLDNN', r'$\beta = 0.5$'], ['FastMLDNN_V15', r'$\beta = 0.1$'],
                    #     ['FastMLDNN_V16', r'$\beta = 0.3$'], ['FastMLDNN_V17', r'$\beta = 0.7$'],
                    #     ['FastMLDNN_V18', r'$\beta = 0.9$']
                    # ],
                    # iq_ap_co_fb=[
                    #     'FastMLDNN', 'MLDNN', 'DSCLDNN', 'rescnn', 'CLDNN', 'CNN4', 'DensCNN',
                    #     'DecisionTree_FB', 'SVM_FB', 'SSRCNN', 'VGGNet', 'AlexNet', 'GoogleNet', 'ResNet',
                    # ],

                    control_experiment_loss=[
                        ['FastMLDNN', r'$L_{ce}+\beta L_{de}$'],
                        ['FastMLDNN_V14', r'$L_{ce}$'],
                        ['FastMLDNN_V21', r'$L_{ce}+L_{c}$'],
                        ['FastMLDNN_V22', r'$L_{f}$'],
                        ['FastMLDNN_V23', r'$L_{ghmc}$'],
                        ['FastMLDNN_V24', r'$L_{kd}$'],
                    ],

                    control_experiment_transformer=[
                        'FastMLDNN',
                        'MCformerLarge',
                        'MCformerSmall',
                        'TRN',
                    ],
                )
            ),
        ),
        # dict(
        #     type='ClassVsF1ScoreWithSNR',
        #     dataset=dict(
        #         deepsig201610A=dict(
        #             control_experiment_data=[
        #                 ['FastMLDNN', r'InputShape $[N\times 4\times 1\times 128]$'],
        #                 ['FastMLDNN_V1', r'InputShape $[N\times 1\times 4\times 128]$']
        #             ],
        #             control_experiment_conv=[
        #                 ['FastMLDNN', 'Group Conv.'], ['FastMLDNN_V2', 'Conv.']
        #             ],
        #             control_experiment_temporal=[
        #                 ['FastMLDNN', 'Transformer'], ['FastMLDNN_V3', 'RNN'],
        #                 ['FastMLDNN_V4', 'lstm2'], ['FastMLDNN_V5', 'BiGRU']
        #             ],
        #             control_experiment_merge=[
        #                 ['FastMLDNN', 'Sum'], ['FastMLDNN_V6', 'Last'], ['FastMLDNN_V7', 'LogSumExp'],
        #                 ['FastMLDNN_V8', 'Max'], ['FastMLDNN_V9', 'Mean'], ['FastMLDNN_V10', 'Median'],
        #                 ['FastMLDNN_V11', 'Min'], ['FastMLDNN_V12', 'Quantile'], ['FastMLDNN_V13', 'Std']
        #             ],
        #             control_experiment_loss=[
        #                 ['FastMLDNN', r'$L_{ce}+\beta L_{de}$'], ['FastMLDNN_V14', r'$L_{ce}$']
        #             ],
        #             control_experiment_balance=[
        #                 ['FastMLDNN', r'$\beta = 0.5$'], ['FastMLDNN_V15', r'$\beta = 0.1$'],
        #                 ['FastMLDNN_V16', r'$\beta = 0.3$'], ['FastMLDNN_V17', r'$\beta = 0.7$'],
        #                 ['FastMLDNN_V18', r'$\beta = 0.9$']
        #             ],
        #             iq_ap_co_fb=[
        #                 'FastMLDNN', 'MLDNN', 'DSCLDNN', 'rescnn', 'CLDNN', 'CNN4', 'DensCNN',
        #                 'DecisionTree_FB', 'SVM_FB', 'SSRCNN', 'VGGNet', 'AlexNet', 'GoogleNet', 'ResNet',
        #             ],
        #         )
        #     ),
        # ),
        # dict(
        #     type='VisFea',
        #     dataset=dict(
        #         deepsig201610A=[
        #             ['FastMLDNN_V14', r'$L_{ce}+\beta L_{de}$'],
        #             ['FastMLDNN', r'$L_{ce}$']
        #         ]
        #     )
        # )
    ],
    # Tables=[
    #     dict(
    #         type='Flops',
    #         dataset=dict(
    #             deepsig201610A=[
    #                 'FastMLDNN', 'FastMLDNN_V1', 'FastMLDNN_V2', 'FastMLDNN_V3', 'FastMLDNN_V4', 'FastMLDNN_V5', 'MLDNN'
    #             ]
    #         )
    #     )
    # ]
    Tables=[
        dict(
            type='Flops',
            dataset=dict(
                deepsig201610A=[
                    'DSCLDNN'
                ]
            )
        )
    ]
)
