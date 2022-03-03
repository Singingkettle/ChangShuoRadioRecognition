log_dir = '/home/zry/Data/SignalProcessing/Workdir'
legends = {
    'MCBLDN': 0,
    'SCBDN_RMSprop': 1,
    'LCBDN_RMSprop': 2,
    'SLCBDN_RMSprop': 3,
    'MCBLDN_RMSprop': 4,
    'MCBLDN_Adam': 5,
    'SCBDN_Adam': 6,
    'LCBDN_Adam': 7,
    'SLCBDN': 8,
    'SCDN_32_Ds10': 9,
    'SCDN_64_Ds10': 10,
    'SCDN_128_Ds10': 11,
    'MCBLDN_128_Ds150_1200': 12,
    'MCBLDN_32_Ds10_4000': 13,
    'MCBLDN_64_Ds150_1200': 14,
    'MCBLDN_64_Ds10_4000': 15,
    'V1': 16,
    'V2': 17,
    'V3': 18,
    'V4': 19,
    'V5': 20,
    'V6': 21,
    'VGGNet': 22,
    'AlexNet': 23,
    'ResNet': 24,
    'GoogleNet': 25,
    'SVM-FB': 26,
    'DecisionTree-FB': 27,
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='cpcstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
    legends=legends,
    #  Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[

        # Motivations
        # dict(
        #     type='SNRModulationCurve',
        #     name='scbdn.pdf',
        #     legends=legends,
        #     log_dir=log_dir,
        #     methods=[
        #         dict(
        #             config='con_128_ds10_4000_mcbldn',
        #             name='MCBLDN_RMS',
        #         ),
        #         dict(
        #             config='cstn_matslot_Con_128_Ds10_4000_v5',
        #             name='SCBDN_RMS',
        #         ),
        #     ],
        # ),
        # dict(
        #     type='SNRModulationCurve',
        #     name='lcbdn.pdf',
        #     legends=legends,
        #     log_dir=log_dir,
        #     methods=[
        #         dict(
        #             config='con_128_ds10_4000_mcbldn',
        #             name='MCBLDN',
        #         ),
        #         dict(
        #             config='cpcnn_matslot_Con_128_Ds10_4000',
        #             name='LCBDN_RMS',
        #         ),
        #     ],
        # ),
        # dict(
        #     type='SNRModulationCurve',
        #     name='slcbdn.pdf',
        #     legends=legends,
        #     log_dir=log_dir,
        #     methods=[
        #         dict(
        #             config='con_128_ds10_4000_mcbldn',
        #             name='MCBLDN',
        #         ),
        #         dict(
        #             config='cpcstn_matslot_Con_128_Ds10_4000_v5',
        #             name='SLCBDN_RMS',
        #         ),
        #     ],
        # ),

        # Ablation Study of Adam

        dict(
            type='SNRModulationCurve',
            name='ablation_Adam.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                # MCBLDN
                dict(
                    config='con_128_ds10_4000_mcbldn',
                    name='MCBLDN',
                ),
                dict(
                    config='con_128_ds10_4000_mcbldn_Adam',
                    name='MCBLDN_Adam',
                ),
                # SCBDN
                dict(
                    config='cstn_matslot_Con_128_Ds10_4000_v5',
                    name='SCBDN_RMSprop',
                ),
                dict(
                    config='cstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
                    name='SCBDN_Adam',
                ),
                # LCBDN
                dict(
                    config='cpcnn_matslot_Con_128_Ds10_4000',
                    name='LCBDN_RMSprop',
                ),
                dict(
                    config='cpcnn_matslot_Con_128_Ds10_4000_Adam',
                    name='LCBDN_Adam',
                ),
                # SLCBDN
                dict(
                    config='cpcstn_matslot_Con_128_Ds10_4000_v5',
                    name='SLCBDN_RMSprop',
                ),
                dict(
                    config='cpcstn_matslot_Con_128_Ds10_4000_Adam_v5_0.0002',
                    name='SLCBDN',
                ),
            ],
        ),

    ],
)
