log_dir = '/home/zry/Data/SignalProcessing/Workdir'
legends = {
    'MCBLDN': 0,
    'SCBDN_RMS': 1,
    'LCBDN_RMS': 2,
    'SLCBDN_RMS': 3,
    'MCBLDN_RMS': 4,
    'SCBDN': 5,
    'LCBDN': 6,
    'SLCBDN': 7,
    'SCDN_32_Ds10': 8,
    'SCDN_64_Ds10': 9,
    'SCDN_128_Ds10': 10,
    'MCBLDN_128_Ds10_1200': 11,
    'MCBLDN_32_Ds10_4000': 12,
    'MCBLDN_64_Ds150_1200': 13,
    'MCBLDN_64_Ds10_4000': 14
}

plot = dict(
    type='MLDNNPlot',
    log_dir=log_dir,
    config='cpcstn_matslot_Con_128_Ds10_4000_Adam',
    legends=legends,
    # Set the configs about confusion maps
    confusion_maps=[
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_cpcstn_128_Ds10_4000_Adam.pdf',
            config='cpcstn_matslot_Con_128_Ds10_4000_Adam',
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_cpcnn_128_Ds10_4000_Adam.pdf',
            config='cpcnn_matslot_Con_128_Ds10_4000_Adam',
        ),
        dict(
            type='ConfusionMap',
            log_dir=log_dir,
            name='confusion_map_cstn_128_Ds10_4000_Adam.pdf',
            config='cstn_matslot_Con_128_Ds10_4000_Adam',
        ),
    ],
    # Set the configs about snr accuracy and modulation F1 score
    snr_modulation=[

        # Motivations
        dict(
            type='AccuracyF1Plot',
            name='scbdn.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='con_128_ds10_4000_mcbldn',
                    name='MCBLDN',
                ),
                dict(
                    config='cstn_matslot_Con_128_Ds10_4000',
                    name='SCBDN_RMS',
                ),
            ],
        ),
        dict(
            type='AccuracyF1Plot',
            name='lcbdn.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='con_128_ds10_4000_mcbldn',
                    name='MCBLDN',
                ),
                dict(
                    config='cpcnn_matslot_Con_128_Ds10_4000',
                    name='LCBDN_RMS',
                ),
            ],
        ),
        dict(
            type='AccuracyF1Plot',
            name='slcbdn.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                dict(
                    config='con_128_ds10_4000_mcbldn',
                    name='MCBLDN',
                ),
                dict(
                    config='cpcstn_matslot_Con_128_Ds10_4000',
                    name='SLCBDN_RMS',
                ),
            ],
        ),

        # Ablation Study of Adam

        dict(
            type='AccuracyF1Plot',
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
                    config='cstn_matslot_Con_128_Ds10_4000',
                    name='SCBDN_RMS',
                ),
                dict(
                    config='cstn_matslot_Con_128_Ds10_4000_Adam',
                    name='SCBDN',
                ),
                # LCBDN
                dict(
                    config='cpcnn_matslot_Con_128_Ds10_4000',
                    name='LCBDN_RMS',
                ),
                dict(
                    config='cpcnn_matslot_Con_128_Ds10_4000_Adam',
                    name='LCBDN',
                ),
                # SLCBDN
                dict(
                    config='cpcstn_matslot_Con_128_Ds10_4000',
                    name='SLCBDN_RMS',
                ),
                dict(
                    config='cpcstn_matslot_Con_128_Ds10_4000_Adam',
                    name='SLCBDN',
                ),
            ],
        ),

        # table
        dict(
            type='AccuracyF1Plot',
            name='table.pdf',
            legends=legends,
            log_dir=log_dir,
            methods=[
                # R = 32， 64 ，128
                dict(
                    config='con_32_ds10_36000_scdn',
                    name='SCDN_32_Ds10',
                ),
                dict(
                    config='con_64_ds10_36000_scdn',
                    name='SCDN_64_Ds10',
                ),
                dict(
                    config='con_128_ds10_36000_scdn',
                    name='SCDN_128_Ds10',
                ),
                # number of slot
                dict(
                    config='con_128_ds10_4000_mcbldn',
                    name='MCBLDN',
                ),
                dict(
                    config='con_128_ds150_1200_mcbldn',
                    name='MCBLDN_128_Ds10_1200',
                ),
                # other
                dict(
                    config='con_32_ds10_4000_mcbldn',
                    name='MCBLDN_32_Ds10_4000',
                ),
                dict(
                    config='con_64_ds150_1200_mcbldn',
                    name='MCBLDN_64_Ds150_1200',
                ),
                dict(
                    config='con_64_ds10_4000_mcbldn',
                    name='MCBLDN_64_Ds10_4000',
                ),
            ],
        ),
    ],
)
