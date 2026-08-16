# Critical-band strict-DPC route on the synthetic AWGN anchor benchmark.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = [
    './petcgdnn_strict-awgn-dpc_iq-snr-synthetic-awgn-v1.py',
]

work_dir = 'work_dirs/synthetic_awgn/petcgdnn_critical-strict-dpc'
method_name = 'critical_strict_awgn_dpc_rcps'

strict_dpc_source = 'work_dirs/synthetic_awgn/teacher_posteriors/petcgdnn_stage1_seed2026_30ep_strict_awgn_dpc_trainval.npz'

model = dict(
    head=dict(
        loss=dict(
            epsilon=dict(
                type='table',
                bins=[0.0, 0.20, 0.35, 0.65, 0.80, 1.0],
                values=[0.05, 0.20, 0.50, 0.50, 0.08, 0.0]),
            base=dict(source=strict_dpc_source),
        ),
    ),
)
