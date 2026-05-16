# RCPS Current Evidence Summary

CSV: `/home/citybuster/Data/RCPS/work_dirs/paper_evidence/rcps_current_evidence_summary.csv`

This document separates main positive evidence from diagnostic tradeoffs. Diagnostic rows are theory-shaping constraints and should not be used as headline wins.

| Tier | Modality | Dataset | Model | Method | Seeds | dAcc(pp) | dNLL | dECE | dBrier | Interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| main-positive | AMC | RadioML2016.10A | PETCGDNN | RCPS-PosteriorBase | 3 | 0.8621 | -0.0200 | -0.0178 | -0.0139 | strong AMC positive result |
| main-positive | AMC | RadioML2016.10A | MCformer | RCPS-Hybrid eps0.10 | 3 | 0.2970 | -0.0064 | -0.0022 | -0.0018 | strong-model AMC positive result |
| main-positive-retention | AMC | RadioML2016.10B | MCformer | RCPS-Hybrid eps0.05 | 3 | 0.0458 | -0.0040 | -0.0006 | -0.0009 | second AMC dataset; conservative retention setting |
| supplementary | AMC | UCSD/RML22 | CNN4 | RCPS-Retention | 3 | 1.3089 | -0.0197 | +0.0079 | -0.0084 | third AMC dataset but high seed variance and CNN4-only gate |
| main-positive-crossmodal | Vision | CIFAR-10-C | ResNet18-CIFAR | RCPS-Retention eps0.10 | 3 | 0.3662 | -0.0055 | -0.0236 | -0.0048 | strongest cross-modal positive evidence |
| diagnostic | Vision | CIFAR-10-C | ResNet34-CIFAR | RCPS-Retention eps0.05 | 1 | -0.3020 | +0.0015 | -0.0168 | +0.0027 | fixed uniform RCPS improves ECE but violates accuracy/Brier retention; do not expand |
| diagnostic | Audio | SpeechCommands-v0.02 | DS-CNN | RCPS-Retention eps0.05 | 1 | 0.0468 | +0.0072 | +0.0261 | -0.0002 | audio motivates validation-constrained/entropy-matched RCPS; fixed smoothing not main result |

## Current Reading

- The defensible main claim is posterior calibration and uncertainty alignment with high-reliability retention; accuracy gains are present in several AMC and CIFAR-10-C settings but are not universal.
- Static label smoothing is not a sufficient baseline substitute: it repeatedly worsens NLL/ECE/Brier in CIFAR-10-C, UCSD/RML22, and Speech Commands diagnostics.
- Fixed uniform RCPS should not be presented as the final theory. The manuscript now frames finite-reliability smoothing as validation-constrained posterior approximation.
- Next experimental priority: consolidate main AMC and CIFAR-10-C figures/tables, then add a stronger adaptive/PosteriorBase audio variant only if validation diagnostics justify it.
