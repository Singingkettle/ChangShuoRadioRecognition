# RCPS Current Evidence Summary

CSV: `/home/citybuster/Data/RCPS/work_dirs/paper_evidence/rcps_current_evidence_summary.csv`

This document separates main positive evidence from diagnostic tradeoffs. Diagnostic rows are theory-shaping constraints and should not be used as headline wins.

| Tier | Modality | Dataset | Model | Method | Seeds | dAcc(pp) | dNLL | dECE | dBrier | Interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| main-positive-dpc | AMC | RadioML2016.10B | MCformer | DPC-RCPS | 3 | 5.5075 | -0.0820 | +0.0012 | -0.0588 | strongest current AMC result; large accuracy/NLL/Brier and high-SNR retention gain; aggregate ECE slightly worse |
| main-positive-dpc | AMC | RadioML2016.10A | PETCGDNN | DPC-RCPS | 3 | 3.0250 | -0.0585 | -0.0113 | -0.0328 | strong DPC positive result on first AMC dataset/backbone |
| diagnostic-dpc | AMC | RadioML2016.10A | MCformer | DPC-RCPS | 3 | 0.0852 | -0.0014 | +0.0012 | -0.0003 | strong-model boundary case; modest NLL/Brier/high-SNR benefit but no large overall gain |
| main-positive | AMC | RadioML2016.10A | PETCGDNN | RCPS-PosteriorBase | 3 | 0.8621 | -0.0200 | -0.0178 | -0.0139 | strong AMC positive result |
| main-positive | AMC | RadioML2016.10A | MCformer | RCPS-Hybrid eps0.10 | 3 | 0.2970 | -0.0064 | -0.0022 | -0.0018 | strong-model AMC positive result |
| main-positive-retention | AMC | RadioML2016.10B | MCformer | RCPS-Hybrid eps0.05 | 3 | 0.0458 | -0.0040 | -0.0006 | -0.0009 | second AMC dataset; conservative retention setting |
| supplementary | AMC | UCSD/RML22 | CNN4 | RCPS-Retention | 3 | 1.3089 | -0.0197 | +0.0079 | -0.0084 | third AMC dataset but high seed variance and CNN4-only gate |
| main-positive-crossmodal | Vision | CIFAR-10-C | ResNet18-CIFAR | RCPS-Retention eps0.10 | 3 | 0.3662 | -0.0055 | -0.0236 | -0.0048 | strongest cross-modal positive evidence |
| diagnostic | Vision | CIFAR-10-C | ResNet34-CIFAR | RCPS-Retention eps0.05 | 1 | -0.3020 | +0.0015 | -0.0168 | +0.0027 | fixed uniform RCPS improves ECE but violates accuracy/Brier retention; do not expand |
| diagnostic | Audio | SpeechCommands-v0.02 | DS-CNN | RCPS-Retention eps0.05 | 1 | 0.0468 | +0.0072 | +0.0261 | -0.0002 | audio motivates validation-constrained/entropy-matched RCPS; fixed smoothing not main result |

## Current Reading

- The defensible main claim is now stronger than fixed smoothing: DPC-RCPS provides degradation-posterior consistency, posterior-quality gains, and high-reliability retention. Accuracy gains are large in `MCformer + RadioML2016.10B` and `PETCGDNN + RadioML2016.10A`, modest in `MCformer + RadioML2016.10A`, and positive in the selected CIFAR-10-C ResNet18 setting.
- Static label smoothing is not a sufficient baseline substitute: it repeatedly worsens NLL/ECE/Brier in CIFAR-10-C, UCSD/RML22, and Speech Commands diagnostics.
- Fixed uniform RCPS should not be presented as the final theory. The manuscript now frames finite-reliability smoothing as a low-cost approximation and DPC-RCPS as the main posterior-path method.
- Next experimental priority: consolidate DPC main AMC and CIFAR-10-C figures/tables, then decide whether to run DPC/PosteriorBase audio only if validation diagnostics justify it.
