# Baseline Reference Registry

This registry is the gatekeeper for RCPS experiments. A model is not eligible for RCPS comparison until its hard-label baseline is reproduced within the registered tolerance.

## Policy

- AMC references are anchored to AMR-Benchmark and model-specific papers. AMR-Benchmark is Keras/TensorFlow based, so PyTorch ports must audit initialization, padding, channel layout, optimizer details, scheduler, split, and epoch budget before judging RCPS.
- The first hard gate is `MLDNN + RadioML2016.10A`: the existing CSRR checkpoint reaches about 62.97%, so new runs must reach at least 61% before RCPS modifications are compared.
- Current `main_10ep_3seed` outputs are marked diagnostic-invalid because they used a 10-epoch MCLDNN path and do not meet the baseline gate.
- Vision and audio gates will use TorchVision/RobustBench and TensorFlow/torchaudio references respectively before adding RCPS.

## Source Anchors

- AMR-Benchmark: https://github.com/Richardzhangxx/AMR-Benchmark
- AMR-Benchmark paper: https://doi.org/10.1016/j.dsp.2022.103650
- DeepSig datasets: https://www.deepsig.ai/datasets/
- Keras Conv2D default initializer: https://keras.io/api/layers/convolution_layers/convolution2d/
- PyTorch Conv2d initialization: https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv2d.html
- CIFAR-C: https://openreview.net/forum?id=HJz6tiCqYm
- RobustBench: https://github.com/RobustBench/robustbench
- Speech Commands: https://tensorflow.google.cn/datasets/catalog/speech_commands


## Current AMC Gate Status

- MLDNN on the native server split is a chain-health anchor, not a main model: three-seed hard CE test accuracy is `62.5519 +/- 0.0870`.
- GRU2 on the AMR-compatible split is useful for RCPS diagnostics but has high seed variance: three-seed hard CE test accuracy is `58.9894 +/- 1.3604`.
- PETCGDNN with Keras-compatible initialization is the current stable non-MLDNN candidate: three-seed hard CE test accuracy is `59.9371 +/- 0.4106`, and paired RCPS diagnostics are running.
- MCLDNN with the current Keras-init override failed to learn and is not used for RCPS claims until the parity issue is debugged.

Decision rule: passing a gate does not mean the model is strong enough for the main TPAMI table. It only means paired RCPS comparison is scientifically meaningful. The main table should keep models that are both stable and externally defensible.

## Next Gate Order

1. Reproduce `MLDNN + RadioML2016.10A` hard CE with original 400-epoch schedule.
2. Reproduce `CGDNet`, `PETCGDNN`, `FastMLDNN`, and `MCformer` with maintained schedules and strict `train/validation/test` splits.
3. For each passing hard CE baseline, build validation-only RCPS tables: entropy-matched epsilon and reliability-conditioned posterior base.
4. Only after passing hard CE parity, run static label smoothing, confidence penalty, `RCPS-Retention`, `RCPS-EntropyMatch`, and `RCPS-PosteriorBase` with identical training budgets.
5. Extend the same baseline-first discipline to RadioML2016.10B and RadioML2018.01A.

## TPAMI Evidence Axes

- Accuracy: overall and reliability/SNR-stratified.
- Posterior quality: NLL, ECE, Brier score, mean confidence, and mean entropy.
- Reliability retention: high-SNR accuracy drop must remain within 1 percentage point unless explicitly discussed as a tradeoff.
- Low-reliability behavior: low-SNR NLL/ECE/Brier should improve for most passing models before the paper claims posterior alignment.
- Training efficiency: best epoch, validation AULC, time/epoch to target validation accuracy, and seed variance.
- Compute cost: parameter and runtime overhead should be reported; loss-only RCPS is expected to leave the backbone unchanged.


## Claim Gate

- Accuracy claim: RCPS must improve mean accuracy over hard CE or static label smoothing on at least two AMC datasets and two model families while keeping high-reliability accuracy drop within `1 pp`.
- Calibration claim: low-reliability bins must improve on most of NLL, ECE, and Brier versus hard CE and static label smoothing.
- Efficiency claim: RCPS must reduce epoch/time to a fixed validation target or improve validation AULC in at least two stable model settings before the paper claims training-efficiency benefits.
- Generality claim: AMC is the main controlled setting, but at least one vision corruption benchmark and one noisy-audio benchmark must reproduce the reliability-conditioned posterior-quality effect before the paper claims cross-domain generality.
- If accuracy gains are inconsistent but posterior-quality gains are stable, the manuscript claim must be narrowed to posterior calibration and uncertainty alignment.
