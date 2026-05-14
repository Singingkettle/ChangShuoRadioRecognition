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
