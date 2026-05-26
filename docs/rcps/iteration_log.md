# RCPS Iteration Log

## Iteration 0: Repository and Data Grounding

- Goal: implement Reliability-Conditioned Posterior Supervision experiments without changing existing hard-label configs.
- Server data root: `/home/citybuster/Data`.
- AMC data confirmed under `/home/citybuster/Data/WirelessRadio/data/ModulationClassification`.
- Existing AMC datasets: DeepSig RadioML 2016.04C, 2016.10A, 2016.10B, 2018.01A; HisarMod2019.1; UCSD RML22.
- Vision data confirmed: `/home/citybuster/Data/Visual/CIFAR-10`.
- Missing cross-domain data to prepare: CIFAR-10-C and Speech Commands v0.02.
- First pilot: CNN2 on RadioML.2016.10A with Hard CE, Static LS, and RCPS-Uniform.

## Iteration 1: AMC CNN2 One-Epoch Pilot

- Branch: `feature/rcps-experiments`.
- Commits: `5f21e3b` added the RCPS framework, `c641448` added worker override, `2eb6c29` made method losses explicit in configs.
- Dataset: `/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A`.
- Command pattern: `python tools/rcps/run_amc_matrix.py --models cnn2 --methods <method> --seeds 2026 --max-epochs 1 --num-workers 0 --execute`.
- Smoke result: training, checkpoint saving, `tools/test.py`, and `tools/rcps/analyze_reliability.py` all completed.
- Test accuracy after 1 epoch: Hard CE 27.68, Static LS 28.38, RCPS-Uniform 32.29.
- Low-SNR check: at -20 dB, RCPS-Uniform improved NLL/ECE over Hard CE (NLL 2.4233 vs. 2.4636; ECE 0.0782 vs. 0.0934).
- High-SNR check: at 18 dB, RCPS-Uniform had higher accuracy but larger ECE than Hard CE, indicating under-confidence from overly strong smoothing.
- Next action: add epsilon calibration grid and high-reliability retention checks before any full 3-seed claim.

## Iteration 2: Cross-Domain Data Preparation

- Speech Commands v0.02 source: `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`.
- Speech Commands archive downloaded to `/home/citybuster/Data/RCPS/raw/SpeechCommands/speech_commands_v0.02.tar.gz`.
- Speech Commands annotations written to `/home/citybuster/Data/RCPS/processed/ReliabilityClassification/Audio/SpeechCommands-v0.02`.
- Annotation sizes: train 593901, validation 69867, test 77035 noisy/clean entries.
- CIFAR-10-C source: `https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1`.
- CIFAR-10-C status: official Zenodo download starts but is slow on the current network path; 60-second probe downloaded about 21 MB. Use overnight `wget -c` or a verified mirror before running vision experiments.
- Runner update: `df80fbb` exposes `--epsilon-max`, `--epsilon-gamma`, and `--label-smoothing` overrides for calibration grid pilots.

## Iteration 3: Strong-Model Pilot Expansion

- Motivation: the CNN2 one-epoch result is useful as a smoke test and an early signal only; it is not sufficient evidence for a TPAMI-level claim.
- Main AMC models are moved to `MCLDNN`, `CGDNet`, `PETCGDNN`, and `MCformer`, covering CNN-RNN, CNN-GRU, phase-enhanced domain-specific, and attention/Transformer-style families already maintained in this repository.
- `FastMLDNN` is postponed because it uses `FastMLDNNHead`; the current RCPS implementation is integrated through `ClsHead`.
- Added RCPS config families for `CGDNet` and `PETCGDNN`, matching the existing `Hard CE`, `Static LS`, `RCPS-Uniform`, and `RCPS-Confusion` method interface.
- Added calibration utilities:
  - `tools/rcps/run_calibration_grid.py` for epsilon and label-smoothing grid runs.
  - `tools/rcps/collect_predictions.py` for validation/test prediction export with SNR metadata.
  - `tools/rcps/select_calibration.py` for validation NLL/ECE selection with a high-SNR retention constraint.
- Current interpretation policy: report RCPS as a posterior calibration and uncertainty-alignment method unless full multi-seed experiments also show stable accuracy gains.

## Iteration 4: Strong-Model One-Epoch Smoke

- Commit range: `97db84c` added strong-model calibration pilots; `33eee78` removed the unregistered CGDNet `RNN` initializer from the RCPS base config.
- Command pattern: `python tools/rcps/run_calibration_grid.py --models <model> --methods hard-ce rcps-uniform --seeds 2026 --max-epochs 1 --epsilon-max 0.5 --epsilon-gamma 2.0 --num-workers 0 --collect-splits validation --analyze --execute`.
- Completed validation export and SNR-bin analysis for `MCLDNN`, `CGDNet`, `PETCGDNN`, and `MCformer`.
- Smoke result: all four strong-model families train and export validation predictions with SNR metadata through the RCPS path.
- Early metric signal:
  - `CGDNet`: RCPS improved all-bin accuracy by +4.19 points and all-bin NLL by -0.0329; at -20 dB, NLL/ECE/Brier improved.
  - `MCformer`: RCPS improved all-bin accuracy by +3.03 points and all-bin NLL by -0.0754; at -20 dB, NLL/ECE/Brier improved.
  - `MCLDNN`: after one epoch, both methods remained near uniform prediction; only tiny NLL/ECE changes are meaningful as a smoke test.
  - `PETCGDNN`: RCPS reduced ECE but strongly hurt accuracy/NLL after one epoch, indicating method sensitivity and the need for calibration/longer training.
- Interpretation: this round supports the existence of low-reliability calibration mismatch and shows that RCPS can improve uncertainty metrics on stronger models, but it is still not proof of the final method. The next iteration must run a calibration grid and reject settings that violate high-SNR accuracy retention.

## Iteration 5: Ten-Epoch Calibration Queue

- Launch time: 2026-05-07 23:18 CST.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/calibration_10ep`.
- GPU0 queue: `CGDNet` and `MCformer`; GPU1 queue: `MCLDNN` and `PETCGDNN`.
- Methods: `Hard CE`, `Static LS`, and `RCPS-Uniform`; seed `2026`; max epochs `10`; `num_workers=0`.
- Static LS grid: smoothing in `{0.05, 0.1, 0.2, 0.3}`.
- RCPS grid: `epsilon_max` in `{0.3, 0.5, 0.7, 1.0}` and `gamma` in `{0.5, 1.0, 2.0}`.
- Watchdog: `/home/citybuster/Data/RCPS/work_dirs/logs/rcps_calibration_watchdog.sh`.
- Watchdog behavior: wait for 68 validation CSVs, run calibration selection, build prior/confusion bases, then launch the `RCPS-Confusion` 10-epoch grid under `/home/citybuster/Data/RCPS/work_dirs/calibration_10ep_confusion`.
- Status at launch verification: both GPUs training, first hard-label validation CSVs created, no conclusions drawn yet.

## Iteration 6: Conservative Monitoring Policy

- Monitoring policy: conservative automatic adjustment.
- Automatic actions allowed: record status snapshots, restart only clearly interrupted identical runs, and stop/report queues if logs contain hard failures such as tracebacks, conda errors, or CUDA OOM.
- Actions requiring confirmation: changing epochs, grid ranges, model set, dataset set, selection rule, or paper claims.
- Added monitor entry points:
  - `tools/rcps/monitor_experiments.py` writes one factual snapshot to JSONL and Markdown logs under `/home/citybuster/Data/RCPS/work_dirs/logs`.
  - `tools/rcps/monitor_loop.sh` runs the snapshotter repeatedly, defaulting to one snapshot every 600 seconds.
- Snapshot status at policy adoption: first-stage calibration still running, 7 validation CSVs observed, error scan clean, no conclusions drawn yet.

## Iteration 7: Night Monitoring Handoff

- Handoff time: 2026-05-07 23:40 CST.
- Active monitor: `tools/rcps/monitor_loop.sh`, appending status snapshots every 600 seconds.
- Active watchdog: `/home/citybuster/Data/RCPS/work_dirs/logs/rcps_calibration_watchdog.sh`, waiting for the first-stage calibration grid to reach 68 validation CSVs before launching the confusion grid.
- Current first-stage status: 10/68 validation CSVs; confusion stage 0/48.
- Current running candidates: `CGDNet` and `MCLDNN` have entered the `RCPS-Uniform` grid after completing hard-label and static label-smoothing candidates.
- Error scan at handoff: clean for tracebacks, conda errors, and CUDA OOM.
- Night policy: do not change epochs, grids, datasets, or model set overnight; only record progress and let watchdog advance the predefined queue.

## Iteration 8: Ten-Epoch Calibration Completed

- Completion time: 2026-05-08 03:00 CST.
- First-stage static/uniform grid completed with 68 validation CSVs and produced `selected_static_uniform.csv`.
- Confusion grid completed with 48 validation CSVs and produced `selected_confusion.csv`.
- Prior base and confusion base were generated under `/home/citybuster/Data/RCPS/work_dirs`.
- Teacher for confusion base: `MCformer hard-ce` validation predictions.
- Error scan: clean for tracebacks, conda errors, and CUDA OOM.
- Selection caveat: `selected_static_uniform.csv` selects the best among Static LS and RCPS-Uniform jointly. For the main comparison, separate best Static LS, best RCPS-Uniform, and best RCPS-Confusion configurations will be selected per model.
- Next stage: launch a 3-seed, 10-epoch stability main run using the selected per-family configurations, collecting both validation and test reliability metrics.

## Iteration 9: Selected Main-Run Configuration

- Main-run selection table: `/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed/selected_main_configs.csv`.
- Selection rule: per model and per family (`Hard CE`, `Static LS`, `RCPS-Uniform`, `RCPS-Confusion`), choose validation NLL first, ECE second, with high-SNR retention enforced where applicable.
- Selected `RCPS-Uniform`: `CGDNet eps=0.5 gamma=2.0`, `MCformer eps=0.3 gamma=2.0`, `MCLDNN eps=1.0 gamma=0.5`, `PETCGDNN eps=0.3 gamma=2.0`.
- Selected `RCPS-Confusion`: `CGDNet eps=0.5 gamma=2.0`, `MCformer eps=0.5 gamma=2.0`, `MCLDNN eps=1.0 gamma=0.5`, `PETCGDNN eps=0.3 gamma=0.5`.
- Selected `Static LS`: `CGDNet 0.05`, `MCformer 0.05`, `MCLDNN 0.3`, `PETCGDNN 0.05`.
- Main-run scope: `RadioML.2016.10A`, seeds `{2026, 2027, 2028}`, 10 epochs, validation and test reliability metrics.
- Interpretation caveat: this is a 3-seed stability run under the 10-epoch calibration budget, not the final extended-budget experiment.

## Iteration 10: Three-Seed Main Run Launch

- Launch time: 2026-05-08 09:45 CST.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed`.
- GPU0 queue: `CGDNet` and `MCformer`; GPU1 queue: `MCLDNN` and `PETCGDNN`.
- Families: `Hard CE`, selected `Static LS`, selected `RCPS-Uniform`, and selected `RCPS-Confusion`.
- Seeds: `{2026, 2027, 2028}`.
- Training budget: 10 epochs, matching the calibration budget; `num_workers=0`.
- Outputs: validation and test predictions plus reliability metrics under `main_10ep_3seed/metrics/main`.
- Conservative policy: do not alter model set, seeds, epochs, or selected hyperparameters during this run unless a hard failure occurs.


## Iteration 11: Baseline-First Reset

- Trigger: MCLDNN 10-epoch hard CE stayed near random while the repository already contains an `MLDNN` historical checkpoint around 62.97% on RadioML2016.10A.
- Action: stopped the current `main_10ep_3seed` queue and marked `/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed/DIAGNOSTIC_INVALID_BASELINE.md`; the generated 42 CSVs are diagnostic-invalid and must not enter the paper.
- Baseline gate: added `docs/rcps/baseline_reference_registry.csv` and `tools/rcps/check_baseline_gate.py`.
- Parity audit: added `tools/rcps/audit_framework_parity.py` to explicitly track Keras/PyTorch differences in initialization, padding, channel layout, optimizer settings, split, and epoch budget.
- Implementation update: added `RCPS-Retention`, MLDNN/FastMLDNN RCPS configs, and data-sample propagation for `MLDNNHead` and `FastMLDNNHead`.
- Config fix: removed duplicate `randomness` from `configs/mldnn/schedules.py`; seeds are now supplied by runtime configs or per-run cfg-options.
- Next action: reproduce `MLDNN + RadioML2016.10A` under the original 400-epoch schedule, then proceed model-by-model through the baseline registry.
- Smoke validation: `MLDNN hard-ce` and `MLDNN RCPS-Retention` both completed 1 epoch on RadioML2016.10A; `FastMLDNN RCPS-Retention` also completed 1 epoch. These are path checks only, not paper evidence.
- Verification: all registered RCPS configs parse through MMEngine, `git diff --check` passes with CRLF-aware whitespace settings, manual RCPS endpoint/retention checks pass; pytest is not installed in the server environment.


## Iteration 12: MLDNN Baseline Gate Launch

- Launch time: 2026-05-08 10:56 CST.
- Scope: `MLDNN + RadioML2016.10A`, hard CE only, original 400-epoch schedule and original `train_and_validation`/`test` split.
- Seeds launched: `2026` on GPU0 and `2027` on GPU1.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate/amc/deepsig201610A/mldnn_hard-ce`.
- Logs: `/home/citybuster/Data/RCPS/work_dirs/logs/baseline_gate_mldnn_seed2026_gpu0.log` and `baseline_gate_mldnn_seed2027_gpu1.log`.
- Post-train wrapper: collect test predictions and write reliability metrics to `/home/citybuster/Data/RCPS/work_dirs/baseline_gate/metrics/baseline_gate`.
- Gate criterion: overall test accuracy should reach at least `0.61`; target reference is the existing CSRR checkpoint around `0.6297`.

## Iteration 13: MLDNN Baseline Gate Recovery

- Recovery time: 2026-05-08 19:40 CST.
- Issue: the two hard-label MLDNN runs completed training, but standalone prediction export failed with `OSError: [Errno 24] Too many open files` because the exporter did not mirror `tools/train.py` dataloader defaults.
- Fix: `tools/rcps/collect_predictions.py` now applies the same `default_collate` dataloader defaults as `tools/train.py`, while still allowing explicit `num_workers=0` for stable export.
- Recovered checkpoints: seed `2026` uses `best_accuracy_top1_epoch_268.pth`; seed `2027` uses `best_accuracy_top1_epoch_274.pth`.
- Test results: seed `2026` accuracy `62.74%`; seed `2027` accuracy `62.98%`; two-seed mean `62.8608%`.
- Gate status: `PASS` against the `61.0%` MLDNN threshold in `docs/rcps/baseline_gate_report.csv`.
- Interpretation: the first AMC baseline gate is now stable enough to proceed to seed `2028`; no RCPS or paper-theory claim is updated from this gate alone.

## Iteration 14: MLDNN Third Baseline Seed Launch

- Launch time: 2026-05-08 19:44 CST.
- Scope: `MLDNN + RadioML2016.10A`, hard CE, seed `2028`, original 400-epoch schedule.
- Code commit used for launch: `42f788a`, which includes the standalone export collate fix.
- GPU/log: GPU1, `/home/citybuster/Data/RCPS/work_dirs/logs/baseline_gate_mldnn_seed2028_gpu1.log`.
- Work dir: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate/amc/deepsig201610A/mldnn_hard-ce/seed_2028`.
- Post-train action: collect test predictions with `num_workers=0` and write `/home/citybuster/Data/RCPS/work_dirs/baseline_gate/metrics/baseline_gate/deepsig201610A_mldnn_hard-ce_seed2028_test.csv`.

## Iteration 15: MLDNN Baseline Gate Completed

- Completion time: 2026-05-09 01:08 CST.
- Scope: `MLDNN + RadioML2016.10A`, hard CE, original 400-epoch schedule, seeds `2026/2027/2028`.
- Seed `2028`: training early-stopped after epoch `370`; test export used `best_accuracy_top1_epoch_320.pth`; test accuracy `63.17%`.
- Three-seed mean CSV: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate/metrics/baseline_gate/deepsig201610A_mldnn_hard-ce_seed2026_2027_2028_test_mean.csv`.
- Three-seed mean test accuracy: `62.9625%`.
- Baseline gate: `PASS` against the `61.0%` threshold; margin `1.9625` percentage points.
- Error scan: clean for `Traceback`, `Too many open files`, CUDA OOM, `CalledProcessError`, `FileNotFoundError`, and `TypeError` during the seed `2028` foreground monitoring chain.
- Interpretation: the first AMC baseline gate is stable and close to the repository historical reference around `62.97%`. The old 10-epoch MCLDNN/RCPS results remain diagnostic-invalid and are not paper evidence.
- Next action: design RCPS/Static LS comparisons on top of this stable MLDNN baseline, using identical data, schedule, optimizer, and backbone, and only replacing the supervision/loss.

## Iteration 16: MLDNN Supervision Comparison Launch

- Launch time: 2026-05-09 14:15 CST.
- Purpose: compare supervision strategies on top of the validated `MLDNN + RadioML2016.10A` baseline, without changing backbone, data split, optimizer, schedule, or export/analyze pipeline.
- Baseline anchor: hard CE three-seed mean test accuracy `62.9625%` from Iteration 15.
- Queue A: `Static LS`, smoothing `0.1`, seeds `2026/2027/2028`, GPU0, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_static-ls_400ep_gpu0.log`.
- Queue B: `RCPS-Retention`, uniform base, SNR reliability map `[-20, 18]`, epsilon `retention_power(max=0.7, gamma=1.0, retain_min=0.8)`, seeds `2026/2027/2028`, GPU1, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-retention_400ep_gpu1.log`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep`.
- Metrics root: `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/metrics/mldnn_supervision_400ep`.
- Initial check: both seed `2026` runs entered training and completed epoch 1 validation; no traceback, OOM, or file-handle errors.
- Interpretation policy: do not compare against hard CE until each method has complete three-seed test CSVs; if gains are calibration-only, keep the paper claim focused on posterior calibration and uncertainty alignment.

### Iteration 17 - MLDNN Probability Export Fix During Supervision Comparison
- Time: 2026-05-09 19:35 CST
- Stage: `mldnn_supervision_400ep`, `RadioML.2016.10A`, `MLDNN`, seed 2026 completed for `static-ls` and `rcps-retention`; seed 2027 running for both queues.
- Finding: exported `pred_score` for `MLDNN + MLDNNHead` was a second softmax over the MLDNN backbone's already-probabilistic `merge` output. Accuracy was effectively unchanged, but confidence, NLL, ECE, and Brier were systematically distorted toward under-confidence.
- Fix: `tools/rcps/collect_predictions.py` now recovers the original MLDNN merge probability during export for `MLDNN + MLDNNHead`. Completed seed-2026 prediction PKLs were backed up with `.double_softmax.pkl`, corrected, and re-analyzed.
- Corrected seed-2026 overall metrics: Static LS acc 63.0625, NLL 1.0596, ECE 0.0250, Brier 0.4418; RCPS-Retention acc 62.8898, NLL 1.1240, ECE 0.0686, Brier 0.4540.
- Corrected seed-2026 stratified observation: RCPS-Retention improves the extreme low-SNR `-20 dB` NLL/ECE/Brier and the high-SNR `18 dB` NLL/ECE relative to Static LS, but loses on overall calibration and some transition SNR bins. Current conclusion is diagnostic only; no paper claim changes until three seeds are complete.


## Iteration 18: MLDNN Supervision Comparison Completed and Probability Metrics Reconciled

- Completion time: 2026-05-10 14:51 CST.
- Scope: `MLDNN + RadioML2016.10A`, original 400-epoch schedule, seeds `2026/2027/2028`.
- Compared methods: hard CE baseline, Static Label Smoothing (`smoothing=0.1`), and `RCPS-Retention` with uniform base and SNR reliability map `[-20, 18]`.
- Export integrity fix: hard CE baseline predictions were re-exported using commit `67d0b07`, because the earlier baseline CSVs still used the stale MLDNN double-softmax probability path. Accuracy was unchanged, but NLL/ECE/Brier/confidence are now comparable across all three methods.
- Completion status: Static LS and RCPS-Retention both completed all three seeds. Logs contain `DONE` markers for every seed and the error scan is clean for `Traceback`, `Too many open files`, CUDA OOM, `CalledProcessError`, `FileNotFoundError`, and `TypeError`.
- Summary files:
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/summary/deepsig201610A_mldnn_supervision_seed2026_2027_2028_mean_std.csv`
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/summary/deepsig201610A_mldnn_supervision_overall_mean_std.csv`
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/summary/deepsig201610A_mldnn_supervision_deltas.csv`
- Three-seed overall metrics after probability reconciliation:
  - hard CE: acc `62.9625 +/- 0.2137`, NLL `1.0562 +/- 0.0046`, ECE `0.0325 +/- 0.0001`, Brier `0.4444 +/- 0.0021`.
  - Static LS: acc `63.1098 +/- 0.0625`, NLL `1.0614 +/- 0.0049`, ECE `0.0267 +/- 0.0045`, Brier `0.4424 +/- 0.0013`.
  - RCPS-Retention: acc `63.0242 +/- 0.1173`, NLL `1.1211 +/- 0.0044`, ECE `0.0680 +/- 0.0016`, Brier `0.4532 +/- 0.0016`.
- Stratified finding: RCPS-Retention improves the extreme low-SNR `-20 dB` bin over Static LS on NLL (`-0.0679`), ECE (`-0.0349`), and Brier (`-0.0179`), and also improves the `18 dB` bin on NLL/ECE. However, it is substantially worse in transition bins such as `-8 dB` and `0 dB`, which dominates the overall result.
- Interpretation: the current monotone retention-style RCPS target is not a paper-ready improvement. The stable signal is narrower: reliability-aware supervision helps at extreme low reliability, but the finite-reliability target schedule must be learned or calibrated from validation posterior/confusion rather than imposed as a simple monotone smoothing curve.
- Next action: pause broad scaling of this RCPS variant. Design the next RCPS iteration around validation-calibrated posterior targets, especially a transition-region-aware epsilon/base schedule, then test it against hard CE and Static LS on the same validated MLDNN gate before expanding to more AMC models or cross-modal datasets.


## Iteration 19: Validation-Calibrated RCPS Diagnostic Launch

- Launch time: 2026-05-10 15:17 CST.
- Motivation: Iteration 18 showed that monotone RCPS-Retention improves the extreme low-SNR bin but over-softens transition SNR bins. The next diagnostic tests posterior-calibrated targets that use validation-set teacher posteriors or restrict smoothing to genuinely low reliability.
- Code commit: `3fde339`.
- Teacher base: hard CE MLDNN validation predictions from seeds `2026/2027/2028` were re-exported with corrected MLDNN probabilities and aggregated into `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/posterior_bases/deepsig201610A_mldnn_hardce_validation_meanprob.npz`.
- New method A: `RCPS-BinPosterior`, where `b_y(r)` is a reliability-bin conditional mean teacher posterior and `epsilon=1.0`.
- New method B: `RCPS-LowGate`, where uniform smoothing is active only below the mapped reliability cutoff corresponding approximately to `-10 dB`.
- Smoke tests: direct RCPS target construction smoke passed; both configs parsed; both methods completed a 1-epoch train/validation smoke without errors.
- Running diagnostics: seed `2026` only, full 400-epoch schedule, same MLDNN backbone/data/optimizer/split as the baseline gate.
  - GPU0: `RCPS-BinPosterior`, launcher PID `2481399`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-binposterior_iter2_seed2026_gpu0.log`.
  - GPU1: `RCPS-LowGate`, launcher PID `2481400`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-lowgate_iter2_seed2026_gpu1.log`.
- Decision rule: do not expand to three seeds unless the seed-2026 test metrics beat or clearly complement hard CE / Static LS in low-SNR calibration without harming transition bins.

## Iteration 20: Validation-Calibrated RCPS Diagnostic Completed

- Completion time: 2026-05-10 22:35 CST.
- Scope: `MLDNN + RadioML2016.10A`, seed `2026`, original 400-epoch schedule, same data split/backbone/optimizer as the validated hard CE baseline.
- Compared against same-seed anchors: hard CE, Static Label Smoothing (`smoothing=0.1`), and the previous `RCPS-Retention` variant.
- Completed diagnostics:
  - `RCPS-LowGate`: uniform smoothing only below the mapped reliability cutoff around `-10 dB`; best validation checkpoint `best_accuracy_top1_epoch_257.pth`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_posterior_iter2_400ep/metrics/mldnn_posterior_iter2_400ep/deepsig201610A_mldnn_rcps-lowgate_seed2026_test.csv`.
  - `RCPS-BinPosterior`: reliability-bin conditional teacher posterior base with `epsilon=1.0`; best validation checkpoint `best_accuracy_top1_epoch_296.pth`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_posterior_iter2_400ep/metrics/mldnn_posterior_iter2_400ep/deepsig201610A_mldnn_rcps-binposterior_seed2026_test.csv`.
- Compact comparison CSV: `/home/citybuster/Data/RCPS/work_dirs/mldnn_posterior_iter2_400ep/metrics/mldnn_posterior_iter2_400ep/deepsig201610A_mldnn_seed2026_supervision_comparison.csv`.
- Seed-2026 overall metrics:
  - hard CE: acc `62.7398`, NLL `1.0567`, ECE `0.0325`, Brier `0.4465`.
  - Static LS: acc `63.0625`, NLL `1.0596`, ECE `0.0250`, Brier `0.4418`.
  - RCPS-Retention: acc `62.8898`, NLL `1.1240`, ECE `0.0686`, Brier `0.4540`.
  - RCPS-LowGate: acc `62.9432`, NLL `1.0518`, ECE `0.0278`, Brier `0.4435`.
  - RCPS-BinPosterior: acc `62.5477`, NLL `1.0547`, ECE `0.0211`, Brier `0.4473`.
- Interpretation:
  - `RCPS-LowGate` is the first RCPS variant that improves all four same-seed overall metrics relative to hard CE while preserving high-SNR accuracy; however, its low-SNR improvements are mixed and it does not beat Static LS on overall accuracy/ECE/Brier in this seed.
  - `RCPS-BinPosterior` gives the lowest overall ECE, but its lower accuracy and worse Brier indicate that directly distilling reliability-bin teacher posteriors sacrifices discriminative information. It should not be expanded as the main algorithm in its current form.
  - The finite-reliability theory should not claim that any monotone or posterior-table target is automatically beneficial. The supported direction is more constrained: selective posterior relaxation can help calibration and preserve high-reliability behavior, but transition-bin behavior must be guarded by validation-calibrated retention constraints.
- Next action: do not scale `RCPS-BinPosterior`. Run `RCPS-LowGate` seeds `2027/2028` only as a candidate sanity check, then decide whether to tune the low-reliability cutoff/epsilon schedule or redesign the base allocation before moving to more AMC backbones.

## Iteration 21: RCPS-LowGate Multi-Seed Diagnostic Launch

- Launch time: 2026-05-10 22:40 CST.
- Scope: `MLDNN + RadioML2016.10A`, `RCPS-LowGate`, seeds `2027/2028`, original 400-epoch schedule.
- Launcher commit: `8fb73ad`.
- Rationale: Iteration 20 showed that `RCPS-LowGate` is the first RCPS variant to improve same-seed hard CE on overall accuracy, NLL, ECE, and Brier while preserving high-SNR accuracy. It remains weaker than Static LS on several overall and low-SNR metrics, so this is a sanity-check expansion rather than a main-result launch.
- Fixed configuration: same `configs/rcps/mldnn/mldnn_rcps-lowgate_iq-ap-snr-deepsig-201610A.py`; no change to cutoff, epsilon, data split, backbone, optimizer, or export/analyze code.
- Queue:
  - GPU0: seed `2027`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-lowgate_iter2_seed2027_gpu0.log`.
  - GPU1: seed `2028`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-lowgate_iter2_seed2028_gpu1.log`.
- Decision rule: after both seeds finish, compare three-seed LowGate against hard CE and Static LS. If gains remain small or mainly calibration-only, the paper framing stays focused on posterior calibration/uncertainty alignment and the next algorithm iteration should tune low-reliability cutoff/epsilon or redesign base allocation before expanding to more models.

## Iteration 22: RCPS-LowGate Multi-Seed Diagnostic Completed

- Completion time: 2026-05-11 04:20 CST.
- Scope: `MLDNN + RadioML2016.10A`, `RCPS-LowGate`, seeds `2026/2027/2028`, same 400-epoch training/export/analyze pipeline as the hard CE baseline gate.
- Completion status: seeds `2027/2028` completed after the seed-2026 diagnostic; all three test CSVs are present and the error scan is clean for traceback, CUDA OOM, file-handle errors, and subprocess failures.
- Summary files:
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_posterior_iter2_400ep/summary/deepsig201610A_mldnn_lowgate_vs_baselines_seed2026_2027_2028_mean_std.csv`.
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_posterior_iter2_400ep/summary/deepsig201610A_mldnn_lowgate_vs_baselines_deltas.csv`.
- Three-seed overall metrics:
  - hard CE: acc `62.9625 +/- 0.2137`, NLL `1.0562 +/- 0.0046`, ECE `0.0325 +/- 0.0001`, Brier `0.4444 +/- 0.0021`.
  - Static LS: acc `63.1098 +/- 0.0625`, NLL `1.0614 +/- 0.0049`, ECE `0.0267 +/- 0.0045`, Brier `0.4424 +/- 0.0013`.
  - RCPS-LowGate: acc `63.0792 +/- 0.1876`, NLL `1.0496 +/- 0.0030`, ECE `0.0255 +/- 0.0045`, Brier `0.4419 +/- 0.0017`.
- Mean deltas:
  - LowGate minus hard CE: accuracy `+0.1167`, NLL `-0.0066`, ECE `-0.0069`, Brier `-0.0025`.
  - LowGate minus Static LS: accuracy `-0.0307`, NLL `-0.0118`, ECE `-0.0012`, Brier `-0.0005`.
- Reliability-bin finding relative to hard CE:
  - `-20 dB`: accuracy `+0.1212`, NLL `-0.0351`, ECE `-0.0131`, Brier `-0.0077`.
  - `-12 dB`: accuracy `-1.3636`, NLL `+0.0225`, ECE `+0.0023`, Brier `+0.0049`.
  - `-10 dB`: accuracy `-0.5076`, NLL `+0.0098`, ECE `-0.0081`, Brier `-0.0010`.
  - High-SNR bins are retained: at `10 dB`, accuracy `+0.1212` and NLL `-0.0013`; at `18 dB`, accuracy `+0.0682` and NLL `-0.0011`.
- Interpretation: `RCPS-LowGate` is a valid candidate improvement over hard CE and is at least competitive with Static LS on calibration/probabilistic metrics, but it is not yet a decisive TPAMI-level algorithmic result. The central evidence supports a narrower claim: reliability-conditioned supervision can improve posterior calibration and uncertainty alignment while preserving high-reliability accuracy. The remaining weakness is transition-bin handling around `-12/-10 dB`.
- Next action: do not expand to more AMC backbones yet. Run a focused validation-calibrated LowGate tuning pass over cutoff/epsilon strength to reduce transition-bin harm, then repeat only the best candidate before scaling.

## Iteration 23: Conservative LowGate Tuning Launch

- Launch time: 2026-05-11 04:23 CST.
- Code commit: `1decad1`.
- Scope: `MLDNN + RadioML2016.10A`, seed `2026`, original 400-epoch schedule, same hard CE baseline gate pipeline.
- Motivation: Iteration 22 showed overall LowGate gains over hard CE but transition-bin weakness around `-12/-10 dB`. This tuning pass tests two conservative schedules designed to reduce transition-bin smoothing while preserving the extreme low-SNR benefit.
- Candidate A: `RCPS-LowGate-C14`, cutoff mapped to approximately `-14 dB`, `epsilon_max=0.7`, `gamma=1.0`; config `configs/rcps/mldnn/mldnn_rcps-lowgate-c14_iq-ap-snr-deepsig-201610A.py`; GPU0 log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-lowgate-c14_tuning_seed2026_gpu0.log`.
- Candidate B: `RCPS-LowGate-G2`, cutoff remains approximately `-10 dB` but `gamma=2.0`; config `configs/rcps/mldnn/mldnn_rcps-lowgate-g2_iq-ap-snr-deepsig-201610A.py`; GPU1 log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-lowgate-g2_tuning_seed2026_gpu1.log`.
- Decision rule: select a candidate only if it improves the seed-2026 transition bins (`-12/-10 dB`) relative to the original LowGate without losing the overall Hard CE gains. Do not expand either candidate before seed-2026 test CSVs are available.

## Iteration 24: Conservative LowGate Tuning Completed

- Completion time: 2026-05-11 08:59 CST.
- Scope: `MLDNN + RadioML2016.10A`, seed `2026`, same 400-epoch schedule/export/analyze pipeline as the validated baseline gate.
- Completed candidates:
  - `RCPS-LowGate-C14`: cutoff mapped to approximately `-14 dB`, `epsilon_max=0.7`, `gamma=1.0`; early-stopped at epoch `295`; best validation checkpoint `best_accuracy_top1_epoch_245.pth`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_tuning_400ep/metrics/mldnn_lowgate_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14_seed2026_test.csv`.
  - `RCPS-LowGate-G2`: cutoff approximately `-10 dB`, `epsilon_max=0.7`, `gamma=2.0`; early-stopped at epoch `306`; best validation checkpoint `best_accuracy_top1_epoch_256.pth`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_tuning_400ep/metrics/mldnn_lowgate_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-g2_seed2026_test.csv`.
- Summary files:
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_tuning_400ep/summary/deepsig201610A_mldnn_lowgate_tuning_seed2026_key_metrics.csv`.
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_tuning_400ep/summary/deepsig201610A_mldnn_lowgate_tuning_seed2026_deltas.csv`.
- Seed-2026 overall metrics:
  - hard CE: acc `62.7398`, NLL `1.0567`, ECE `0.0325`, Brier `0.4465`.
  - Static LS: acc `63.0625`, NLL `1.0596`, ECE `0.0250`, Brier `0.4418`.
  - Original RCPS-LowGate: acc `62.9432`, NLL `1.0518`, ECE `0.0278`, Brier `0.4435`.
  - RCPS-LowGate-C14: acc `63.0352`, NLL `1.0543`, ECE `0.0283`, Brier `0.4443`.
  - RCPS-LowGate-G2: acc `63.0670`, NLL `1.0520`, ECE `0.0328`, Brier `0.4436`.
- Transition-bin result relative to hard CE:
  - `C14` improves accuracy at `-20/-12/-10 dB` by `+0.4545/+0.5000/+0.3409` pp and improves ECE by `-0.0168/-0.0103/-0.0034`, but NLL/Brier remain worse at `-12/-10 dB`.
  - `G2` improves overall accuracy and NLL/Brier but does not fix transition bins: at `-12/-10 dB`, accuracy changes are `-0.2955/-1.1591` pp and NLL increases by `+0.0325/+0.0412`.
- Interpretation:
  - `C14` is the only tuning candidate that addresses the Iteration 22 transition-bin accuracy/ECE weakness relative to hard CE and improves substantially over the original LowGate at `-12/-10 dB`; it is the better RCPS candidate if we continue this branch.
  - Neither tuning candidate is a decisive TPAMI-level result yet. Static LS still remains very competitive and often stronger in low/transition SNR bins, while RCPS shows its clearest benefit as selective posterior calibration with high-reliability retention.
  - The theory should remain conservative: finite-reliability target allocation is a modeling choice, and the useful empirical principle is not generic smoothing but reliability-gated posterior relaxation with validation constraints.
- Next action: before expanding to more AMC backbones, test one improved target that uses `C14`-style conservative activation but replaces the uniform base in transition/noisy bins with a validation-estimated class-overlap base or adds an explicit transition-retention constraint. The goal is to preserve C14's accuracy/ECE transition repair while reducing NLL/Brier harm.

## Iteration 25: C14 Posterior-Base LowGate Launch

- Launch time: 2026-05-11 09:02 CST.
- Code basis: follows `RCPS-LowGate-C14` from Iteration 24 but replaces the uniform base with the validation hard-CE reliability-bin posterior base `/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/posterior_bases/deepsig201610A_mldnn_hardce_validation_meanprob.npz`.
- Motivation: Iteration 24 showed that conservative C14 gating repairs transition-bin accuracy/ECE relative to hard CE, but NLL/Brier remain worse at `-12/-10 dB`. The hypothesis is that low-reliability mass should be allocated to empirically confused classes rather than uniformly to every class.
- Candidate A: `RCPS-LowGate-C14-Posterior-E0p7`, cutoff approximately `-14 dB`, `epsilon_max=0.7`, `gamma=1.0`, reliability-bin posterior base; config `configs/rcps/mldnn/mldnn_rcps-lowgate-c14-posterior-e0p7_iq-ap-snr-deepsig-201610A.py`.
- Candidate B: `RCPS-LowGate-C14-Posterior-E0p5`, same base and cutoff but `epsilon_max=0.5`; config `configs/rcps/mldnn/mldnn_rcps-lowgate-c14-posterior-e0p5_iq-ap-snr-deepsig-201610A.py`.
- Decision rule: expand only if a candidate preserves the C14 accuracy/ECE transition repair while reducing NLL/Brier harm and remaining competitive with Static LS on seed `2026`.

- Runtime:
  - GPU0: `RCPS-LowGate-C14-Posterior-E0p7`, launcher PID `2668658`, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-e0p7_seed2026_gpu0.log`.
  - GPU1: the first `E0p5` launch failed because a Windows carriage return was appended to the config path in the shell wrapper; this did not affect code or data. It was recovered once with the same config and seed. Retry launcher PID `2670245`, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-e0p5_seed2026_gpu1_retry1.log`.

## Iteration 25: C14 Posterior-Base LowGate Completed

- Completion time: 2026-05-11 13:54 CST.
- Scope: `MLDNN + RadioML2016.10A`, seed `2026`, same 400-epoch schedule/export/analyze pipeline as the validated baseline gate.
- Completed candidates:
  - `RCPS-LowGate-C14-Posterior-E0p5`: early-stopped at epoch `307`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_tuning_400ep/metrics/mldnn_lowgate_posterior_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-e0p5_seed2026_test.csv`.
  - `RCPS-LowGate-C14-Posterior-E0p7`: early-stopped at epoch `295`; test CSV `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_tuning_400ep/metrics/mldnn_lowgate_posterior_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-e0p7_seed2026_test.csv`.
- Summary files:
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_tuning_400ep/summary/deepsig201610A_mldnn_posterior_tuning_seed2026_overall.csv`.
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_tuning_400ep/summary/deepsig201610A_mldnn_posterior_tuning_seed2026_delta_vs_hard.csv`.
  - `/home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_tuning_400ep/summary/deepsig201610A_mldnn_posterior_tuning_seed2026_candidate_criteria.csv`.
- Seed-2026 overall metrics:
  - hard CE: acc `62.7398`, NLL `1.0567`, ECE `0.0325`, Brier `0.4465`.
  - Static LS: acc `63.0625`, NLL `1.0596`, ECE `0.0250`, Brier `0.4418`.
  - C14 uniform: acc `63.0352`, NLL `1.0543`, ECE `0.0283`, Brier `0.4443`.
  - C14 posterior E0p5: acc `63.0875`, NLL `1.0479`, ECE `0.0373`, Brier `0.4429`.
  - C14 posterior E0p7: acc `63.2102`, NLL `1.0453`, ECE `0.0437`, Brier `0.4423`.
- Diagnostic finding relative to hard CE:
  - Posterior allocation improves overall accuracy, NLL, and Brier more than the uniform C14 target. E0p7 gives the strongest same-seed overall accuracy and NLL/Brier gains: accuracy `+0.4705` pp, NLL `-0.0114`, Brier `-0.0042`.
  - The same posterior allocation increases confidence and worsens ECE: E0p5 ECE `+0.0048`, E0p7 ECE `+0.0112` relative to hard CE. This is not acceptable as a calibration-improving claim.
  - In the transition region `-12/-10 dB`, posterior E0p5 improves accuracy by `+1.0227` pp and reduces NLL/Brier by `-0.0102/-0.0047`, but does not preserve C14's ECE repair. Posterior E0p7 has stronger overall metrics but worse transition ECE and gate-bin `-14 dB` NLL/Brier.
- Interpretation:
  - The result supports the paper's posterior-allocation idea: reliability-conditioned mass should not be uniformly spread across all classes when empirical class overlap is structured.
  - It also reveals a finite-reliability modeling issue: using the hard-CE validation posterior base directly is too sharp and can convert calibration gains into accuracy/NLL gains with overconfidence.
  - Do not expand either posterior candidate to three seeds yet. The next algorithmic step should soften the posterior base, for example by temperature scaling or blending the posterior base with the uniform/prior base, and then retest only seed `2026` before scaling.

## Iteration 26: Soft Posterior-Base LowGate Launch

- Launch time: 2026-05-11 14:00 CST.
- Scope: `MLDNN + RadioML2016.10A`, seed `2026`, same 400-epoch schedule/export/analyze pipeline.
- Motivation: Iteration 25 showed that reliability-bin posterior allocation improves accuracy, NLL, and Brier but worsens ECE because the hard-CE posterior base is too sharp. This launch tests whether temperature-softening the posterior base preserves posterior mass allocation while reducing overconfidence.
- Candidate A: `RCPS-LowGate-C14-Posterior-T2-E0p7`, cutoff approximately `-14 dB`, `epsilon_max=0.7`, posterior base temperature `2.0`; config `configs/rcps/mldnn/mldnn_rcps-lowgate-c14-posterior-t2-e0p7_iq-ap-snr-deepsig-201610A.py`.
- Candidate B: `RCPS-LowGate-C14-Posterior-T2-E0p5`, same but `epsilon_max=0.5`; config `configs/rcps/mldnn/mldnn_rcps-lowgate-c14-posterior-t2-e0p5_iq-ap-snr-deepsig-201610A.py`.
- Decision rule: select only if the candidate keeps the posterior-base accuracy/NLL/Brier gains while materially reducing the ECE penalty relative to Iteration 25. If temperature alone fails, the next candidate should blend posterior with uniform/prior base rather than expanding seeds.
- Runtime: GPU0 `T2-E0p7` launcher PID `2723091`, child train PID `2723100`, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-e0p7_seed2026_gpu0.log`; GPU1 `T2-E0p5` launcher PID `2723092`, child train PID `2723099`, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-e0p5_seed2026_gpu1.log`.



## Iteration 26: Soft Posterior-Base LowGate Completed

- Completion time: 2026-05-11 19:28 CST.
- Scope: MLDNN + RadioML2016.10A, seed 2026, same validated 400-epoch train/export/analyze pipeline.
- Completed candidates:
  - RCPS-LowGate-C14-Posterior-T2-E0p5: test CSV /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_soft_tuning_400ep/metrics/mldnn_lowgate_posterior_soft_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-t2-e0p5_seed2026_test.csv.
  - RCPS-LowGate-C14-Posterior-T2-E0p7: test CSV /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_soft_tuning_400ep/metrics/mldnn_lowgate_posterior_soft_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-t2-e0p7_seed2026_test.csv.
- Summary files are under /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_soft_tuning_400ep/summary/.
- Seed-2026 overall metrics:
  - hard CE: acc 62.7398, NLL 1.0567, ECE 0.0325, Brier 0.4465.
  - Static LS: acc 63.0625, NLL 1.0596, ECE 0.0250, Brier 0.4418.
  - C14 uniform E0p7: acc 63.0352, NLL 1.0543, ECE 0.0283, Brier 0.4443.
  - C14 posterior T1 E0p5: acc 63.0875, NLL 1.0479, ECE 0.0373, Brier 0.4429.
  - C14 posterior T1 E0p7: acc 63.2102, NLL 1.0453, ECE 0.0437, Brier 0.4423.
  - C14 posterior T2 E0p5: acc 63.1477, NLL 1.0490, ECE 0.0389, Brier 0.4433.
  - C14 posterior T2 E0p7: acc 63.1193, NLL 1.0535, ECE 0.0409, Brier 0.4444.
- Diagnostic finding:
  - Temperature softening alone does not fix the overall ECE penalty of posterior-base RCPS. T2 E0p5 still worsens overall ECE by +0.0064 relative to hard CE.
  - T2 E0p5 is nevertheless informative: in the transition region -12/-10 dB, it improves accuracy by +1.9886 pp, NLL by -0.0264, ECE by -0.0072, and Brier by -0.0098; high-SNR accuracy is retained.
  - Region diagnostics show the remaining ECE problem comes from very-low and gate bins, where posterior base still increases confidence and lowers entropy. The empirical posterior base is still too sharp in the lowest-reliability regime.
- Decision:
  - Do not expand T2 posterior candidates to three seeds.
  - The next minimal algorithmic diagnostic is to blend the reliability-conditioned posterior base with uniform or prior mass at low reliability, while keeping the same C14 gate, epsilon_max 0.5, temperature 2.0, model, seed, and training schedule.

## Iteration 27: Prior-Blend Posterior-Base LowGate Plan

- Launch scope: MLDNN + RadioML2016.10A, seed 2026, same 400-epoch schedule/export/analyze pipeline.
- Motivation: Iteration 26 indicates that the posterior base is useful for transition NLL/Brier and accuracy, but remains overconfident in very-low reliability bins. Prior blending should move the finite-reliability base toward the low-information limit without discarding structured class-overlap information.
- Candidate A: RCPS-LowGate-C14-Posterior-T2-B0p5-E0p5, posterior base temperature 2.0, prior_blend 0.5, uniform prior.
- Candidate B: RCPS-LowGate-C14-Posterior-T2-B1p0-E0p5, posterior base temperature 2.0, prior_blend 1.0, uniform prior.
- Decision rule: a candidate must preserve the transition-region NLL/Brier and high-reliability retention while reducing the overall ECE penalty relative to the unblended posterior T2 candidate. If both fail, the theory should emphasize that structured posterior allocation helps likelihood and accuracy but requires separate calibration, rather than claiming calibration improvement from posterior bases.

## Iteration 27: Prior-Blend Posterior-Base LowGate Launch

- Launch time: 2026-05-11 19:40 CST.
- Code commit at launch: 0fcab70.
- Runtime: GPU0 candidate B0p5 launcher PID 2781599, child train PID 2781608, log /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b0p5-e0p5_seed2026_gpu0.log.
- Runtime: GPU1 candidate B1p0 launcher PID 2781600, child train PID 2781609, log /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b1p0-e0p5_seed2026_gpu1.log.
- Monitoring rule: no matrix expansion during this run. If one candidate fails from export or file-handle recovery issues, rerun export only from the same checkpoint. If training code fails, stop and diagnose before changing parameters.


## Iteration 27: Prior-Blend Posterior-Base LowGate Completed

- Completion time: 2026-05-12 01:35 CST.
- Scope: MLDNN + RadioML2016.10A, seed 2026, same 400-epoch train/export/analyze pipeline.
- Completed candidates:
  - RCPS-LowGate-C14-Posterior-T2-B0p5-E0p5 completed at 2026-05-11 23:36 CST.
  - RCPS-LowGate-C14-Posterior-T2-B1p0-E0p5 completed at 2026-05-12 01:35 CST.
- Summary files are under /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_tuning_400ep/summary/.
- Seed-2026 overall metrics:
  - hard CE: acc 62.7398, NLL 1.0567, ECE 0.0325, Brier 0.4465.
  - Static LS: acc 63.0625, NLL 1.0596, ECE 0.0250, Brier 0.4418.
  - C14 posterior T2 E0p5 without blend: acc 63.1477, NLL 1.0490, ECE 0.0389, Brier 0.4433.
  - C14 posterior T2 B0p5 E0p5: acc 62.7886, NLL 1.0570, ECE 0.0318, Brier 0.4469.
  - C14 posterior T2 B1p0 E0p5: acc 63.1705, NLL 1.0644, ECE 0.0402, Brier 0.4456.
- Diagnostic finding:
  - Prior blending partially controls overconfidence. B0p5 slightly improves overall ECE relative to hard CE, but loses the likelihood and Brier gains of posterior allocation.
  - B1p0 improves accuracy and very-low ECE, but worsens overall NLL and ECE.
  - No Iteration 27 candidate should be expanded to three seeds.
- Decision:
  - Run a narrower blend fine-tuning pass between no blend and B0p5.
  - Candidate blend strengths are 0.25 and 0.35 with temperature 2.0, epsilon_max 0.5, same C14 gate, same model/data/seed/schedule.

## Iteration 28: Fine Prior-Blend Posterior-Base Plan

- Scope: MLDNN + RadioML2016.10A, seed 2026, same 400-epoch schedule/export/analyze pipeline.
- Candidate A: RCPS-LowGate-C14-Posterior-T2-B0p25-E0p5, prior_blend 0.25.
- Candidate B: RCPS-LowGate-C14-Posterior-T2-B0p35-E0p5, prior_blend 0.35.
- Decision rule: select only if the candidate preserves part of the posterior-base NLL/Brier gain and reduces the ECE penalty versus the unblended T2 E0p5 candidate. Do not expand to three seeds unless it also keeps high-reliability accuracy retention.

## Iteration 28: Fine Prior-Blend Posterior-Base Launch

- Launch time: 2026-05-12 14:34 CST.
- Code commit at launch: adf24ce.
- Runtime: GPU0 candidate B0p25 launcher PID 2938388, child train PID 2938398, log /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b0p25-e0p5_seed2026_gpu0.log.
- Runtime: GPU1 candidate B0p35 launcher PID 2938389, child train PID 2938397, log /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b0p35-e0p5_seed2026_gpu1.log.
- Monitoring rule: keep the matrix fixed; no new model or dataset expansion until both test CSV files and the fine-blend summary exist.

## Iteration 28: Fine Prior-Blend Posterior-Base Completed

- Completion time: 2026-05-12 20:36 CST.
- Scope: MLDNN + RadioML2016.10A, seed 2026, same 400-epoch train/export/analyze pipeline.
- Completed candidates:
  - RCPS-LowGate-C14-Posterior-T2-B0p25-E0p5 completed at 2026-05-12 20:00 CST.
  - RCPS-LowGate-C14-Posterior-T2-B0p35-E0p5 completed at 2026-05-12 20:31 CST.
- Summary files are under /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_fine_tuning_400ep/summary/.
- Seed-2026 overall metrics:
  - hard CE: acc 62.7398, NLL 1.0567, ECE 0.0325, Brier 0.4465.
  - Static LS: acc 63.0625, NLL 1.0596, ECE 0.0250, Brier 0.4418.
  - C14 posterior T2 E0p5 without blend: acc 63.1477, NLL 1.0490, ECE 0.0389, Brier 0.4433.
  - C14 posterior T2 B0p25 E0p5: acc 63.2011, NLL 1.0507, ECE 0.0366, Brier 0.4420.
  - C14 posterior T2 B0p35 E0p5: acc 63.0977, NLL 1.0551, ECE 0.0355, Brier 0.4431.
  - C14 posterior T2 B0p5 E0p5: acc 62.7886, NLL 1.0570, ECE 0.0318, Brier 0.4469.
- Diagnostic finding:
  - B0p25 is the strongest fine-blend candidate on overall accuracy, NLL, and Brier among the fine blends. Relative to hard CE it improves accuracy by +0.4614 pp, NLL by -0.0060, and Brier by -0.0045.
  - B0p25 reduces the ECE penalty of the unblended posterior T2 candidate (+0.0040 vs +0.0064 relative to hard CE), but still does not beat hard CE or Static LS on overall ECE.
  - B0p35 further reduces overall ECE penalty (+0.0029 relative to hard CE) and improves very-low-reliability ECE, but gives up most of the posterior-base NLL gain and worsens transition-region NLL/Brier.
  - Both fine blends satisfy the high-reliability retention check. B0p25 has high-SNR accuracy +0.4795 pp relative to hard CE; B0p35 has +0.3341 pp.
- Decision:
  - Treat B0p25 as a diagnostic candidate for robustness testing because it preserves the main accuracy/NLL/Brier benefit while partially reducing posterior overconfidence.
  - Do not claim calibration superiority for the posterior-blend branch. Current evidence supports a tradeoff: posterior allocation improves likelihood/accuracy/Brier, while ECE needs an explicit calibration constraint or separate temperature/post-hoc calibration.
  - Next step is a narrow robustness expansion of B0p25 to seeds 2027 and 2028 on MLDNN + RadioML2016.10A only, before any model/dataset expansion.

## Iteration 29: B0p25 Robustness Expansion Plan

- Scope: MLDNN + RadioML2016.10A, seeds 2027 and 2028, same 400-epoch train/export/analyze pipeline.
- Candidate: RCPS-LowGate-C14-Posterior-T2-B0p25-E0p5 only.
- Motivation: Iteration 28 seed 2026 showed the best fine-blend accuracy/NLL/Brier tradeoff, but ECE remained worse than hard CE and Static LS. This expansion tests whether the accuracy/NLL/Brier benefit is stable across seeds before any model or dataset expansion.
- Decision rule:
  - If the three-seed mean keeps accuracy/NLL/Brier improvements over hard CE while ECE remains worse, the paper must present this branch as a likelihood/accuracy tradeoff rather than a calibration solution.
  - If the benefit disappears across seeds, do not use posterior-blend as a main method; keep it as diagnostic evidence that posterior mass allocation needs stronger calibration constraints.
  - Do not launch broader AMC/cross-modal runs until this robustness check is summarized.

## Iteration 29: B0p25 Robustness Expansion Launch

- Launch time: 2026-05-12 20:47 CST.
- Code commit at launch: 2a8735b.
- Runtime plan: GPU0 seed 2027, GPU1 seed 2028.
- Logs:
  - /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b0p25-e0p5_seed2027_gpu0.log
  - /home/citybuster/Data/RCPS/work_dirs/logs/rcps-lowgate-c14-posterior-t2-b0p25-e0p5_seed2028_gpu1.log
- Monitoring rule: if training succeeds but export fails, rerun export/analyze only from the same checkpoint; do not change the algorithm or schedule mid-run.

## Iteration 29: B0p25 Robustness Expansion Completed

- Completion time: 2026-05-13 02:33 CST.
- Actual code commit at completion: d1160f0. The launch record used 2a8735b, and the later alignment commit moved HEAD to d1160f0 without changing the running config semantics.
- Scope: MLDNN + RadioML2016.10A, seeds 2026/2027/2028, same 400-epoch train/export/analyze pipeline with early stopping.
- Completed robustness seeds:
  - seed 2027 completed at 2026-05-13 00:56 CST; test CSV: /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_fine_tuning_400ep/metrics/mldnn_lowgate_posterior_blend_fine_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-t2-b0p25-e0p5_seed2027_test.csv.
  - seed 2028 completed at 2026-05-13 02:29 CST; test CSV: /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_fine_tuning_400ep/metrics/mldnn_lowgate_posterior_blend_fine_tuning_400ep/deepsig201610A_mldnn_rcps-lowgate-c14-posterior-t2-b0p25-e0p5_seed2028_test.csv.
- Three-seed B0p25 overall metrics:
  - acc 63.1587 +/- 0.0451; NLL 1.0460 +/- 0.0049; ECE 0.0343 +/- 0.0038; Brier 0.4415 +/- 0.0004.
- Comparison against established three-seed baselines:
  - hard CE: acc 62.9625, NLL 1.0562, ECE 0.0325, Brier 0.4444.
  - Static LS: acc 63.1098, NLL 1.0614, ECE 0.0267, Brier 0.4424.
  - RCPS-LowGate: acc 63.0792, NLL 1.0496, ECE 0.0255, Brier 0.4419.
  - B0p25 relative to hard CE: acc +0.1962 pp, NLL -0.0102, ECE +0.0018, Brier -0.0029.
  - B0p25 relative to RCPS-LowGate: acc +0.0795 pp, NLL -0.0036, ECE +0.0088, Brier -0.0004.
- Region-level diagnostic summary:
  - B0p25 improves the transition region (-12/-10 dB) relative to hard CE: acc +0.3864 pp, NLL -0.0123, Brier -0.0034, with a small ECE penalty (+0.0012).
  - B0p25 improves mid-reliability and high-reliability NLL/Brier while preserving high-SNR accuracy retention.
  - B0p25 worsens the very-low-reliability region relative to hard CE on NLL/ECE/Brier, whereas Static LS and RCPS-LowGate remain better calibration-oriented choices there.
- Decision:
  - B0p25 is robust as a posterior-allocation / likelihood-accuracy branch: three-seed accuracy, NLL, and Brier improve over hard CE, and NLL/Brier slightly improve over RCPS-LowGate.
  - B0p25 is not a calibration-superiority branch because overall ECE is worse than hard CE, Static LS, and RCPS-LowGate.
  - The paper theory should explicitly separate two effects: reliability-conditioned entropy control for calibration, and posterior/confusion-aware mass allocation for transition-region likelihood and accuracy. Do not claim a single target parameterization dominates all metrics.
  - Next algorithmic step should be a calibration-constrained posterior target, e.g., posterior allocation with temperature or validation-calibrated post-hoc temperature on logits, but only after preserving this baseline-first evidence chain.
- Summary files:
  - /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_fine_tuning_400ep/summary/deepsig201610A_mldnn_b0p25_3seed_overall_summary.csv
  - /home/citybuster/Data/RCPS/work_dirs/mldnn_lowgate_posterior_blend_fine_tuning_400ep/summary/deepsig201610A_mldnn_b0p25_3seed_region_summary.csv

## Iteration 30: Strict-Split Baseline Gate Plan

- Discovery time: 2026-05-13 02:45 CST.
- Issue found during temperature-scaling pilot: the current RadioML2016.10A MLDNN RCPS configs use `train_and_validation.json` for training and `test.json` for both validation and testing. This is acceptable only as a legacy diagnostic setting, not as a final TPAMI protocol, because early stopping and hyperparameter decisions can observe the test split.
- Immediate decision:
  - Quarantine all previous 400-epoch MLDNN results as diagnostic/development evidence only.
  - Establish a strict split gate with `train.json`, `validation.json`, and `test.json`.
  - First rerun only MLDNN hard CE on RadioML2016.10A with the strict split. Do not launch RCPS/static/temperature variants until this strict baseline is stable.
- New strict config:
  - `configs/rcps/mldnn/mldnn_hard-ce_strict_iq-ap-snr-deepsig-201610A.py`.
  - train ann_file: `train.json`; val ann_file: `validation.json`; test ann_file: `test.json`.
- Baseline gate:
  - seeds: 2026/2027/2028.
  - same MLDNN backbone/loss/export/analyze path.
  - acceptance target: reproduce a credible hard CE baseline under the strict split before any RCPS claims are made.
- Paper implication:
  - The final paper must state the validation protocol explicitly.
  - Results selected with test-as-validation must not be reported as final results.


### Strict-Split Launch Update: 2026-05-13 02:58 CST

- Commit: `fa74c89`.
- Seed 2026 launched cleanly on GPU0.
- Seed 2027 first launch was stopped because the PowerShell-to-bash here-string preserved a CR in the seed/log path. The polluted partial work directory and log were moved to `aborted_cr_polluted`; no metrics from that launch will be used.
- Seed 2027 relaunched cleanly on GPU1 via a `tr -d \r | bash -s` launch path.
- No RCPS/static/temperature experiments are launched until strict hard CE baseline gate is complete.

### Strict-Split Baseline Gate Partial Result: 2026-05-13 09:50 CST

- Seeds completed: `2026`, `2027`.
- Protocol: train=`train.json`, validation=`validation.json`, test=`test.json`; previous test-as-validation runs remain diagnostic only.
- Seed 2026 test: acc `62.6250`, NLL `1.0735`, ECE `0.0358`, Brier `0.4502`; best validation checkpoint epoch `262`.
- Seed 2027 test: acc `62.4557`, NLL `1.0710`, ECE `0.0356`, Brier `0.4515`; best validation checkpoint epoch `272`.
- Two-seed mean: acc `62.5403 +/- 0.1197`, NLL `1.0723 +/- 0.0018`, ECE `0.0357 +/- 0.0001`, Brier `0.4509 +/- 0.0009`.
- Baseline gate status: partial pass against threshold `61.0%`; seed `2028` is still required before final hard CE baseline closure.
- Interpretation: strict validation produces a credible MLDNN baseline near the expected 0.63 level; no RCPS/static result will be promoted until the strict three-seed baseline is complete.

### Strict-Split Seed 2028 Launch: 2026-05-13 09:52 CST

- Seed `2028` launched on GPU0 using commit `d17edbf`.
- Work dir: `/home/citybuster/Data/RCPS/work_dirs/strict_split_400ep/amc/deepsig201610A/mldnn_hard-ce-strict/seed_2028`.
- Log: `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_hard-ce-strict_seed2028_gpu0.log`.
- Scope remains baseline gate only; RCPS/static experiments remain blocked until strict three-seed hard CE closure.

### Strict-Split Baseline Gate Closure: 2026-05-13 12:31 CST

- Seed `2028` completed without runtime/export/analyze errors. Test metrics: acc `62.5750`, NLL `1.0494`, ECE `0.0244`, Brier `0.4461`; best validation checkpoint epoch `196`.
- Strict three-seed hard CE MLDNN baseline on RadioML2016.10A: acc `62.5519 +/- 0.0870`, NLL `1.0646 +/- 0.0133`, ECE `0.0319 +/- 0.0066`, Brier `0.4493 +/- 0.0028`.
- Baseline gate report: `/home/citybuster/Data/RCPS/work_dirs/strict_split_400ep/summary/deepsig201610A_mldnn_hard-ce-strict_seed2026_2027_2028_gate_report.csv`; status `pass`, threshold `61.0`, margin `1.5519 pp`.
- This closes the first strict baseline gate. The old train+validation/test-as-validation runs remain diagnostic only and must not be reported as final.
- Next allowed step: strict-split Static LS and RCPS comparisons using identical backbone, optimizer, splits, export, and analysis.

## Iteration 31: Strict-Split Supervision Comparison Pilot

- Start decision time: 2026-05-13 12:35 CST.
- Precondition satisfied: strict MLDNN hard CE baseline gate passed with three-seed accuracy `62.5519 +/- 0.0870`.
- New strict configs added for Static LS, RCPS-LowGate-C14, RCPS-Retention, and RCPS-Uniform. All use train=`train.json`, validation=`validation.json`, test=`test.json`.
- First pilot launch: `static-ls-strict` seed `2026` and `rcps-lowgate-c14-strict` seed `2026`.
- Rationale: avoid teacher/posterior bases until strict validation posterior bases are rebuilt; first compare label smoothing and a purely reliability-conditioned uniform RCPS target.
- Success criteria for expansion: no export/analyze errors; compare same-seed hard CE seed `2026` on accuracy/NLL/ECE/Brier and reliability-bin behavior before launching three-seed expansion.

### Iteration 31 Pilot Launch: 2026-05-13 12:36 CST

- Commit: `cb7997b`.
- GPU0: `static-ls-strict`, seed `2026`.
- GPU1: `rcps-lowgate-c14-strict`, seed `2026`.
- Both use strict split and identical MLDNN schedule/backbone/export/analyze.
- These are same-seed pilots against strict hard CE seed `2026`, not final paper claims.

## Iteration 32 - Strict MLDNN seed-2026 supervision pilot result and follow-up launch

Time: 2026-05-14 11:35 CST
Commit before follow-up launch: e5b5148
Dataset/protocol: RadioML2016.10A strict split (`train.json` / `validation.json` / `test.json`), MLDNN, seed 2026.

Completed same-seed test metrics:

| method | test acc (%) | NLL | ECE | Brier | mean confidence | mean entropy |
|---|---:|---:|---:|---:|---:|---:|
| hard-ce-strict | 62.6250 | 1.0735 | 0.0358 | 0.4502 | 0.6621 | 0.9512 |
| static-ls-strict | 62.4989 | 1.0695 | 0.0217 | 0.4478 | 0.6210 | 1.1608 |
| rcps-lowgate-c14-strict | 62.9761 | 1.0756 | 0.0420 | 0.4476 | 0.6717 | 0.9446 |

Interpretation: Static label smoothing gives the cleanest calibration signal on this seed (lower NLL/ECE/Brier with a small accuracy drop). The current `rcps-lowgate-c14` variant improves overall accuracy (+0.351 pp) and Brier slightly, but worsens NLL/ECE and does not sufficiently raise predictive entropy in low-to-middle SNR. This variant should be treated as a diagnostic discriminative-gain signal, not yet as the final RCPS formulation.

Action: launched two stricter RCPS follow-up pilots on the same protocol and seed to test whether wider reliability-conditioned smoothing improves calibration while retaining accuracy:

- `rcps-retention-strict`, GPU0, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-retention-strict_seed2026_gpu0.log`
- `rcps-uniform-strict`, GPU1, log `/home/citybuster/Data/RCPS/work_dirs/logs/mldnn_rcps-uniform-strict_seed2026_gpu1.log`

Expected metrics:

- `/home/citybuster/Data/RCPS/work_dirs/strict_split_400ep/metrics/deepsig201610A_mldnn_rcps-retention-strict_seed2026_test.csv`
- `/home/citybuster/Data/RCPS/work_dirs/strict_split_400ep/metrics/deepsig201610A_mldnn_rcps-uniform-strict_seed2026_test.csv`

No paper claim is changed yet. The current working hypothesis is that the theory needs to emphasize posterior calibration/uncertainty alignment, while the algorithm may need a broader or entropy-matched epsilon schedule rather than the narrow `SNR <= -14 dB` lowgate.

## Iteration 33 - TPAMI baseline-first redesign implementation

Time: 2026-05-14 14:45 CST

User direction: stop treating MLDNN as the main evidence carrier; use it as a reproducibility anchor, then move the primary AMC evidence to stronger and more representative model families. The paper evidence must include accuracy, calibration, reliability-bin behavior, training efficiency, seed stability, and compute overhead rather than final accuracy alone.

Implemented infrastructure:

- Added Stage-A hard-label baseline gate runner for `CGDNet`, `PETCGDNN`, `MCformer`, and `FastMLDNN`.
- Added validation-curve training-efficiency analysis: best epoch, validation AULC, wall time, time/epoch to target validation accuracy.
- Added cross-seed metric summarization for reliability CSVs.
- Added entropy-matched RCPS epsilon table support and a builder that derives reliability-bin epsilon from validation prediction entropy.
- Added `RCPS-EntropyMatch` and `RCPS-PosteriorBase` configs for the four primary AMC model families.

Guardrails:

- No new main claim is made from the existing MLDNN pilot.
- The currently running MLDNN `rcps-retention-strict` and `rcps-uniform-strict` jobs remain diagnostic only.
- RCPS comparisons on primary models are blocked until hard CE parity gates are available.
- Any final paper statement must be backed by landed CSV/PKL outputs and manifest entries, not by training-log snippets.

### Iteration 33 Resource Intervention: 2026-05-14 15:30 CST

- Rationale: `MLDNN` is now only a diagnostic/appendix model, while the TPAMI evidence path requires strong-model baseline gates. Continuing the two MLDNN diagnostic jobs to natural completion would occupy both GPUs and delay the actual Stage-A parity gates.
- Action: stopped `mldnn_rcps-retention-strict` and `mldnn_rcps-uniform-strict` after preserving their best checkpoints, then exported test predictions and reliability metrics from those checkpoints.
- Diagnostic results on seed `2026`:
  - `rcps-retention-strict`: checkpoint epoch `241`, test acc `62.6000`, NLL `1.1311`, ECE `0.0744`, Brier `0.4589`, mean confidence `0.5516`, mean entropy `1.3992`.
  - `rcps-uniform-strict`: checkpoint epoch `230`, test acc `62.0670`, NLL `1.2186`, ECE `0.1198`, Brier `0.4861`, mean confidence `0.5009`, mean entropy `1.5730`.
- Interpretation: broad uniform smoothing and the current retention smoothing are not viable final RCPS variants on this MLDNN diagnostic. They increase entropy but damage NLL/ECE/Brier. This supports moving the algorithm toward validation-derived `RCPS-EntropyMatch` and reliability-conditioned posterior bases.
- Stage-A launched immediately afterward:
  - GPU0 queue: `CGDNet`, then `MCformer`, hard CE seeds `2026/2027/2028`.
  - GPU1 queue: `PETCGDNN`, then `FastMLDNN`, hard CE seeds `2026/2027/2028`.
  - Work root: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_v2`.

### Iteration 33 Stage-A Screen Adjustment: 2026-05-14 16:00 CST

- First seed-2026 hard CE screen completed for two models:
  - `CGDNet`: validation acc `52.8455`, test acc `52.8523`, NLL `1.2477`, ECE `0.0247`, Brier `0.5587`.
  - `PETCGDNN`: validation acc `57.4955`, test acc `57.3682`, NLL `1.1342`, ECE `0.0193`, Brier `0.5020`.
- Both are below the expected AMR-Benchmark/literature range for strong AMC baselines, especially `CGDNet`. Continuing full three-seed runs before checking other families would waste GPU time.
- Action: stopped the queued `CGDNet/PETCGDNN` seed-2027 continuations and switched Stage-A to one-seed screening for all primary families.
- Launched:
  - GPU0: `MCformer` hard CE seed `2026`.
  - GPU1: `FastMLDNN` hard CE seed `2026`.
- Decision rule: after all four one-seed screens are available, only models with credible parity or a clear, fixable parity gap are expanded to three seeds. Low-parity models enter parity debugging instead of RCPS comparison.


### Iteration 33 Baseline Parity Debug: 2026-05-14 17:35 CST

- Foreground monitoring found that `FastMLDNN` was using a 3200-epoch schedule without EarlyStopping in the current config. At epoch 180 it remained around 54-55% validation accuracy, so continuing would occupy GPU1 for many hours without credible baseline-gate value.
- Action: stopped the `FastMLDNN` seed-2026 screen after preserving `best_accuracy_top1_epoch_180.pth`; exported validation/test predictions from that checkpoint.
- Landed FastMLDNN metrics: validation acc `54.8318`, test acc `54.6568`, test NLL `1.3360`, ECE `0.0753`, Brier `0.5513`. This model/config is marked parity-failed for now and will not enter RCPS comparisons before deeper reproduction debugging.
- Added a reliability-metadata fallback in `tools/rcps/collect_predictions.py`: if an original baseline config does not pack `sample_idx`, validation/test export now recovers SNR from the sequential dataset index. This enables original repository configs to be used for baseline gate and reliability-bin analysis.
- Extended `tools/rcps/run_amc_matrix.py` with hard-CE entries for classic baseline candidates: `CNN4`, `GRU2`, `LSTM2`, `CLDNNL`, `CLDNNW`, `DSCLDNN`, `HCGDNN`, `MCNet`, and `DensCNN`.
- Launched classic one-seed screen on GPU1: `CNN4 -> GRU2 -> CLDNNW`, max `400` epochs, num_workers `0`, work root `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_classic_screen`, log `/home/citybuster/Data/RCPS/work_dirs/logs/stage_a_classic_screen_gpu1.log`.
- Current `CGDNet` rerun with restored GRU initializer and `RNN` alias is still low (~52% by epoch 86). It will be allowed to finish/export, but current evidence marks it as a parity-debug candidate rather than a main-table model.

Decision: no RCPS comparison is launched on parity-failed models. The next goal is to identify at least two stable AMC backbones from the classic/strong candidate pool before testing RCPS variants.


### Iteration 33 Runner Environment Fix: 2026-05-14 17:40 CST

- The first classic baseline queue failed before training because `run_baseline_gate.py` spawned child commands with literal `python`, while the nohup environment did not expose `python` on PATH.
- Fix: changed RCPS runner scripts to spawn child processes with `sys.executable` so the same conda interpreter is used in foreground and background sessions.
- Verification: dry-run commands now expand to `/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python`.
- Relaunched `CNN4 -> GRU2 -> CLDNNW` on GPU1. `CNN4` started normally and reached `40.18%` validation accuracy by epoch 3, so the queue is now healthy.


### Iteration 34 - AMR-compatible Split Support: 2026-05-14 17:50 CST

- Reference audit of AMR-Benchmark showed that its RadioML2016.10A loader uses a per modulation/SNR split of `600/200/200` for train/validation/test with seed `2016`.
- Server converted JSONs currently use `500/100/400`, so external parity and strict-server robustness must be treated as two different protocols.
- Added `tools/rcps/prepare_amr_compatible_split.py` and generated `/home/citybuster/Data/RCPS/processed/amr_compatible/RadioML.2016.10A` with train `132000`, validation `44000`, test `44000`; `iq/` is a symlink to the original converted dataset.
- Added `--data-root` support to RCPS runners and fixed prediction export so data-root overrides are applied consistently during train, collect, and test stages.
- Policy: AMR-compatible split will be used for external baseline parity; server-strict split remains a robustness protocol. Results from the two protocols will not be mixed in the same table.


### Iteration 34 Runtime Adjustment: 2026-05-14 18:15 CST

- Strict-server split diagnostic results so far:
  - `CNN4`: validation acc `50.5318`, test acc `50.5170`; parity-failed for main use.
  - `LSTM2`: stopped at best epoch `65`, validation acc `40.8864`, test acc `40.6466`; parity-failed for main use.
  - `GRU2`: still running and promising relative to the other classic candidates, reaching about `57.3%` validation accuracy by epoch `58`.
- Action: stopped the low-value `LSTM2 -> CLDNNL -> DSCLDNN` strict queue after exporting `LSTM2` metrics. This frees GPU0 for the more relevant AMR-compatible protocol.
- Launched AMR-compatible strong-model screen on GPU0: `CGDNet -> PETCGDNN -> MCformer`, seed `2026`, max `400` epochs, data root `/home/citybuster/Data/RCPS/processed/amr_compatible/RadioML.2016.10A`, work root `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_amr_split`, log `/home/citybuster/Data/RCPS/work_dirs/logs/stage_a_amr_split_strong_gpu0.log`.
- Rationale: AMR-compatible split aligns with the external AMR-Benchmark 600/200/200 protocol and is the correct place to judge framework parity for these strong baselines.


### Iteration 34 Classic Screen Update: 2026-05-14 18:45 CST

- Strict-server split classic metrics:
  - `GRU2`: validation acc `58.9227`, test acc `58.6500`, test NLL `1.1341`, ECE `0.0339`, Brier `0.4970`. This is the strongest non-MLDNN classic candidate so far, but still below the MLDNN anchor.
  - `CLDNNW`: stopped at best epoch `60`, validation acc `53.7273`, test acc `53.6602`; parity-failed for main use.
- Action: stopped the remaining strict classic queue after exporting `CLDNNW` metrics. The strict protocol has now served its diagnostic purpose: `CNN4/LSTM2/CLDNNW` are weak, `GRU2` is usable but not main-strength.
- Launched AMR-compatible classic/anchor queue on GPU1: `GRU2 -> MLDNN`, seed `2026`, max `400` epochs, data root `/home/citybuster/Data/RCPS/processed/amr_compatible/RadioML.2016.10A`, work root `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_amr_split_classic`, log `/home/citybuster/Data/RCPS/work_dirs/logs/stage_a_amr_split_classic_gpu1.log`.


### Iteration 35 GRU2 RCPS Readiness: 2026-05-14 19:20 CST

- Foreground monitoring status:
  - AMR-compatible strong queue is running `PETCGDNN` seed `2026`; validation accuracy is around `57.5%` by epoch `73`, with no `Traceback`, CUDA OOM, file-handle error, or subprocess error.
  - AMR-compatible classic queue is running `GRU2` seed `2026`; validation accuracy crossed `60%` and is plateauing around `60.2-60.5%` by epoch `115`, with no runtime errors.
  - `CGDNet` on AMR-compatible split completed earlier with validation acc `53.8227`, test acc `53.4727`, NLL `1.2199`, ECE `0.0241`, Brier `0.5485`; it remains parity-failed and will not enter RCPS comparisons before deeper reproduction debugging.
- Added GRU2 RCPS-ready configs that preserve the original `GRU2` backbone and data transform while packing reliability metadata: hard CE with metadata, Static LS, RCPS-Uniform, RCPS-Retention, RCPS-Confusion, RCPS-EntropyMatch, and RCPS-PosteriorBase.
- Added GRU2 method entries to `tools/rcps/run_amc_matrix.py` so paired comparisons can use the same runner and data-root overrides as the baseline gate.
- Verification: all new GRU2 configs parse with `mmengine.Config.fromfile`, expose `('sample_idx', 'snr', 'snr_label', 'modulation')`, and the runner scripts pass `py_compile`.
- Decision: no RCPS job is launched until the AMR-compatible `GRU2` test metrics are exported. If its test accuracy remains near or above `60%`, it becomes the first candidate for paired `Hard CE / Static LS / RCPS-Retention / RCPS-PosteriorBase` comparison; otherwise it remains diagnostic.


### Iteration 36 AMR GRU2 Gate and Paired Launch: 2026-05-14 19:35 CST

- AMR-compatible baseline results landed:
  - `GRU2` seed `2026`: validation acc `60.6227`, test acc `60.5591`, test NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - `PETCGDNN` seed `2026`: validation acc `57.6182`, test acc `57.4591`, test NLL `1.1321`, ECE `0.0262`, Brier `0.4998`.
- `GRU2` is not a final TPAMI main model yet, but it is the first non-MLDNN candidate on the AMR-compatible protocol to clear the 60% diagnostic threshold. It is therefore eligible for a one-seed paired supervision comparison.
- The AMR-compatible MLDNN anchor failed before training because the MLDNN config expects `train_and_validation.json`. This was a data-entry issue; generated a correct merged JSON with `176000` records from AMR-compatible train and validation splits.
- Built GRU2 validation-derived RCPS tables from validation predictions only:
  - entropy table: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/gru2_hard-ce_seed2026_entropy_match.npz`
  - reliability-conditioned posterior base: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/gru2_hard-ce_seed2026_reliability_base.npz`
  - class confusion base: `/home/citybuster/Data/RCPS/work_dirs/confusion_bases/deepsig201610A_gru2_seed2026.npy`
- Launched GRU2 paired comparison on GPU1 with work root `/home/citybuster/Data/RCPS/work_dirs/gru2_paired_amr_400ep` and log `/home/citybuster/Data/RCPS/work_dirs/logs/gru2_paired_amr_gpu1.log`.
- Paired methods: `Static LS`, `RCPS-Retention`, `RCPS-EntropyMatch`, and `RCPS-PosteriorBase`; all use the same GRU2 backbone, AMR-compatible split, seed `2026`, and training schedule. This is a diagnostic paired comparison, not yet a paper claim.


### Iteration 37 GRU2 Three-Seed Baseline Expansion: 2026-05-14 20:20 CST

- AMR-compatible `MCformer` seed `2026` completed: validation accuracy `58.35%`, test accuracy `57.97%`. It is below the current GRU2 hard-CE baseline and is not promoted to RCPS comparison at this stage.
- `GRU2` paired comparison on GPU1 is still running normally. The first candidate, Static LS `0.05`, has recovered to roughly `59.8%` validation accuracy by epoch `130`, close to but below hard CE (`60.62%` validation for seed `2026`). No conclusions are drawn before CSV export.
- GPU0 was freed after MCformer. Action: launched AMR-compatible `GRU2` hard CE seeds `2027` and `2028` on GPU0 to test baseline stability before any three-seed RCPS claim.
- Rationale: GRU2 is currently the cleanest non-MLDNN baseline candidate. Completing hard CE three seeds is more useful than spending the freed GPU on the MLDNN anchor, whose current config trains on train+validation and validates on test.


### Iteration 38 Static-LS Triage and RCPS Focus: 2026-05-14 21:10 CST

- GRU2 Static-LS diagnostics on AMR-compatible split, seed `2026`:
  - `ls=0.05`: test acc `59.9750`, NLL `1.1370`, ECE `0.0276`, Brier `0.4853`; low-SNR NLL/ECE improve but overall NLL/Brier and high-SNR accuracy worsen relative to hard CE.
  - `ls=0.10`: test acc `59.8977`, NLL `1.1517`, ECE `0.0323`, Brier `0.4840`; low-SNR NLL/ECE improve further, but overall ECE and NLL do not beat hard CE.
- Interpretation: static smoothing supports the phenomenon claim (low-reliability hard CE is overconfident) but does not solve the main tradeoff. It reduces low-SNR overconfidence at the cost of overall sharpness/high-reliability performance.
- Intervention: stopped the remaining Static-LS `0.20` run and the original grid runner after preserving the landed `0.05/0.10` metrics. Relaunched GPU1 with only RCPS candidates: `RCPS-Retention`, `RCPS-EntropyMatch`, and `RCPS-PosteriorBase`.
- This is a deliberate matrix reduction to prioritize hypotheses that can address the observed tradeoff. The skipped `ls=0.20` is recorded as a low-value static-smoothing candidate, not as a completed result.

### Iteration 39 Baseline Stability and Keras-Init Parity Probe: 2026-05-14 21:45 CST

- AMR-compatible `GRU2` hard CE three-seed baseline is complete:
  - seed `2026`: test acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - seed `2027`: test acc `58.1523`, NLL `1.1262`, ECE `0.0296`, Brier `0.4974`.
  - seed `2028`: test acc `58.2568`, NLL `1.1201`, ECE `0.0270`, Brier `0.4956`.
  - mean/std: acc `58.9894 +/- 1.3604`, NLL `1.1161 +/- 0.0127`, ECE `0.0293 +/- 0.0022`, Brier `0.4895 +/- 0.0121`.
- Interpretation: GRU2 is useful for supervision diagnostics but has too much seed variance to serve as a main TPAMI evidence backbone by itself.
- AMR-Benchmark audit confirms its Keras implementations use `glorot_uniform` Conv/Dense kernels and recurrent defaults, while several PyTorch baselines in this repo lacked explicit initialization. This is a plausible framework-parity gap because the already-stable MLDNN anchor includes Xavier/RNN initialization.
- Added parity-only configs:
  - `configs/rcps/parity/petcgdnn_hard-ce_kerasinit_iq-snr-deepsig-201610A.py`
  - `configs/rcps/parity/mcldnn_hard-ce_kerasinit_iq-snr-deepsig-201610A.py`
- Launched GPU0 Keras-init parity probe on AMR-compatible split: `PETCGDNN -> MCLDNN`, seed `2026`, max `400` epochs, same optimizer/loss/split as baseline, only initialization changed. Log: `/home/citybuster/Data/RCPS/work_dirs/logs/parity_kerasinit_gpu0.log`.
- GPU1 continues the GRU2 RCPS-only grid. The first candidate, `rcps-retention_eps0.3_gamma1.0`, has validation accuracy around `61.5%` by epoch `90+`, above hard CE seed `2026`; no conclusion until test CSV lands.

### Iteration 40 First GRU2 RCPS-Retention Result: 2026-05-14 21:58 CST

- First RCPS candidate on AMR-compatible `GRU2` seed `2026` completed: `rcps-retention_eps0.3_gamma1.0`.
- Test comparison against same-seed hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - Static LS `0.05`: acc `59.9750`, NLL `1.1370`, ECE `0.0276`, Brier `0.4853`.
  - Static LS `0.10`: acc `59.8977`, NLL `1.1517`, ECE `0.0323`, Brier `0.4840`.
  - RCPS-Retention `eps0.3/gamma1`: acc `61.6614`, NLL `1.1175`, ECE `0.0250`, Brier `0.4625`.
- Reliability-bin behavior for RCPS-Retention vs hard CE:
  - Low SNR (`<= -10 dB`) NLL improves by `-0.0267`, ECE by `-0.0266`, Brier by `-0.0106`.
  - High SNR (`>= 10 dB`) accuracy improves by `+1.7091 pp`, while high-SNR ECE worsens.
- Interpretation: this supports the central observation that reliability-conditioned targets can improve low-reliability posterior quality and avoid the high-SNR accuracy loss of static smoothing. However, overall NLL worsens and high-SNR ECE worsens, so the claim remains a tradeoff-aware `posterior calibration / uncertainty alignment` claim rather than a universal all-metric win.
- Training-efficiency summary using target accuracy `57.5916` (95% of hard CE best validation): hard CE reaches target at epoch `42`; RCPS-Retention also reaches target at epoch `42`; Static LS `0.05` reaches it at epoch `48`; Static LS `0.10` at epoch `41`. RCPS has the highest validation AULC among these four (`55.7713`).
- Next action: continue the RCPS grid to test whether entropy/posterior variants can repair the NLL/ECE tradeoff; do not expand to three seeds until a variant beats hard CE on a clearer metric set or the tradeoff is theoretically justified.

### Iteration 41 PETCGDNN Keras-Init Parity Result: 2026-05-14 22:22 CST

- PETCGDNN Keras-compatible initialization parity probe completed on AMR-compatible RadioML2016.10A, seed `2026`.
- Original PETCGDNN hard CE on the same AMR-compatible protocol: test acc `57.4591`, NLL `1.1321`, ECE `0.0262`, Brier `0.4998`.
- Keras-init PETCGDNN: validation acc `60.2159`, test acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`.
- Delta vs original: accuracy `+2.5318 pp`, NLL `-0.0128`, Brier `-0.0174`; ECE slightly worsens by `+0.0009`.
- Interpretation: framework/parity details materially affect AMC baseline reproduction. Keras-style Conv/Dense/RNN initialization brings PETCGDNN close to the 60% diagnostic line and prevents us from incorrectly dismissing it as a weak model. The main experiment should use parity-corrected baselines where justified, not raw PyTorch-default reproductions.
- The same parity queue has moved to MCLDNN Keras-init; early epochs are still at chance level, so this variant is under watch and will be diagnosed if it does not leave the plateau.

### Iteration 42 MCLDNN Parity Probe Stop and PETCGDNN Seed Expansion: 2026-05-14 22:32 CST

- MCLDNN Keras-init parity probe stayed at chance accuracy (`9.0909%`) through epoch `40`, with validation loss near `2.3979` and no sign of learning.
- Intervention: stopped the MCLDNN Keras-init run to avoid wasting GPU time. This is recorded as a parity-probe failure for this particular initialization override, not as evidence against MCLDNN generally.
- PETCGDNN Keras-init completed earlier with test acc `59.9909`, a `+2.5318 pp` improvement over the original PETCGDNN reproduction. Because this is now a plausible parity-corrected baseline, launched PETCGDNN Keras-init seeds `2027` and `2028` on GPU0 for a three-seed baseline gate.
- New work root: `/home/citybuster/Data/RCPS/work_dirs/petcgdnn_kerasinit_gate_amr`; log: `/home/citybuster/Data/RCPS/work_dirs/logs/petcgdnn_kerasinit_gate_gpu0.log`.

### Iteration 43 GRU2 Retention Gamma-2 Diagnostic: 2026-05-14 22:50 CST

- GRU2 paired RCPS grid continued on the AMR-compatible RadioML2016.10A split, seed `2026`.
- New completed candidate: `rcps-retention_eps0.3_gamma2.0`.
- Test comparison against same-seed hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Retention `eps0.3/gamma1`: acc `61.6614`, NLL `1.1175`, ECE `0.0250`, Brier `0.4625`.
  - RCPS-Retention `eps0.3/gamma2`: acc `60.3818`, NLL `1.1137`, ECE `0.0098`, Brier `0.4746`.
- Interpretation: a steeper reliability schedule (`gamma=2`) strongly improves ECE but loses the accuracy/Brier advantage of `gamma=1` and still does not repair NLL. This suggests that the RCPS schedule controls a real accuracy-calibration tradeoff rather than giving a free universal improvement.
- Current working hypothesis: `eps0.3/gamma1` remains the best GRU2 diagnostic candidate for balanced accuracy/ECE/Brier, while `eps0.3/gamma2` is useful evidence for uncertainty alignment. Continue the grid to test whether higher `epsilon_max`, entropy-matched, or posterior-base variants improve NLL without losing retention.
- GPU1 automatically advanced to `rcps-retention_eps0.5_gamma1.0`. GPU0 continues PETCGDNN Keras-init seed `2027`, which has reached approximately `60.15%` validation accuracy by epoch `52`, close to seed `2026`.

### Iteration 44 PETCGDNN Keras-Init Seed 2027 Complete: 2026-05-14 23:15 CST

- PETCGDNN Keras-init baseline gate expansion progressed normally; no export or file-handle failure occurred.
- Seed `2027` completed on the AMR-compatible split:
  - validation accuracy `60.5159`;
  - test accuracy `60.3182`;
  - test NLL `1.1062`;
  - test ECE `0.0298`;
  - test Brier `0.4753`.
- Two available PETCGDNN Keras-init seeds now give mean test acc `60.1545 +/- 0.2314`, mean NLL `1.1128 +/- 0.0093`, mean ECE `0.0285 +/- 0.0019`, and mean Brier `0.4789 +/- 0.0050`.
- Interpretation: PETCGDNN with Keras-compatible initialization is currently the most stable non-MLDNN baseline candidate on the AMR-compatible protocol. Seed `2028` has started automatically; wait for it before deciding baseline-gate pass/fail or launching PETCGDNN RCPS variants.
- GRU2 `rcps-retention_eps0.5_gamma1.0` is still running and has reached validation accuracy above `61.7%`, so it remains a promising diagnostic candidate pending test export.

### Iteration 45 GRU2 Retention eps0.5/gamma1 Diagnostic: 2026-05-14 23:38 CST

- GRU2 `rcps-retention_eps0.5_gamma1.0` completed and exported validation/test predictions.
- Overall test comparison against same-seed hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Retention `eps0.3/gamma1`: acc `61.6614`, NLL `1.1175`, ECE `0.0250`, Brier `0.4625`.
  - RCPS-Retention `eps0.5/gamma1`: acc `61.8114`, NLL `1.1451`, ECE `0.0644`, Brier `0.4661`.
- Reliability-bin aggregates for `eps0.5/gamma1`:
  - Low SNR (`<= -10 dB`): acc `13.0379`, NLL `2.3448`, ECE `0.1098`, Brier `0.9070`.
  - High SNR (`>= 10 dB`): acc `92.2818`, NLL `0.3031`, ECE `0.0892`, Brier `0.1485`.
- Compared with hard CE, `eps0.5/gamma1` improves overall accuracy by `+1.2523 pp`, improves Brier by `-0.0094`, improves low-SNR NLL/ECE/Brier, and improves high-SNR accuracy/NLL/Brier. However, it worsens overall ECE by `+0.0330` and high-SNR ECE by `+0.0198`.
- Interpretation: this candidate strengthens the evidence that reliability-conditioned retention targets can improve accuracy and reliability-stratified Brier/low-SNR behavior, but it also shows that schedule-only RCPS can distort calibration shape. The next algorithmic priority is entropy matching, posterior-base targets, or temperature treatment rather than simply increasing epsilon.
- GPU1 automatically advanced to `rcps-retention_eps0.5_gamma2.0`. PETCGDNN Keras-init seed `2028` remains running normally.

### Iteration 46 PETCGDNN Keras-Init Baseline Gate Complete: 2026-05-15 00:02 CST

- PETCGDNN Keras-init hard CE baseline gate completed all three seeds on the AMR-compatible RadioML2016.10A split.
- Test metrics:
  - seed `2026`: acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`;
  - seed `2027`: acc `60.3182`, NLL `1.1062`, ECE `0.0298`, Brier `0.4753`;
  - seed `2028`: acc `59.5023`, NLL `1.1296`, ECE `0.0295`, Brier `0.4879`.
- Three-seed mean/std: acc `59.9371 +/- 0.4106`, NLL `1.1184 +/- 0.0117`, ECE `0.0288 +/- 0.0015`, Brier `0.4819 +/- 0.0063`.
- Interpretation: although the mean accuracy is slightly below the 60% diagnostic line, the variance is small and the Keras-init correction gives a clear improvement over the raw PETCGDNN reproduction. PETCGDNN Keras-init is therefore promoted to paired RCPS comparison as a stable non-MLDNN candidate.
- Added Keras-init PETCGDNN paired configs for `static-ls`, `rcps-retention`, `rcps-entropy`, and `rcps-posterior`, and registered the `petcgdnn_kerasinit` model alias in the RCPS matrix runner.
- Built validation-only tables from PETCGDNN Keras-init seed `2026`:
  - entropy match table: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/petcgdnn_kerasinit_seed2026_entropy_match.npz`;
  - reliability-conditioned posterior base: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/petcgdnn_kerasinit_seed2026_reliability_base.npz`.
- Next action: launch PETCGDNN Keras-init paired comparison on GPU0 using the same AMR-compatible split, seed `2026`, and max `400` epochs; this is a diagnostic paired comparison before any three-seed RCPS claim.

### Iteration 47 GRU2 Retention eps0.5/gamma2 Diagnostic and PETCGDNN Paired Monitor: 2026-05-15 00:20 CST

- Foreground monitor status: GPU0 is running PETCGDNN Keras-init paired diagnostics (`static-ls` first stage); GPU1 is running the GRU2 RCPS grid. No new `Traceback`, `CUDA out of memory`, or `Too many open files` was found. The only error line remains the previously intentional `SIGTERM` for `gru2_static-ls_ls0.2`.
- GRU2 `rcps-retention_eps0.5_gamma2.0` completed and exported validation/test predictions.
- Overall test comparison against same-seed hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Retention `eps0.5/gamma2`: acc `62.1068`, NLL `1.0943`, ECE `0.0186`, Brier `0.4546`.
- Delta versus hard CE: accuracy `+1.5477 pp`, NLL `-0.0075`, ECE `-0.0128`, Brier `-0.0209`.
- Reliability-bin aggregates for `eps0.5/gamma2`:
  - Low SNR (`<= -10 dB`): acc `13.2273`, NLL `2.3612`, ECE `0.1388`, Brier `0.9146`.
  - High SNR (`>= 10 dB`): acc `92.3909`, NLL `0.2804`, ECE `0.0803`, Brier `0.1429`.
- Interpretation: this is the strongest GRU2 single-seed RCPS candidate so far because it improves all four overall metrics and improves low-SNR NLL/ECE/Brier. High-SNR accuracy/NLL/Brier also improve, but high-SNR ECE remains worse than hard CE, so the theory should still emphasize reliability-conditioned posterior alignment rather than claiming uniformly better calibration in every reliability bin.
- PETCGDNN Keras-init paired diagnostic is still in progress: `static-ls_ls0.05` reached validation accuracy above `57.6%` by epoch `37`; no intervention is needed yet.

### Iteration 48 PETCGDNN Keras-Init Static LS 0.05 Diagnostic: 2026-05-15 00:49 CST

- PETCGDNN Keras-init paired diagnostic exported `static-ls_ls0.05` validation/test predictions on the AMR-compatible RadioML2016.10A split, seed `2026`.
- Same-seed test comparison against hard CE Keras-init:
  - Hard CE: acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`.
  - Static LS `0.05`: acc `59.5068`, NLL `1.1431`, ECE `0.0267`, Brier `0.4880`.
- Delta versus hard CE: accuracy `-0.4841 pp`, NLL `+0.0238`, ECE `-0.0004`, Brier `+0.0056`.
- Reliability-bin deltas: low-SNR NLL `+0.0171`, low-SNR ECE `+0.0214`, high-SNR accuracy `-0.7909 pp`, high-SNR ECE `+0.0160`.
- Interpretation: static label smoothing is not solving the degraded-observation mismatch for this stable PETCGDNN candidate. It slightly changes global ECE but hurts accuracy, NLL, Brier, and low-SNR posterior quality. This strengthens the need for reliability-conditioned and retention-aware targets rather than sample-independent smoothing.
- Tooling update: `tools/rcps/compare_reliability_metrics.py` now supports `--ignore-model` so parity-corrected aliases such as `petcgdnn` versus `petcgdnn_kerasinit` can be compared against the same hard-CE baseline without hand editing CSV files.

### Iteration 49 GRU2 Retention eps0.7/gamma1 Diagnostic: 2026-05-15 01:11 CST

- GRU2 `rcps-retention_eps0.7_gamma1.0` completed and exported validation/test predictions.
- Same-seed test comparison against hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Retention `eps0.7/gamma1`: acc `61.7205`, NLL `1.1975`, ECE `0.0988`, Brier `0.4821`.
- Delta versus hard CE: accuracy `+1.1614 pp`, NLL `+0.0957`, ECE `+0.0675`, Brier `+0.0066`.
- Reliability-bin deltas: low-SNR NLL improves by `-0.0168`, but high-SNR ECE worsens by `+0.0247` while global posterior metrics degrade.
- Interpretation: increasing `epsilon_max` to `0.7` with a shallow schedule over-softens the targets. It may still improve accuracy relative to hard CE, but it damages NLL/ECE/Brier and should not be a main RCPS candidate. Current GRU2 best remains `eps0.5/gamma2`, which is the only completed GRU2 candidate that improves accuracy, NLL, ECE, and Brier together.

### Iteration 50 PETCGDNN Keras-Init Static LS 0.1 Diagnostic: 2026-05-15 01:32 CST

- PETCGDNN Keras-init paired diagnostic exported `static-ls_ls0.1` validation/test predictions on the AMR-compatible RadioML2016.10A split, seed `2026`.
- Same-seed test comparison against hard CE Keras-init:
  - Hard CE: acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`.
  - Static LS `0.1`: acc `60.2955`, NLL `1.1567`, ECE `0.0325`, Brier `0.4791`.
- Delta versus hard CE: accuracy `+0.3045 pp`, NLL `+0.0374`, ECE `+0.0054`, Brier `-0.0033`.
- Reliability-bin deltas: low-SNR NLL `+0.0030`, low-SNR ECE `+0.0015`, high-SNR accuracy `+1.0636 pp`, high-SNR ECE `+0.0520`.
- Interpretation: static label smoothing can improve accuracy/Brier slightly on this seed, but it does not improve posterior quality. It worsens NLL and ECE overall and does not improve low-SNR calibration. This supports the claim that sample-independent smoothing is an incomplete baseline for degraded-observation supervision.
- PETCGDNN paired queue has advanced to `rcps-retention_eps0.3_gamma1.0`; keep monitoring before making any RCPS claim on PETCGDNN.

### Iteration 51 GRU2 Retention eps0.7/gamma2 Diagnostic: 2026-05-15 01:58 CST

- Foreground monitor status: PETCGDNN Keras-init `rcps-retention_eps0.3_gamma1.0` is still running normally on GPU0; GRU2 has completed `rcps-retention_eps0.7_gamma2.0` and automatically advanced to `rcps-entropy_eps0.3_gamma1.0` on GPU1. No new `Traceback`, CUDA OOM, or file-handle failure was observed. The only error scan hit remains the previously intentional `SIGTERM` for `gru2_static-ls_ls0.2`.
- GRU2 `rcps-retention_eps0.7_gamma2.0` exported validation/test predictions.
- Same-seed test comparison against hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Retention `eps0.7/gamma2`: acc `61.7955`, NLL `1.1141`, ECE `0.0398`, Brier `0.4599`.
- Delta versus hard CE: accuracy `+1.2364 pp`, NLL `+0.0122`, ECE `+0.0085`, Brier `-0.0156`.
- Reliability-bin deltas: low-SNR NLL `-0.0236`, low-SNR ECE `-0.0431`, high-SNR accuracy `+2.5364 pp`, high-SNR ECE `+0.0124`.
- Interpretation: the steeper `gamma=2` schedule makes `epsilon_max=0.7` much safer than the shallow `gamma=1` schedule, and it improves accuracy/Brier plus low-SNR posterior behavior. However, it still worsens global NLL/ECE compared with hard CE. The current GRU2 best remains `rcps-retention_eps0.5_gamma2.0`, which is the completed candidate that improves overall accuracy, NLL, ECE, and Brier simultaneously.

### Iteration 52 PETCGDNN Keras-Init RCPS-Retention eps0.3/gamma1 Diagnostic: 2026-05-15 02:08 CST

- PETCGDNN Keras-init `rcps-retention_eps0.3_gamma1.0` completed and exported validation/test predictions. The run stopped after patience expired with best validation accuracy `60.6682` at epoch `67`; test accuracy is `60.5273`.
- Same-seed test comparison against hard CE Keras-init:
  - Hard CE: acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`.
  - Static LS `0.05`: acc `59.5068`, NLL `1.1431`, ECE `0.0267`, Brier `0.4880`.
  - Static LS `0.1`: acc `60.2955`, NLL `1.1567`, ECE `0.0325`, Brier `0.4791`.
  - RCPS-Retention `eps0.3/gamma1`: acc `60.5273`, NLL `1.1403`, ECE `0.0229`, Brier `0.4743`.
- Delta versus hard CE: accuracy `+0.5364 pp`, NLL `+0.0209`, ECE `-0.0042`, Brier `-0.0081`.
- Reliability-bin deltas: low-SNR NLL `+0.0061`, low-SNR ECE `-0.0053`, high-SNR accuracy `+1.3909 pp`, high-SNR ECE `+0.0198`.
- Interpretation: this is the first PETCGDNN paired result where RCPS beats static label smoothing on accuracy, ECE, and Brier under the same seed and training budget. The NLL regression means the method is still not publication-ready as-is for this backbone. Continue with `gamma=2`, posterior base, and entropy matching to test whether the target schedule can retain the accuracy/Brier benefit while fixing likelihood quality.

### Iteration 53 Posterior-Base Candidate Emerges: 2026-05-15 10:16 CST

- The overnight paired diagnostics completed a full PETCGDNN Keras-init single-seed sweep and several GRU2 posterior-base points. GPU0 is now free; GPU1 continues the GRU2 posterior grid.
- PETCGDNN Keras-init single-seed test comparison against hard CE:
  - Hard CE: acc `59.9909`, NLL `1.1193`, ECE `0.0271`, Brier `0.4824`.
  - Best static LS (`0.1` by accuracy/Brier): acc `60.2955`, NLL `1.1567`, ECE `0.0325`, Brier `0.4791`.
  - Best retention by accuracy (`eps0.5/gamma1`): acc `61.2455`, NLL `1.1638`, ECE `0.0596`, Brier `0.4717`.
  - Best posterior-base candidate (`eps0.3/gamma1`): acc `61.0591`, NLL `1.0882`, ECE `0.0104`, Brier `0.4622`.
- PETCGDNN posterior-base `eps0.3/gamma1` delta versus hard CE: accuracy `+1.0682 pp`, NLL `-0.0311`, ECE `-0.0167`, Brier `-0.0202`; low-SNR NLL `-0.0193` and low-SNR ECE `-0.0080`; high-SNR accuracy `+2.3091 pp` with only `+0.0017` high-SNR ECE.
- GRU2 posterior-base `eps0.3/gamma2` also gives a consistent single-seed improvement: accuracy `+0.9386 pp`, NLL `-0.0267`, ECE `-0.0159`, Brier `-0.0158`, with low-SNR NLL `-0.0123`.
- Interpretation: the strongest evidence now favors the theory variant where the low-reliability base is not merely uniform but a reliability-conditioned posterior/confusion base estimated on validation data. This better matches class-overlap structure and fixes the NLL/ECE weakness seen in uniform/retention-only smoothing. The manuscript should move `RCPS-PosteriorBase` to the algorithmic center and treat uniform/retention as ablations, pending multi-seed confirmation.
- Action: start PETCGDNN Keras-init posterior-base `eps0.3/gamma1` for seeds `2027` and `2028` on GPU0, using the same AMR-compatible split and training budget. The hard-CE PETCGDNN Keras-init baseline already has seeds `2026/2027/2028`, so this is the first multi-seed confirmation step for the emerging main method.

### Iteration 54 GRU2 Posterior eps0.5/gamma1 Diagnostic: 2026-05-15 10:37 CST

- Foreground monitor status: PETCGDNN Keras-init posterior-base seed `2027` is running normally on GPU0; GRU2 posterior grid is running on GPU1; CIFAR-10-C is downloading in the background and has grown beyond the earlier incomplete 21MB archive. No new runtime error was observed.
- GRU2 `rcps-posterior_eps0.5_gamma1.0` completed and exported validation/test predictions.
- Same-seed test comparison against hard CE:
  - Hard CE: acc `60.5591`, NLL `1.1019`, ECE `0.0313`, Brier `0.4755`.
  - RCPS-Posterior `eps0.5/gamma1`: acc `61.7136`, NLL `1.0824`, ECE `0.0111`, Brier `0.4565`.
- Delta versus hard CE: accuracy `+1.1545 pp`, NLL `-0.0195`, ECE `-0.0202`, Brier `-0.0189`.
- Reliability-bin deltas: low-SNR NLL `-0.0113`, low-SNR ECE `-0.0093`, high-SNR accuracy `+2.2000 pp`, high-SNR ECE `+0.0049`.
- Interpretation: GRU2 now has two posterior-base candidates that improve all overall metrics versus hard CE. This reinforces the emerging algorithmic decision that validation-estimated posterior/confusion bases are central, while uniform, entropy-only, and retention-only variants should be treated as ablations.

### Iteration 55 PETCGDNN 10B Ensemble Posterior Correction: 2026-05-16 04:57 CST

- PETCGDNN Keras-init `RadioML2016.10B` ensemble posterior-base confirmation completed for seeds `2026/2027/2028`.
- Important correction: the legacy work directory name says `eps0p3_g1p0`, but the actual config used the default retention-power schedule from `petcgdnn_rcps-posterior_kerasinit_iq-snr-deepsig-201610A.py`: `epsilon.max=0.7`, `gamma=1.0`, `retain_min=0.8`, with the 3-teacher validation posterior table. Corrected metrics were regenerated under `metrics_corrected` with method name `rcps-posterior-ens3_retmax0p7_g1p0_retain0p8`.
- Three-seed test deltas versus hard CE on the AMR-compatible RadioML2016.10B split:
  - mean accuracy `+1.4508 pp` (`61.6133%` absolute);
  - mean NLL `-0.0110`;
  - mean ECE `-0.0121`;
  - mean Brier `-0.0190`.
- Reliability-bin behavior:
  - low-SNR mean ECE `-0.0196` and Brier `-0.0029`, but low-SNR accuracy `-0.3528 pp`;
  - high-SNR accuracy `+2.0400 pp` and Brier `-0.0269`, but high-SNR ECE `+0.0106`.
- Interpretation: ensemble posterior bases stabilize PETCGDNN on a second AMC dataset and provide accuracy, ECE, and Brier gains. The low-SNR accuracy and high-SNR ECE tradeoffs mean the paper should not claim universal accuracy gains in every reliability bin. The stronger, defensible claim is posterior-quality and reliability-alignment improvement, with accuracy gains observed in aggregate and high-reliability bins for this model/dataset.

### Iteration 56 CGDNet Baseline Gate Launched: 2026-05-16 04:58 CST

- GPU0 became free after PETCGDNN 10B ensemble posterior export and corrected analysis.
- Launched `CGDNet + RadioML2016.10A_AMR + hard CE` baseline parity gate for seeds `2026/2027/2028`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_amr_split_strong_v2`.
- Log: `/home/citybuster/Data/RCPS/work_dirs/logs/cgdnet_hard_gate_10A_2026_2028_gpu0.log`.
- This is a baseline-first step only. No RCPS result will be run on CGDNet until hard CE is stable enough to pass the parity gate.

### Iteration 57 CGDNet Baseline Gate Paused: 2026-05-16 05:42 CST

- CGDNet hard-CE seed `2026` completed on the AMR-compatible RadioML2016.10A split with validation accuracy `54.60%` and test accuracy `54.33%`.
- Public/secondary AMR-Benchmark-style summaries place CGDNet on RadioML2016.10A around the high-50s rather than the low-50s; the observed result is therefore below the parity target by roughly `3 pp`.
- The RCPS config inherits the repository CGDNet backbone/init/schedule, so the likely causes are protocol differences, split differences, or a CGDNet-specific sensitivity rather than the RCPS loss.
- Action: stopped the remaining CGDNet gate queue after preserving seed `2026` metrics and logs. CGDNet is not removed from the study, but it is quarantined for parity debugging and will not be used as RCPS evidence until the hard baseline is repaired or the gap is explained.

### Iteration 58 MCformer Baseline Gate Launched: 2026-05-16 05:43 CST

- GPU0 was reassigned to `MCformer + RadioML2016.10A_AMR + hard CE` baseline parity gate for seeds `2026/2027/2028`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_amr_split_strong_v2`.
- Log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_gate_10A_2026_2028_gpu0.log`.
- Rationale: MCformer gives a transformer/attention-family baseline, which is more suitable for the main TPAMI evidence chain than a parity-failed CGDNet run.

### Iteration 59 GRU2 Ensemble Posterior Three-Seed Confirmation: 2026-05-16 07:11 CST

- GRU2 `RCPS-PosteriorBase` with a three-teacher validation posterior base completed seeds `2026/2027/2028` on the AMR-compatible RadioML2016.10A split.
- Method label: `rcps-posterior-ens3_retmax0p5_g2p0`; posterior base: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/gru2_ensemble3_reliability_base.npz`.
- Three-seed test deltas versus hard CE:
  - mean accuracy `+2.0523 pp`;
  - mean NLL `-0.0252`;
  - mean ECE `-0.0101`;
  - mean Brier `-0.0226`.
- Reliability-bin behavior:
  - high-SNR mean accuracy `+3.9364 pp`, NLL `-0.0459`, ECE `-0.0002`, Brier `-0.0399`;
  - low-SNR mean accuracy `-0.7778 pp`, NLL `+0.0069`, ECE `+0.0061`, Brier `+0.0032`.
- Interpretation: this is a clean second-model-family confirmation that posterior-base RCPS can improve aggregate accuracy and posterior quality without changing the backbone. The low-SNR tradeoff prevents a blanket claim that every reliability bin improves. The theory should emphasize reliability-conditioned posterior allocation and uncertainty alignment, with aggregate and high-reliability accuracy gains reported as empirical benefits rather than guaranteed consequences.
- Summary CSV: `/home/citybuster/Data/RCPS/work_dirs/gru2_main_posterior_ens3_400ep/summary/gru2_ens3_3seed_compare.csv`.

### Iteration 60 MCformer/FastMLDNN Baseline Gates Continue: 2026-05-16 07:13 CST

- MCformer hard-CE seed `2026` completed on the AMR-compatible RadioML2016.10A split and exported validation/test predictions.
- MCformer seed `2026` test metrics: accuracy `58.15%`, NLL `1.1062`, ECE `0.0136`, Brier `0.4908`. Seed `2027` is running normally on GPU0.
- GPU1 was reassigned to `FastMLDNN + RadioML2016.10A_AMR + hard CE` baseline parity gate for seeds `2026/2027/2028`.
- FastMLDNN is a baseline-only step at this stage. No RCPS modification will be treated as evidence until its hard CE baseline is stable enough or any gap to external/reference behavior is explained.

### Iteration 61 MCformer Hard-CE Three-Seed Baseline Gate: 2026-05-16 08:42 CST

- MCformer hard CE completed seeds `2026/2027/2028` on the AMR-compatible RadioML2016.10A split.
- Three-seed test metrics:
  - mean accuracy `58.1795%` with std `0.1043`;
  - mean NLL `1.1058`;
  - mean ECE `0.0134`;
  - mean Brier `0.4906`.
- Per-seed test accuracy: `58.15%`, `58.30%`, `58.09%`.
- Interpretation: MCformer is highly stable under the current protocol and can serve as an attention/transformer-family baseline, but it is not a stronger baseline than PETCGDNN or the existing MLDNN anchor. Any RCPS result on MCformer should be interpreted as cross-family stability evidence rather than the central strongest-result table.
- Summary CSV: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_amr_split_strong_v2/summary/mcformer_hard_ce_3seed_summary.csv`.

### Iteration 62 FastMLDNN Parity Correction: 2026-05-16 08:55 CST

- Stopped the active `FastMLDNN` hard-CE gate in `baseline_gate_amr_split_strong_v2` because the maintained config reproduced the same parity-failed behavior as the previous diagnostic run.
- Exported the current best seed-2026 checkpoint (`best_accuracy_top1_epoch_129.pth`) before stopping: test accuracy `53.2068`, test NLL `1.3988`, ECE `0.1338`, Brier `0.5817`.
- Root cause identified from config comparison: the maintained hard-CE config uses `dp=0.5`, `beta=0`, and plain CE, whereas the paper-style FastMLDNN configuration uses low dropout (`dp=0.07`), class-distance regularization (`beta/balance=0.5`), FocalLoss, and a lower Adam learning rate. The current hard-CE FastMLDNN config is therefore marked parity-failed and excluded from RCPS evidence.
- Added a paper-like diagnostic config: `configs/rcps/parity/fastmldnn_paperlike_focal-beta_iq-ap-snr-deepsig-201610A.py`.

### Iteration 63 MCformer Paired Control Launched: 2026-05-16 09:10 CST

- `MCformer + RCPS-PosteriorBase ensemble` seed 2026 is healthy and reached validation accuracy `58.2841` by epoch 56, already matching or slightly exceeding the hard-CE three-seed test mean (`58.1795`). This is an early training-efficiency signal only; final claims require exported test metrics.
- Stopped the FastMLDNN paper-like diagnostic at epoch 20 after exporting validation/test metrics. Test accuracy was `28.9364`; the quick parity fix is insufficient. FastMLDNN remains quarantined until the old paper FMLNet/init/head details are reproduced more faithfully.
- Reassigned GPU1 to `MCformer + Static LS (0.1)` seeds `2026/2027/2028` so the MCformer RCPS run has a same-backbone paired smoothing baseline.

### Iteration 64 MCformer RCPS Seed-2026 Landed: 2026-05-16 10:08 CST

- `MCformer + RCPS-PosteriorBase ensemble` seed 2026 completed and exported validation/test reliability metrics.
- Test comparison against same-seed hard CE: overall accuracy `+0.2568 pp`, NLL `+0.0050`, ECE `-0.0017`, Brier `+0.0006`.
- High-SNR comparison: accuracy `+0.8273 pp`, NLL `-0.0034`, ECE `+0.0027`, Brier `-0.0015`.
- Low-SNR comparison: accuracy `-0.8182 pp`, NLL `+0.0152`, ECE `+0.0078`, Brier `+0.0059`.
- Interpretation: this seed supports modest aggregate/high-reliability benefits and slight overall calibration improvement, but does not support a blanket low-SNR improvement claim. The manuscript should keep the theory centered on reliability-conditioned posterior allocation and report low-SNR tradeoffs honestly.
- `MCformer + Static LS (0.1)` is still running as the same-backbone smoothing control.

### Iteration 65 MCformer Seed-2026 Three-Way Control: 2026-05-16 10:23 CST

- `MCformer + Static LS (0.1)` seed 2026 completed and was compared with same-seed hard CE and RCPS-PosteriorBase.
- Static LS vs hard CE: overall accuracy `+0.0523 pp`, NLL `+0.0295`, ECE `+0.0161`, Brier `+0.0017`. Low-SNR metrics improve, but overall/high-SNR calibration worsens.
- RCPS-PosteriorBase vs hard CE: overall accuracy `+0.2568 pp`, ECE `-0.0017`, high-SNR accuracy `+0.8273 pp`, high-SNR NLL/Brier improve, while low-SNR metrics worsen.
- Interpretation: on this seed, RCPS is stronger than static uniform smoothing on overall accuracy/ECE and high-reliability behavior, but static smoothing is more conservative at low reliability. The paper should separate low-reliability calibration from high-reliability retention instead of treating all reliability bins as one monotone story.

### Iteration 66 MCformer Three-Seed Paired Control Completed: 2026-05-16 11:58 CST

- `MCformer + RadioML2016.10A_AMR` completed the three-way paired control for `Hard CE`, `Static LS (0.1)`, and `RCPS-PosteriorBase ensemble` on seeds `2026/2027/2028`.
- Summary CSVs:
  - comparison rows: `/home/citybuster/Data/RCPS/work_dirs/mcformer_main_posterior_ens3_400ep/summary/mcformer_hard_static_rcps_3seed_compare.csv`;
  - aggregate mean/std: `/home/citybuster/Data/RCPS/work_dirs/mcformer_main_posterior_ens3_400ep/summary/mcformer_hard_static_rcps_3seed_aggregate.csv`.
- Three-seed aggregate versus hard CE:
  - `RCPS-PosteriorBase`: overall accuracy `+0.5371 pp`, NLL `-0.0027`, ECE `-0.0035`, Brier `-0.0030`; high-SNR accuracy `+1.0364 pp`, NLL `-0.0082`, Brier `-0.0054`; low-SNR accuracy `-0.8258 pp`, NLL `+0.0104`, ECE `+0.0063`, Brier `+0.0043`.
  - `Static LS (0.1)`: overall accuracy `+0.0424 pp`, NLL `+0.0285`, ECE `+0.0171`, Brier `+0.0016`; low-SNR NLL `-0.0223`, ECE `-0.0227`, Brier `-0.0078`; high-SNR accuracy `-0.3848 pp`, NLL `+0.0596`, ECE `+0.0321`, Brier `+0.0077`.
- Interpretation: MCformer provides a clean attention-family confirmation that posterior-base RCPS can improve aggregate and high-reliability behavior while keeping the backbone unchanged. However, the low-SNR degradation means the current posterior-base target is not the final TPAMI method. Static uniform smoothing has the opposite failure mode: it improves low-reliability calibration but damages high-reliability behavior and overall NLL/ECE.
- Next algorithmic adjustment: test a reliability-gated hybrid RCPS variant that uses stronger prior/uniform mass at very low reliability while preserving posterior-base allocation and retention at mid/high reliability. The theory should describe this as reliability-conditioned posterior approximation with retention constraints, not as a theorem that a single uniform smoothing schedule improves every reliability bin.

### Iteration 67 MCformer Hybrid RCPS Pilot Launched: 2026-05-16 12:01 CST

- Launched two single-seed (`2026`) MCformer hybrid RCPS pilots on the AMR-compatible RadioML2016.10A split after Iteration 66 showed complementary failure modes for `RCPS-PosteriorBase` and `Static LS`.
- Common setup: same MCformer backbone, AMR-compatible split, posterior table `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/mcformer_ensemble3_reliability_base.npz`, `prior_blend=1.0`, `gamma=2.0`, `retain_min=0.8`, max epochs `400`, no worker multiprocessing.
- GPU0 variant: `rcps-hybrid-prior1_eps0p1_g2_retain0p8`, work root `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_400ep`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_eps0p1_2026_gpu0.log`.
- GPU1 variant: `rcps-hybrid-prior1_eps0p3_g2_retain0p8`, work root `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p3_g2_400ep`, log `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_eps0p3_2026_gpu1.log`.
- Evaluation rule: compare seed `2026` against hard CE, Static LS, and RCPS-PosteriorBase. If a hybrid variant recovers low-SNR NLL/ECE/Brier without losing the aggregate/high-SNR gains, promote it to three-seed confirmation. Otherwise, keep it as a negative algorithmic iteration.

### Iteration 68 MCformer Hybrid eps0.1 Promoted to Three-Seed Confirmation: 2026-05-16 13:02 CST

- `MCformer + rcps-hybrid-prior1_eps0p1_g2_retain0p8` seed `2026` completed and was compared against same-seed hard CE, Static LS, and RCPS-PosteriorBase.
- Seed-2026 deltas versus hard CE: overall accuracy `+0.1773 pp`, NLL `-0.0051`, ECE `-0.0026`, Brier `-0.0006`; low-SNR NLL `-0.0056`, ECE `-0.0032`, Brier `-0.0004`; high-SNR accuracy `+0.3545 pp`, NLL `-0.0029`, ECE `-0.0027`, Brier `-0.0002`.
- Interpretation: unlike the stronger posterior-base variant, this hybrid setting improves low-reliability posterior metrics while retaining aggregate and high-reliability gains on the first seed. It is therefore promoted to seeds `2027/2028` for confirmation.
- Launched seeds `2027/2028` on GPU0 with log `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_eps0p1_2027_2028_gpu0.log`.

### Iteration 69 MCformer Hybrid eps0.3 Pilot and eps0.2 Launch: 2026-05-16 13:09 CST

- `MCformer + rcps-hybrid-prior1_eps0p3_g2_retain0p8` seed `2026` completed. It improved low-SNR NLL/ECE/Brier versus hard CE, but was less balanced than `eps0.1` on aggregate NLL and high-SNR ECE.
- Seed-2026 deltas versus hard CE for `eps0.3`: overall accuracy `+0.1364 pp`, NLL `-0.0002`, ECE `-0.0023`, Brier `-0.0012`; low-SNR NLL `-0.0054`, ECE `-0.0046`, Brier `-0.0012`; high-SNR accuracy `+0.5818 pp`, NLL `-0.0048`, ECE `+0.0006`, Brier `-0.0024`.
- Decision: do not promote `eps0.3` yet; keep it as a useful pilot point. `eps0.1` remains the three-seed confirmation candidate because it improves aggregate NLL/ECE/Brier and low/high reliability metrics more consistently on seed `2026`.
- Launched a midpoint pilot `rcps-hybrid-prior1_eps0p2_g2_retain0p8` on GPU1 to check whether `epsilon_max=0.2` provides a better accuracy/calibration tradeoff. Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p2_g2_400ep`; log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_eps0p2_2026_gpu1.log`.


## Iteration 70 - MCformer hybrid eps0.2 promoted to three-seed confirmation (2026-05-16 14:15:01 CST)

- Foreground monitoring confirmed `MCformer + rcps-hybrid-prior1_eps0p2_g2_retain0p8` seed `2026` completed without export or dataloader errors. Validation/test prediction export succeeded with `num_workers=0`.
- Seed-2026 comparison against hard CE and prior pilots was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p2_g2_400ep/summary/mcformer_seed2026_hard_static_posterior_hybrid_eps_compare.csv`.
- Seed-2026 deltas versus hard CE: overall accuracy `+0.2068 pp`; low-SNR NLL `-0.0126`; high-SNR ECE `-0.0032`. This is a stronger single-seed balance than `eps0.1` on low-SNR NLL and high-SNR ECE, while remaining less aggressive than the pure posterior variant.
- Decision: promote `eps0.2` to seeds `2027` and `2028` on GPU1 for confirmation, while `eps0.1` seed `2028` continues on GPU0. Log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_eps0p2_2027_2028_gpu1.log`.
- No paper conclusion is updated yet. The current decision only expands a promising pilot to a paired three-seed test.


## Iteration 71 - MCformer hybrid eps0.1 three-seed result (2026-05-16 15:18:05 CST)

- `MCformer + rcps-hybrid-prior1_eps0p1_g2_retain0p8` completed all three seeds on `RadioML.2016.10A` with validation/test prediction export and reliability-bin analysis.
- Paired comparison against hard CE, static label smoothing, and posterior-base RCPS was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_400ep/summary/mcformer_hard_static_posterior_hybrid_eps0p1_3seed_compare.csv`.
- Aggregate summary was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_400ep/summary/mcformer_hard_static_posterior_hybrid_eps0p1_3seed_aggregate.csv`.
- Mean deltas versus hard CE for the hybrid eps0.1 method: overall accuracy `+0.2970 pp`, NLL `-0.0064`, ECE `-0.0022`, Brier `-0.0018`; low-SNR accuracy `+0.2955 pp`, NLL `-0.0061`, ECE `-0.0044`, Brier `-0.0011`; high-SNR accuracy `+0.2333 pp`, NLL `-0.0030`, ECE `-0.0016`, Brier `-0.0006`.
- Interpretation: `eps0.1` is currently the most balanced MCformer variant. It avoids the low-reliability degradation seen in pure posterior-base RCPS and avoids the high-reliability NLL/ECE damage of static label smoothing. This is still one model and one dataset, so it is a candidate result, not a final paper claim.
- Ongoing: `eps0.2` seeds `2027/2028` are running to test whether the midpoint gives a stronger tradeoff.

## Iteration 72 - PETCGDNN hybrid eps0.2 second-family pilot launched (2026-05-16 15:32:10 CST)

- Foreground monitoring confirmed `MCformer + rcps-hybrid-prior1_eps0p2_g2_retain0p8` seed `2027` is still running normally on GPU1. At launch time it had reached around epoch `148`, with best validation accuracy `58.8023`, and no `Traceback`, CUDA OOM, file-handle, or dataloader errors in the log.
- GPU0 was idle, so a conservative second-model-family pilot was launched for `PETCGDNN Keras-init + rcps-hybrid-prior1_eps0p2_g2_retain0p8` on `RadioML.2016.10A_AMR`, seed `2026` only.
- Setup matches the MCformer hybrid logic: same Keras-init PETCGDNN backbone and AMR-compatible split as the hard-CE gate, posterior table `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/petcgdnn_kerasinit_seed2026_reliability_base.npz`, `prior_blend=1.0`, `epsilon.max=0.2`, `gamma=2.0`, `retain_min=0.8`, max epochs `400`, and all dataloaders set to `num_workers=0`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/petcgdnn_kerasinit_hybrid_prior1_eps0p2_g2_400ep`; log: `/home/citybuster/Data/RCPS/work_dirs/logs/petcgdnn_kerasinit_hybrid_eps0p2_seed2026_gpu0.log`.
- This is a single-seed pilot, not a paper result. It will be promoted only if paired comparison against PETCGDNN hard CE, static LS, and posterior-base results shows a better reliability tradeoff.


## Iteration 73 - MCformer hybrid eps0.2 seed-2027 comparison landed (2026-05-16 15:43:30 CST)

- `MCformer + rcps-hybrid-prior1_eps0p2_g2_retain0p8` seed `2027` completed training, validation/test prediction export, and reliability-bin analysis without runtime errors. The sequential runner then started seed `2028` automatically on GPU1.
- Paired comparison was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p2_g2_400ep/summary/mcformer_seed2027_hard_static_posterior_hybrid_eps_compare.csv`.
- Versus same-seed hard CE, `eps0.2` improved overall accuracy by `+0.4114 pp`, NLL by `-0.0069`, ECE by `-0.0051`, and Brier by `-0.0031`.
- In the low-SNR bins, `eps0.2` reduced NLL by `-0.0076`, ECE by `-0.0076`, and Brier by `-0.0018`, while low-SNR accuracy changed by `-0.1742 pp`.
- In the high-SNR bins, `eps0.2` improved accuracy by `+0.7273 pp`, NLL by `-0.0035`, ECE by `-0.0019`, and Brier by `-0.0026`.
- Interpretation: seed `2027` strengthens the case that the hybrid target can combine the low-reliability calibration behavior of smoothing with the high-reliability retention of posterior-base RCPS. The low-SNR accuracy dip remains a tradeoff to track in the three-seed aggregate.


## Iteration 74 - PETCGDNN hybrid eps0.2 pilot completed and not promoted (2026-05-16 16:17:00 CST)

- `PETCGDNN Keras-init + rcps-hybrid-prior1_eps0p2_g2_retain0p8` seed `2026` completed training, validation/test export, and reliability-bin analysis without runtime errors. Best validation accuracy was `60.5227` at epoch `64`; test accuracy was `60.45%`.
- Paired comparison was written to `/home/citybuster/Data/RCPS/work_dirs/petcgdnn_kerasinit_hybrid_prior1_eps0p2_g2_400ep/summary/petcgdnn_kerasinit_seed2026_hard_static_retention_posterior_hybrid_compare.csv`.
- Versus same-seed hard CE, the hybrid improved overall accuracy by `+0.4591 pp`, NLL by `-0.0079`, ECE by `-0.0054`, and Brier by `-0.0064`; high-SNR accuracy improved by `+0.7455 pp`, NLL by `-0.0156`, ECE by `-0.0047`, and Brier by `-0.0122`.
- Low-SNR posterior quality worsened for this seed: low-SNR NLL `+0.0058`, ECE `+0.0091`, and Brier `+0.0044`, despite a small low-SNR accuracy gain of `+0.0682 pp`.
- Existing `PETCGDNN Keras-init + RCPS-PosteriorBase` remains stronger on the same seed: overall accuracy `+1.0682 pp`, NLL `-0.0311`, ECE `-0.0167`, and Brier `-0.0202` versus hard CE, with low-SNR NLL/ECE/Brier also improved.
- Decision: do not promote PETCGDNN hybrid eps0.2 to three seeds. Keep it as a negative/boundary algorithmic iteration and retain PosteriorBase as the PETCGDNN candidate unless later evidence changes the tradeoff.


## Iteration 75 - MCformer RadioML2016.10B hard-CE baseline gate launched (2026-05-16 16:18:00 CST)

- GPU0 was released after the PETCGDNN hybrid pilot, so the next baseline-first step was launched on the second AMC dataset: `MCformer + RadioML2016.10B_AMR + hard CE`, seed `2026`.
- This is a baseline gate, not an RCPS run. The goal is to establish a trustworthy MCformer baseline on `RadioML2016.10B` before constructing any 10B posterior tables or RCPS variants.
- Config base: `configs/rcps/mcformer/mcformer_hard-ce_iq-snr-deepsig-201610A.py`; data root overridden to `/home/citybuster/Data/RCPS/processed/amr_compatible/RadioML.2016.10B`; max epochs `400`; all dataloaders use `num_workers=0`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep`; log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_ce_10B_seed2026_gpu0.log`.
- Existing `MCformer + RadioML2016.10A` hybrid eps0.2 seed `2028` continues on GPU1.


## Iteration 76 - MCformer hybrid eps0.2 three-seed aggregate completed (2026-05-16 17:12:00 CST)

- `MCformer + rcps-hybrid-prior1_eps0p2_g2_retain0p8` completed seeds `2026/2027/2028` on `RadioML.2016.10A_AMR` with validation/test export and reliability-bin analysis.
- Three-seed comparison rows were written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p2_g2_400ep/summary/mcformer_hard_static_posterior_hybrid_eps0p2_3seed_compare.csv`.
- Aggregate mean/std was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p2_g2_400ep/summary/mcformer_hard_static_posterior_hybrid_eps0p2_3seed_aggregate.csv`.
- Mean deltas versus hard CE for `eps0.2`: overall accuracy `+0.2614 pp`, NLL `-0.0058`, ECE `-0.0031`, Brier `-0.0018`; low-SNR accuracy `-0.1515 pp`, NLL `-0.0113`, ECE `-0.0075`, Brier `-0.0029`; high-SNR accuracy `+0.4545 pp`, NLL `-0.0009`, ECE `-0.0012`, Brier `-0.0003`.
- Interpretation: `eps0.2` is a useful calibration-oriented ablation. It improves low-reliability NLL/ECE/Brier more strongly than `eps0.1`, but introduces a small low-SNR accuracy loss and weaker aggregate accuracy than `eps0.1`. The most balanced MCformer candidate remains `eps0.1`; `eps0.2` should be reported as a tradeoff/ablation rather than the default method.


## Iteration 77 - MCformer RadioML2016.10B hard-CE seed-2027 launched (2026-05-16 17:13:00 CST)

- After MCformer 10A hybrid eps0.2 completed and released GPU1, the `RadioML2016.10B_AMR + MCformer + hard CE` baseline gate was extended to seed `2027` on GPU1.
- Seed `2026` continues on GPU0; seed `2027` uses the same config, data root, max epochs, and dataloader settings.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep`; seed-2027 log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_ce_10B_seed2027_gpu1.log`.
- This is still baseline-first work. No 10B RCPS run will be launched until hard-CE baseline metrics are exported and checked.


## Iteration 78 - MCformer RadioML2016.10B hard-CE seed-2026 completed and seed-2028 launched (2026-05-16 17:24:10 CST)

- `MCformer + RadioML2016.10B_AMR + hard CE` seed `2026` completed training, validation/test export, and reliability analysis without runtime errors. Validation accuracy from the exported best checkpoint was `58.57%`; test accuracy was `58.44%`.
- The metrics landed at `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep/metrics/deepsig201610B_amr_mcformer_hard-ce_seed2026_test.csv`.
- Seed `2027` continues on GPU1. GPU0 was released after seed `2026`, so seed `2028` was launched with the same data root, config, and dataloader settings.
- Seed-2028 log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_ce_10B_seed2028_gpu0.log`.
- The 10B work remains baseline-gate only; no 10B RCPS run is launched until the hard-CE three-seed baseline is complete and inspected.


## Iteration 79 - MCformer RadioML2016.10B hard-CE seed-2027 completed (2026-05-16 17:57:00 CST)

- `MCformer + RadioML2016.10B_AMR + hard CE` seed `2027` completed training, validation/test prediction export, and reliability-bin analysis without runtime errors.
- Exported best-checkpoint validation accuracy was `58.46%`; test accuracy was `58.69%`.
- Metrics landed at `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep/metrics/deepsig201610B_amr_mcformer_hard-ce_seed2027_test.csv`.
- Seed `2028` remains running on GPU0 and reached validation best `58.7550%` by epoch `77` during the foreground patrol. The 10B stage remains a baseline gate; no RCPS variant is launched until the three-seed hard baseline is complete and summarized.


## Iteration 80 - MCformer RadioML2016.10B hard-CE three-seed baseline completed (2026-05-16 18:20:00 CST)

- `MCformer + RadioML2016.10B_AMR + hard CE` completed seeds `2026/2027/2028` with validation/test prediction export and reliability-bin analysis.
- Test accuracies were `58.44%`, `58.69%`, and `58.64%`; the three-seed mean/std is `58.5883 +/- 0.1338`.
- Aggregate metrics: NLL `0.9891`, ECE `0.0134`, Brier `0.4565`.
- Per-seed summary: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep/summary/mcformer_hard_ce_10B_3seed_per_seed.csv`.
- Reliability-bin aggregate: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_400ep/summary/mcformer_hard_ce_10B_3seed_aggregate.csv`.
- Interpretation: the second AMC dataset baseline gate is stable enough to proceed to matched RCPS construction. The next step is to build a validation-only posterior/confusion base for `RadioML2016.10B_AMR` and launch matched MCformer RCPS-Hybrid eps0.1 runs, without changing backbone, split, seed, optimizer, or epoch budget.


## Iteration 81 - RadioML2016.10B MCformer class-count mismatch diagnosed (2026-05-16 18:28:00 CST)

- The first `MCformer + RadioML2016.10B_AMR + hard CE` gate used the 10A MCformer base config without overriding `model.backbone.num_classes`.
- Audit showed `RadioML2016.10B_AMR` has `10` classes (`classes` in exported prediction files are length 10 and labels are `0..9`), while the model emitted `11` probabilities because the 10A config sets `num_classes=11`.
- The resulting three-seed 10B hard-CE metrics are therefore quarantined as diagnostic-only and must not be used as paper evidence or as an RCPS teacher.
- The invalid posterior table was moved to `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610B_amr/mcformer_hard-ce_3seed_reliability_base.invalid_numcls11.npz`.
- Corrected baseline gate is restarted under `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep` with `model.backbone.num_classes=10`, same data split, same seeds, same optimizer, same epoch budget, and `num_workers=0`.


## Iteration 82 - Corrected MCformer RadioML2016.10B hard-CE numcls10 gate launched (2026-05-16 18:32:00 CST)

- Corrected `MCformer + RadioML2016.10B_AMR + hard CE` baseline runs were launched with explicit `model.backbone.num_classes=10`.
- Seed `2026` runs on GPU0 and seed `2027` runs on GPU1. Seed `2028` will be launched after one GPU is released.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep`.
- Logs: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_ce_10B_numcls10_seed2026_gpu0.log` and `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hard_ce_10B_numcls10_seed2027_gpu1.log`.
- This correction is required before any 10B RCPS teacher/base construction. The previous 11-output 10B baseline remains quarantined.


## Iteration 83 - Corrected MCformer RadioML2016.10B seed-2027 completed and seed-2028 launched (2026-05-16 19:14:00 CST)

- Corrected `MCformer + RadioML2016.10B_AMR + hard CE + num_classes=10` seed `2027` completed validation/test export and reliability analysis.
- Exported validation accuracy was `58.87%`; test accuracy was `58.96%`.
- Metrics landed at `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep/metrics/deepsig201610B_amr_mcformer_hard-ce_seed2027_test.csv`.
- Seed `2026` remains running on GPU0. Seed `2028` was launched on GPU1 with the same corrected `num_classes=10` override and identical training/data settings.


## Iteration 84 - Corrected MCformer RadioML2016.10B seed-2026 completed (2026-05-16 19:25:00 CST)

- Corrected `MCformer + RadioML2016.10B_AMR + hard CE + num_classes=10` seed `2026` completed validation/test export and reliability-bin analysis.
- Exported validation accuracy was `58.35%`; test accuracy was `58.30%`.
- Overall test metrics: NLL `0.9892`, ECE `0.0120`, Brier `0.4577`.
- Metrics landed at `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep/metrics/deepsig201610B_amr_mcformer_hard-ce_seed2026_test.csv`.
- Seed `2028` remains running on GPU1 and will complete the corrected three-seed baseline gate.


## Iteration 85 - Corrected MCformer RadioML2016.10B hard-CE numcls10 gate completed (2026-05-16 20:08:00 CST)

- Corrected `MCformer + RadioML2016.10B_AMR + hard CE + num_classes=10` seed `2028` completed validation/test export and reliability-bin analysis.
- Exported validation accuracy was `58.18%`; test accuracy was `58.39%`.
- Sanity check confirmed all corrected test prediction files emit probability tensors of shape `(40000, 10)` and class lists of length `10`; the earlier 11-output run remains quarantined and is not used as evidence.
- Corrected three-seed test accuracies are `58.30%`, `58.96%`, and `58.39%`; the mean/std is `58.5517 +/- 0.3565`.
- Corrected aggregate metrics: NLL `0.9908`, ECE `0.0117`, Brier `0.4574`.
- Per-seed summary: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep/summary/mcformer_hard_ce_10B_numcls10_3seed_per_seed.csv`.
- Reliability-bin aggregate: `/home/citybuster/Data/RCPS/work_dirs/mcformer_gate_10B_numcls10_400ep/summary/mcformer_hard_ce_10B_numcls10_3seed_aggregate.csv`.
- Interpretation: the corrected second-dataset MCformer baseline gate is stable enough for matched RCPS construction. The next step is to build a validation-only 10-class posterior base and launch `RCPS-Hybrid eps0.1` under identical backbone, data split, seeds, optimizer, and epoch budget.


## Iteration 86 - MCformer RadioML2016.10B RCPS-Hybrid numcls10 eps0.1 launched (2026-05-16 20:12:00 CST)

- Built a validation-only posterior base from corrected 10-class hard-CE validation predictions for seeds `2026/2027/2028`.
- Posterior base path: `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610B_amr/mcformer_hard-ce_numcls10_3seed_reliability_base.npz`.
- Sanity check: `base` shape `(20, 10, 10)`, `counts` shape `(20, 10)`, all class/bin counts are nonzero, and base rows sum to one up to floating-point tolerance.
- Launched matched `MCformer + RadioML2016.10B_AMR + RCPS-Hybrid eps0.1` runs with `model.backbone.num_classes=10`, `prior_blend=1.0`, `epsilon.max=0.1`, `gamma=2.0`, and `retain_min=0.8`.
- Seed `2026` runs on GPU0; seed `2027` runs on GPU1 and the same GPU1 script will continue to seed `2028` after seed `2027` completes.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_10B_numcls10_400ep`.
- Logs: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_10B_numcls10_eps0p1_seed2026_gpu0.log` and `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_10B_numcls10_eps0p1_seed2027_2028_gpu1.log`.
- This is the first matched second-dataset RCPS check after the corrected 10-class baseline gate; the previous 11-output 10B run remains excluded.


## Iteration 87 - MCformer RadioML2016.10B eps0.05 retention pilot launched (2026-05-16 21:12:00 CST)

- `RCPS-Hybrid eps0.1` completed test export for seeds `2026/2027` while seed `2028` continued running.
- Partial test metrics are mixed: seed `2026` improves accuracy/NLL/Brier but worsens ECE; seed `2027` improves ECE but loses accuracy/NLL/Brier.
- Decision: keep the planned `eps0.1` three-seed run unchanged, but use the released GPU0 for a conservative `eps0.05` pilot on seed `2026` to test whether weaker smoothing improves retention on `RadioML2016.10B_AMR`.
- Launched `MCformer + RadioML2016.10B_AMR + RCPS-Hybrid eps0.05 + num_classes=10` seed `2026` on GPU0.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p05_g2_10B_numcls10_400ep`.
- Log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcformer_hybrid_10B_numcls10_eps0p05_seed2026_gpu0.log`.
- This is an algorithmic iteration, not a replacement for the ongoing `eps0.1` confirmation. Promotion requires completed CSV comparison against hard CE.


## Iteration 88 - MCformer RadioML2016.10B eps0.1 completed and eps0.05 promoted (2026-05-16 22:03:00 CST)

- `MCformer + RadioML2016.10B_AMR + RCPS-Hybrid eps0.1 + num_classes=10` completed seeds `2026/2027/2028` with validation/test export and reliability-bin analysis.
- Prediction sanity check confirmed all three test probability tensors are `(40000, 10)` with `10` classes.
- Three-seed comparison versus corrected hard CE was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_10B_numcls10_400ep/summary/mcformer_10B_numcls10_hard_vs_hybrid_eps0p1_3seed_compare.csv`.
- Aggregate summary was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p1_g2_10B_numcls10_400ep/summary/mcformer_10B_numcls10_hard_vs_hybrid_eps0p1_3seed_aggregate.csv`.
- Mean deltas versus hard CE for `eps0.1`: overall accuracy `+0.0842 pp`, NLL `-0.0052`, ECE `-0.0002`, Brier `-0.0011`; low-SNR accuracy `+0.0361 pp`, NLL `-0.0051`, ECE `-0.0032`, Brier `-0.0002`; high-SNR accuracy `-0.4267 pp`, NLL `+0.0027`, ECE `-0.0037`, Brier `+0.0021`.
- Interpretation: `eps0.1` remains within the high-SNR retention constraint and improves aggregate/low-reliability posterior metrics, but high-SNR NLL/Brier and high-SNR accuracy weaken. It is useful evidence, not yet the cleanest 10B setting.
- The `eps0.05` seed `2026` pilot was cleaner: overall accuracy `+0.3500 pp`, NLL `-0.0080`, ECE `-0.0020`, Brier `-0.0029`; low-SNR accuracy `+0.4417 pp`; high-SNR NLL/ECE/Brier also improved with high-SNR accuracy `-0.1800 pp`.
- Decision: promote `eps0.05` to seeds `2027/2028` for confirmation. These two runs were launched on GPU0/GPU1 with the same corrected 10-class base and unchanged training setup.


## Iteration 89 - MCformer RadioML2016.10B eps0.05 three-seed aggregate completed (2026-05-16 23:04:00 CST)

- `MCformer + RadioML2016.10B_AMR + RCPS-Hybrid eps0.05 + num_classes=10` completed seeds `2026/2027/2028` with validation/test export and reliability-bin analysis.
- Prediction sanity check confirmed all three test probability tensors are `(40000, 10)` with `10` classes.
- Three-seed comparison versus corrected hard CE was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p05_g2_10B_numcls10_400ep/summary/mcformer_10B_numcls10_hard_vs_hybrid_eps0p05_3seed_compare.csv`.
- Aggregate summary was written to `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p05_g2_10B_numcls10_400ep/summary/mcformer_10B_numcls10_hard_vs_hybrid_eps0p05_3seed_aggregate.csv`.
- Mean deltas versus hard CE for `eps0.05`: overall accuracy `+0.0458 pp`, NLL `-0.0040`, ECE `-0.0006`, Brier `-0.0009`; low-SNR accuracy `+0.0389 pp`, NLL `-0.0036`, ECE `-0.0024`, Brier `+0.0000`; high-SNR accuracy `-0.0667 pp`, NLL `-0.0006`, ECE `-0.0038`, Brier `+0.0002`.
- Comparison with `eps0.1`: `eps0.1` gives slightly larger overall accuracy/NLL/Brier gains but weakens high-SNR accuracy and high-SNR NLL/Brier; `eps0.05` is the cleaner retention-oriented 10B candidate.
- Interpretation: the 10B evidence supports reliability-conditioned smoothing as a modest but consistent posterior-quality improvement when `epsilon` is selected conservatively under a high-reliability retention constraint. The paper should present this as calibration/uncertainty alignment with validation-constrained retention, not as a theorem-level guarantee of large accuracy gains.


## Iteration 90 - MCformer RadioML2016.10B eps0.05 confirmation finalized (2026-05-16 23:05:00 CST)

- Completed all promoted `MCformer + RadioML2016.10B_AMR + RCPS-Hybrid eps0.05 + num_classes=10` seeds `2026/2027/2028`.
- All test prediction tensors were checked as `(40000, 10)` with `10` classes.
- The final three-seed aggregate is stored at `/home/citybuster/Data/RCPS/work_dirs/mcformer_hybrid_prior1_eps0p05_g2_10B_numcls10_400ep/summary/mcformer_10B_numcls10_hard_vs_hybrid_eps0p05_3seed_aggregate.csv`.
- Key deltas versus corrected hard CE: overall accuracy `+0.0458 pp`, NLL `-0.0040`, ECE `-0.0006`, Brier `-0.0009`; low-SNR accuracy `+0.0389 pp`, NLL `-0.0036`, ECE `-0.0024`; high-SNR accuracy `-0.0667 pp`, high-SNR NLL `-0.0006`, high-SNR ECE `-0.0038`.
- Compared with `eps0.1`, `eps0.05` is the cleaner retention-constrained choice for 10B; compared with 10A, the preferred epsilon appears dataset-dependent. This supports a validation-constrained calibration view of `epsilon`, not a fixed universal smoothing constant.
- Next evidence step: inspect `RadioML2018.01A` historical checkpoints and/or launch a carefully staged third-dataset baseline gate, while continuing to avoid unsupported claims.


## Iteration 91 - RadioML2018.01A checkpoint audit and current-code gate launch (2026-05-16 23:24:00 CST)

- Audited available historical `RadioML2018.01A` checkpoints before using them as evidence. They are not usable as reliable baselines in the current code path: the historical MLDNN checkpoint cannot be read by PyTorch, and the historical CNN4/DSCLDNN checkpoints show state-dict incompatibility with near-random validation accuracy around `4%`.
- Decision: quarantine those historical audits as diagnostics only. They must not enter paper tables or RCPS comparisons.
- Launched current-code `RadioML2018.01A` hard-CE baseline gates instead, using the present configs and explicit data root `/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2018.01A`.
- Running jobs: `MCformer + hard CE + seed 2026` on GPU0 and `CGDNet + hard CE + seed 2026` on GPU1, each with `max_epochs=400`, `num_workers=0`, and unchanged train/validation/test split files.
- At launch monitoring, both processes were alive with no `Traceback`, CUDA OOM, file-handle error, or shape error. Logs had not yet reached epoch output, consistent with large cached dataset preparation.
- This is a baseline-first third-dataset gate. No RCPS run will be launched for `RadioML2018.01A` until at least one current-code baseline produces sane validation/test metrics.


## Iteration 92 - RadioML2018.01A full-gate stopped and UCSD/RML22 gate launched (2026-05-16 23:40:00 CST)

- Intervention was required on the attempted full `RadioML2018.01A` current-code gate.
- `CGDNet + RadioML2018.01A + hard CE` failed immediately with a backbone shape error: `shape '[-1, 50, 472]' is invalid for input of size 81120000`. This is a model/config compatibility issue for the current 2018 pipeline, not an RCPS result.
- `MCformer + RadioML2018.01A + hard CE` entered epoch 1, but the logged ETA was about `13 days` for the configured `400`-epoch run because the full 2018 split has `3195` train iterations per epoch. The run was stopped before wasting resources.
- Decision: `RadioML2018.01A` remains an important benchmark, but it needs a separate feasible protocol such as a faster compatible backbone, a reduced/validated epoch budget, or a reliability-balanced subset. The stopped logs are diagnostics only and are not used as paper evidence.
- To keep the AMC evidence chain moving, launched a third-dataset baseline gate on `UCSD/RML22`, which is much smaller (`2.8G`) and has explicit SNR metadata in `train/validation/test.json`.
- Launched `MCformer + UCSD/RML22 + hard CE + seed 2026` on GPU0 and `PETCGDNN + UCSD/RML22 + hard CE + seed 2026` on GPU1 with the same current-code training/export/analysis chain, `max_epochs=400`, and `num_workers=0` for stability.
- No RCPS comparison will be run on UCSD/RML22 until these hard-CE baselines produce sane validation/test metrics.


## Iteration 93 - UCSD/RML22 strong-model gate failed and lightweight screen launched (2026-05-16 23:53:00 CST)

- `PETCGDNN + UCSD/RML22 + hard CE` failed with a kernel/input-size incompatibility: `Kernel size: (2 x 8). Kernel size can't be greater than actual input size`. This is a config/backbone compatibility issue for the current UCSD pipeline.
- `MCformer + UCSD/RML22 + hard CE` ran quickly, but validation accuracy stayed exactly `10.0000%` and loss stayed near `2.3026` through epoch `21`. The run was stopped as a non-learning baseline.
- Decision: do not use either failed strong-model UCSD attempt as evidence. Treat them as baseline-gate diagnostics.
- Launched a 30-epoch UCSD/RML22 screen with more stable candidate backbones: `CNN4 + hard CE + seed 2026` on GPU0 and `GRU2 + hard CE + seed 2026` on GPU1.
- The screen goal is only to find a backbone whose UCSD data/label/shape path actually learns. If one passes, it can be promoted to a formal baseline gate and later to a matched RCPS comparison.


## Iteration 94 - UCSD/RML22 CNN4 baseline gate promoted (2026-05-17 00:11:00 CST)

- UCSD/RML22 screen outcome: `CNN4 + hard CE + seed 2026` is the first stable candidate. It learned normally and produced validation accuracy `53.48%` and test accuracy `53.37%` at the best checkpoint (`epoch 22`) under a 30-epoch budget.
- Reliability-bin metrics were exported to `/home/citybuster/Data/RCPS/work_dirs/ucsd_rml22_baseline_screen_30ep/metrics/ucsd_rml22_cnn4_hard-ce-screen_seed2026_test.csv`.
- Other UCSD candidates are not usable without adaptation: `GRU2` failed with `input.size(-1) must be equal to input_size`, `FastMLDNN` failed with a batch-size mismatch, `PETCGDNN` failed with kernel/input-size mismatch, and `MCLDNN` remained at `10%` accuracy through 15 epochs before being stopped.
- Decision: promote `CNN4` as the feasible UCSD/RML22 baseline family for third-AMC-dataset supplementary evidence. This does not replace the strong-model evidence on 10A/10B.
- Launched `CNN4 + UCSD/RML22 + hard CE` seeds `2027` and `2028`, each with the same 30-epoch budget and export/analysis chain. If they match seed `2026`, the next step is a matched RCPS comparison on the same CNN4 backbone and budget.


## Iteration 95 - UCSD/RML22 CNN4 matched RCPS comparison completed (2026-05-17 01:30:00 CST)

- Completed the matched `CNN4 + UCSD/RML22` 30-epoch comparison for `Hard CE`, `Static LS`, and `RCPS-Retention` with seeds `2026/2027/2028`.
- Hard-CE seed `2026` comes from the promoted screen run; hard-CE seeds `2027/2028` come from the formal gate. Static LS and RCPS-Retention were run under the same CNN4 backbone, UCSD/RML22 split, epoch budget, optimizer schedule, export path, and reliability-bin analysis chain.
- Summary files were written under `/home/citybuster/Data/RCPS/work_dirs/ucsd_rml22_matched_30ep/summary`.
- Three-seed test aggregate: hard CE accuracy `48.4992 +/- 6.5126`, NLL `1.2705`, ECE `0.0250`, Brier `0.5769`; Static LS accuracy `48.9032 +/- 6.3672`, NLL `1.3150`, ECE `0.0491`, Brier `0.5827`; RCPS-Retention accuracy `49.8081 +/- 4.9086`, NLL `1.2508`, ECE `0.0329`, Brier `0.5685`.
- Mean deltas versus hard CE: Static LS gives accuracy `+0.4040 pp` but worsens NLL by `+0.0445`, ECE by `+0.0241`, and Brier by `+0.0058`; RCPS-Retention gives accuracy `+1.3089 pp`, NLL `-0.0197`, Brier `-0.0084`, but ECE `+0.0079`.
- Reliability-band deltas for RCPS-Retention versus hard CE: low-SNR accuracy `+0.9278 pp`, NLL `-0.0060`, Brier `-0.0020`, ECE `+0.0028`; high-SNR accuracy `+1.6042 pp`, NLL `-0.0415`, ECE `-0.0055`, Brier `-0.0169`.
- Interpretation: UCSD/RML22 provides useful supplementary evidence that RCPS-Retention is stronger than Static LS and can improve accuracy/NLL/Brier on a third AMC dataset. Because the feasible UCSD backbone is CNN4 and the hard-CE seed variance is high, this result should remain supplementary rather than a primary TPAMI claim.


## Iteration 96 - CIFAR-10-C cross-modal runner added and seed-2026 pilot launched (2026-05-17 01:36:00 CST)

- Added a standalone `tools/rcps/run_crossmodal_vision.py` runner for cross-modal RCPS validation on CIFAR-10-C. It uses a CIFAR-style ResNet18 backbone, controlled train-time corruptions, and the shared RCPS target builder from `csrr.models.losses.rcps_loss`.
- Data sources: clean CIFAR-10 images from `/home/citybuster/Data/Visual/CIFAR-10` and CIFAR-10-C arrays from `/home/citybuster/Data/RCPS/raw/CIFAR-10-C/CIFAR-10-C`.
- The first launch exposed a motion-blur implementation error (`PIL.ImageFilter.Kernel` rejected larger kernels). The corruption code was fixed with a numpy one-dimensional averaging implementation and committed before relaunch.
- Launched seed `2026` pilot with identical backbone/data/epoch setup for `Hard CE`, `Static LS`, and `RCPS-Retention`; GPU0 runs Hard CE then Static LS, GPU1 runs RCPS-Retention.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_30ep`.
- Logs: `/home/citybuster/Data/RCPS/work_dirs/logs/cifar10c_resnet18_seed2026_gpu0_hard_static.log` and `/home/citybuster/Data/RCPS/work_dirs/logs/cifar10c_resnet18_seed2026_gpu1_rcps.log`.
- Early monitor at epoch `3/30` showed both Hard CE and RCPS learning normally on clean validation (`~48%`) with no `Traceback`, CUDA OOM, file-handle error, or shape error. No conclusion will be drawn until test CSVs land.


## Iteration 97 - CIFAR-10-C seed-2026 pilot summarized and eps0.10 promoted (2026-05-17 02:20:00 CST)

- Completed the first CIFAR-10-C seed `2026` comparison for `Hard CE`, `Static LS`, and `RCPS-Retention` under the standalone ResNet18-CIFAR runner.
- Initial `RCPS-Retention eps0.30` improved overall corrupted accuracy by `+0.4967 pp`, ECE by `-0.0210`, and Brier by `-0.0031`, but worsened NLL by `+0.0166`. This indicated that the smoothing strength was useful for calibration but too strong for likelihood.
- Ran a same-seed epsilon tuning check for `eps0.15` and `eps0.10`. `eps0.10` was the cleanest setting: overall corrupted accuracy `+0.3073 pp`, NLL `-0.0015`, ECE `-0.0240`, and Brier `-0.0036` versus Hard CE. Clean test accuracy also improved by `+0.18 pp`, with clean ECE `-0.0125`.
- Static LS was not competitive in this pilot: overall corrupted accuracy `-0.2927 pp`, NLL `+0.0556`, ECE `+0.0213`, and Brier `+0.0066` versus Hard CE.
- Decision: promote `RCPS-Retention eps0.10` to seeds `2027/2028` and run matched Hard CE / Static LS / RCPS comparisons. This is cross-modal validation evidence, not yet a final paper claim until the three-seed aggregate is complete.
- Launched seed `2027` full queue on GPU0 and seed `2028` full queue on GPU1. Each queue runs Hard CE, Static LS, then RCPS-Retention eps0.10 with identical backbone/data/epoch setup.


## Iteration 98 - CIFAR-10-C ResNet18 three-seed RCPS validation completed (2026-05-17 03:08:00 CST)

- Completed the matched CIFAR-10-C cross-modal comparison for `ResNet18-CIFAR` with seeds `2026/2027/2028` under the same clean-CIFAR training set, controlled train-time corruptions, 30-epoch budget, optimizer schedule, and evaluation protocol.
- Summary files were written under `/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_30ep/summary`, with the promoted RCPS setting stored under `/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_eps0p10_30ep`.
- Three-seed corrupted-test aggregate: Hard CE accuracy `85.3071 +/- 0.1743`, NLL `0.4432`, ECE `0.0396`, Brier `0.2122`; Static LS accuracy `85.4242 +/- 0.5231`, NLL `0.4852`, ECE `0.0550`, Brier `0.2132`; RCPS-Retention eps0.10 accuracy `85.6733 +/- 0.3387`, NLL `0.4377`, ECE `0.0160`, Brier `0.2074`.
- Mean deltas on CIFAR-10-C versus Hard CE: Static LS gives accuracy `+0.1171 pp` but worsens NLL by `+0.0420`, ECE by `+0.0154`, and Brier by `+0.0009`; RCPS-Retention eps0.10 gives accuracy `+0.3662 pp`, NLL `-0.0055`, ECE `-0.0236`, and Brier `-0.0048`.
- Clean-test retention is also positive: RCPS-Retention eps0.10 improves clean accuracy by `+0.2233 pp`, NLL by `-0.0014`, ECE by `-0.0093`, and Brier by `-0.0022` versus Hard CE. This is important because the reliability-conditioned target does not trade away high-reliability clean performance in this setting.
- Severity-wise RCPS deltas are consistently favorable: severity 1/2/3/4/5 accuracy deltas are `+0.4667/+0.2744/+0.3533/+0.4200/+0.3167 pp`, NLL deltas are `-0.0064/-0.0045/-0.0047/-0.0045/-0.0076`, ECE deltas are `-0.0149/-0.0189/-0.0249/-0.0262/-0.0304`, and Brier deltas are `-0.0050/-0.0046/-0.0043/-0.0045/-0.0057`.
- Training-efficiency diagnostics show a small but useful trend: RCPS-Retention eps0.10 reaches 95% of the Hard-CE best validation accuracy in `19.67` epochs on average versus `20.33` for Hard CE, with slightly higher validation AULC (`76.38` vs `76.31`) and no added model parameters.
- Interpretation: this is the strongest cross-modal evidence so far. It supports the paper's general degraded-observation claim beyond AMC and suggests that RCPS can improve posterior quality and modest accuracy under controlled visual corruptions, while Static LS is not an adequate substitute.


## Iteration 99 - Speech Commands audio runner added and smoke-tested (2026-05-17 03:18:00 CST)

- Added `tools/rcps/run_crossmodal_audio.py`, a standalone Speech Commands runner for noisy-audio RCPS validation. It uses balanced `label x SNR` sampling, additive background noise from the official `_background_noise_` files, log-mel features, DS-CNN/log-mel ResNet backbones, and the shared `build_rcps_targets` implementation.
- The runner keeps the comparison paired: Hard CE, Static LS, and RCPS-Retention share the same data split, balanced sampling policy, model, optimizer, epoch budget, and evaluation code; only the supervision target/loss changes.
- A minimal hard-CE smoke test completed successfully with 231 train, 154 validation, and 154 test examples. It exercised waveform loading, SNR mixing, feature extraction, training, checkpoint selection, test export, and reliability/SNR-bin metrics writing.
- Smoke metrics landed at `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_smoke/metrics/speechcommands_ds-cnn_hard-ce_seed2026_test.csv`. These numbers are not paper evidence; they only validate the execution path.


## Iteration 100 - Speech Commands DS-CNN pilot launched (2026-05-17 03:23:00 CST)

- Launched the first noisy-audio pilot on Speech Commands v0.02 using `DS-CNN + seed 2026` for `Hard CE`, `Static LS`, and `RCPS-Retention eps0.10`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_pilot_15ep`.
- GPU0 runs Hard CE followed by Static LS; GPU1 runs RCPS-Retention. Logs are `/home/citybuster/Data/RCPS/work_dirs/logs/speechcommands_ds_cnn_seed2026_gpu0_hard_static.log` and `/home/citybuster/Data/RCPS/work_dirs/logs/speechcommands_ds_cnn_seed2026_gpu1_rcps.log`.
- The run uses balanced sampling with `500` train, `150` validation, and `250` test samples per `label x SNR` bucket, giving 38,500 train, 11,550 validation, and 19,250 test examples per method. This avoids an inflated baseline caused by the large `unknown` class.
- Initial logs show dataset construction succeeded and no `Traceback`, CUDA OOM, file-handle, or missing-file error has appeared. These pilot results will decide whether to expand to full-test and three-seed audio validation.


## Iteration 101 - Speech Commands audio pipeline intervention (2026-05-17 03:31:00 CST)

- The first Speech Commands pilot launch was stopped before any epoch completed because GPU utilization stayed near zero for several minutes. Process inspection showed the bottleneck was CPU-side per-sample log-mel extraction inside the dataset, which would make training-efficiency comparisons unreliable.
- Updated `run_crossmodal_audio.py` so the dataset returns fixed-length waveforms and a `LogMelFeature` module computes log-mel features on the GPU batch. This keeps the model/loss comparison unchanged while removing an avoidable preprocessing bottleneck.
- A post-patch speed check with 7,700 train, 3,850 validation, and 3,850 test examples completed one epoch plus test export in about 67 seconds wall time, compared with no epoch output after several minutes before the patch. The stopped pilot is diagnostic only and will not be used as evidence.


## Iteration 102 - Speech Commands 10-epoch pilot analyzed (2026-05-17 03:38:00 CST)

- Completed the repaired 10-epoch Speech Commands pilot for `DS-CNN + seed 2026` on a balanced `label x SNR` subset.
- Hard CE is a valid baseline in this setting: test accuracy and macro accuracy are both `79.1429`, NLL `0.6479`, ECE `0.0777`, and Brier `0.2960`.
- Static LS improves accuracy slightly (`+0.2338 pp`) but substantially worsens posterior metrics: NLL `+0.0923`, ECE `+0.0806`, and Brier `+0.0272` versus Hard CE.
- RCPS-Retention `eps0.10` is too strong for audio: accuracy `+0.0260 pp`, but NLL `+0.0376`, ECE `+0.0350`, and Brier `+0.0109` versus Hard CE.
- A weaker sweep improved the tradeoff. `eps0.03` gives accuracy `+0.1195 pp` but still worsens NLL/ECE/Brier slightly. `eps0.05` gives the best accuracy gain (`+0.4364 pp`) and slightly improves Brier (`-0.0004`), but still worsens NLL (`+0.0074`) and ECE (`+0.0146`).
- Interpretation: this pilot does not support fixed-strength uniform RCPS as a universal audio solution. It supports the need for validation-constrained or entropy-matched RCPS: if a reliability bin is already under-confident, additional smoothing should be reduced or disabled. A 20-epoch check is launched to verify whether this is a short-training artifact before expanding audio experiments.


## Iteration 103 - Speech Commands 20-epoch check completed (2026-05-17 03:53:00 CST)

- Completed the 20-epoch Speech Commands check for `DS-CNN + seed 2026`, comparing Hard CE with the best 10-epoch RCPS candidate (`RCPS-Retention eps0.05`).
- Hard CE improves substantially with the longer schedule: test accuracy `82.6078`, NLL `0.5252`, ECE `0.0238`, Brier `0.2434`, mean confidence `0.8024`, and mean entropy `0.5877`.
- `RCPS-Retention eps0.05` gives almost identical accuracy (`+0.0468 pp`) and slightly better Brier (`-0.0002`), but worsens NLL by `+0.0072` and ECE by `+0.0261` versus Hard CE.
- SNR-bin behavior confirms the diagnosis: RCPS lowers confidence in bins where the hard-CE model is already close to calibrated or under-confident, so fixed uniform smoothing is not appropriate for this audio setup.
- Decision: do not expand fixed-uniform RCPS on Speech Commands as a main result. Audio should be used to motivate validation-constrained RCPS/EntropyMatch or PosteriorBase rather than a universal smoothing rule. This is a theory-shaping negative result, not a failed baseline.


## Iteration 104 - CIFAR-10-C ResNet34 pilot support added (2026-05-17 04:00:00 CST)

- Extended `run_crossmodal_vision.py` with a `--model` argument supporting both `resnet18-cifar` and `resnet34-cifar` while preserving the existing ResNet18 output naming and behavior.
- A minimal ResNet34-CIFAR smoke run completed and wrote metrics, validating the model switch, path layout, and CSV export. The smoke subset is intentionally tiny and not evidence.
- Next step: launch a matched seed-2026 ResNet34 pilot for Hard CE, Static LS, and RCPS-Retention eps0.10 on CIFAR-10-C. If the trend matches ResNet18, this becomes the second vision backbone for cross-modal robustness evidence.


## Iteration 105 - CIFAR-10-C ResNet34 pilot completed (2026-05-17 04:38:00 CST)

- Completed the seed-2026 CIFAR-10-C ResNet34-CIFAR pilot for Hard CE, Static LS, RCPS-Retention eps0.10, and a weaker RCPS-Retention eps0.05 check.
- Hard CE test: clean accuracy `87.65`, corrupted accuracy `84.8467`, NLL `0.4626`, ECE `0.0461`, and Brier `0.2195` on the corrupted aggregate.
- Static LS is again not competitive: corrupted accuracy `-0.3420 pp`, NLL `+0.0490`, ECE `+0.0001`, and Brier `+0.0050` versus Hard CE.
- RCPS eps0.10 improves corrupted ECE by `-0.0184` but hurts accuracy by `-0.4220 pp`, NLL by `+0.0069`, and Brier by `+0.0046`. RCPS eps0.05 reduces the tradeoff but still hurts corrupted accuracy by `-0.3020 pp`, NLL by `+0.0015`, and Brier by `+0.0027`, while improving ECE by `-0.0168`.
- Decision: do not expand ResNet34-CIFAR to three seeds under fixed uniform RCPS. The pilot is useful as a constraint: the successful ResNet18 setting should be described as validation-selected, not as a universal epsilon. For deeper/stronger backbones, RCPS needs retention/entropy constraints that explicitly prevent accuracy and Brier loss.


## Iteration 106 - DPC-RCPS theory upgrade and AMC 10A evidence (2026-05-18 22:00:00 CST)

- Upgraded the method line from fixed reliability smoothing to `DPC-RCPS` (Degradation-Posterior Consistent RCPS), motivated by posterior consistency along a degradation path.
- Implemented sample-posterior target support, entropy-projected variants, reliability-conditioned temperature scaling, and MCformer DPC configs. Key commits include `a20525c`, `4e4f9a2`, `a9f8438`, `28d136c`, and `d97a589`.
- PETCGDNN on RadioML2016.10A completed a paired three-seed DPC comparison. Hard CE reached `56.08 +/- 1.56` accuracy, while DPC-v1 reached `59.10 +/- 1.25`, with paired deltas of `+3.03 pp` accuracy, `-0.0585` NLL, `-0.0113` ECE, and `-0.0328` Brier. Entropy-projected DPC with reliability-temperature scaling reached `59.51 +/- 1.71` accuracy and stronger NLL/Brier/low-SNR ECE gains, but worsened aggregate ECE.
- MCformer on RadioML2016.10A completed a DPC-v1 three-seed check. DPC-v1 gave modest but stable deltas: `+0.085 pp` accuracy, `-0.0014` NLL, `+0.0012` ECE, and `-0.0003` Brier. Reliability-temperature scaling improved NLL and reliability-bin ECE but worsened aggregate ECE.
- Interpretation: 10A supports DPC as a posterior-quality and retention framework, with PETCGDNN giving strong positive evidence and MCformer showing a model-dependent boundary case. This shaped the manuscript claim away from universal smoothing and toward posterior consistency with validation constraints.


## Iteration 107 - MCformer RadioML2016.10B DPC-RCPS three-seed result (2026-05-19 06:52:00 CST)

- Added RadioML2016.10B MCformer DPC configs in commit `fda7149`. The valid 10B setup uses `num_classes=10`, with 600k train, 120k validation, and 480k test samples.
- Built the sample-posterior teacher from the valid MCformer hard-CE seed-2026 checkpoint:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201610B/mcformer_hard-ce_seed2026_train.npz`.
  The artifact contains 600000 train samples, 10 classes, normalized probabilities, and full `sample_idx` coverage.
- Completed DPC-RCPS seeds `2026/2027/2028` and summarized them against the valid hard-CE `num_classes=10` baseline:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_main/summary/dpc_mcformer_10B_three_seed_summary.csv`.
- Three-seed hard CE: accuracy `58.55 +/- 0.36`, NLL `0.9908 +/- 0.0064`, ECE `0.0117 +/- 0.0010`, Brier `0.4574 +/- 0.0031`.
- Three-seed DPC-RCPS: accuracy `64.06 +/- 0.22`, NLL `0.9088 +/- 0.0053`, ECE `0.0130 +/- 0.0008`, Brier `0.3986 +/- 0.0035`.
- Paired DPC minus hard-CE deltas: accuracy `+5.51 +/- 0.36 pp`, NLL `-0.0820 +/- 0.0027`, ECE `+0.0012 +/- 0.0013`, Brier `-0.0588 +/- 0.0024`.
- Low-SNR deltas: accuracy `-0.18 +/- 0.23 pp`, NLL `-0.0047 +/- 0.0035`, ECE `-0.0045 +/- 0.0042`, Brier approximately unchanged.
- High-SNR deltas: accuracy `+8.50 +/- 0.46 pp`, NLL `-0.1230 +/- 0.0088`, ECE `-0.0013 +/- 0.0057`, Brier `-0.0918 +/- 0.0065`.
- Interpretation: this is currently the strongest AMC evidence. It supports the DPC-RCPS claim that degradation-aware posterior targets can improve accuracy, likelihood, Brier score, and high-reliability retention under a stable baseline. Aggregate ECE is slightly worse, so the paper should not claim universal ECE improvement; it should emphasize reliability-stratified calibration and posterior recovery.


## Iteration 108 - RadioML2018.01A PETCGDNN DPC preparation (2026-05-19 07:52:00 CST)

- While `PETCGDNN + RadioML2018.01A + hard CE + seed 2026` baseline gate was running, added SNR-aware PETCGDNN 2018A config for future DPC training without touching the active process.
- Added `configs/rcps/_base_/models/petcgdnn_iq-snr-deepsig-201801A.py` with `sample_idx`, `snr`, `snr_label`, and `modulation` metadata packed for all splits.
- Added `configs/rcps/dpc/petcgdnn_dpc-rcps_iq-snr-deepsig-201801A.py` with `sample_posterior` base pointing to the planned train-split teacher artifact.
- Verified both configs with `mmengine.Config.fromfile`: 24 classes, correct absolute data root, correct metadata, and expected hard/DPC loss types.
- No experiment matrix expansion has been launched yet. The next decision remains gated on the completed hard-CE seed-2026 test metrics for RadioML2018.01A.


## Iteration 109 - RadioML2018.01A PETCGDNN baseline gate in progress (2026-05-19 11:00:00 CST)

- `PETCGDNN + RadioML2018.01A + hard CE + seed 2026` is running on GPU1 with `num_workers=0`; no traceback, CUDA OOM, file-handle error, or export error has been observed.
- Seed 2026 validation accuracy has repeatedly improved despite oscillations, reaching `62.1133%` at epoch 57. No test CSV has landed yet because early stopping has not triggered.
- After MCformer 2018A 1-epoch smoke completed (`34.9935%` validation accuracy with high per-epoch cost), GPU0 was used only for a second hard-CE baseline seed, not for DPC/RCPS.
- `PETCGDNN + RadioML2018.01A + hard CE + seed 2027` is running on GPU0. It reached `60.7607%` validation accuracy at epoch 19, then entered a low-accuracy oscillatory phase and was recovering to `58.1468%` by epoch 42.
- Both seeds show substantial validation oscillation after initially crossing 60%. This does not invalidate the baseline because the checkpoint hook preserves the best validation checkpoint, but any 2018A evidence must use best-checkpoint test CSVs, paired seeds, and explicit training-stability discussion.
- No RadioML2018.01A DPC result is claimed yet. DPC remains gated on completed hard-CE test metrics.


## Iteration 110 - RadioML2018.01A PETCGDNN hard-CE gate two seeds completed (2026-05-19 22:55:00 CST)

- The first two `PETCGDNN + RadioML2018.01A + hard CE` baseline seeds completed naturally and successfully exported test predictions and reliability-bin CSVs.
- Seed 2026 test overall: accuracy `62.7312`, NLL `1.1236`, ECE `0.0043`, Brier `0.3962`, mean confidence `0.6312`, mean entropy `1.1208`.
- Seed 2027 test overall: accuracy `60.7220`, NLL `1.1789`, ECE `0.0065`, Brier `0.4198`, mean confidence `0.6080`, mean entropy `1.1859`.
- Two-seed mean so far: accuracy `61.7266`, NLL `1.1512`, ECE `0.0054`, Brier `0.4080`.
- The export path is now verified for the large 2018A test split (`1,022,736` samples), with no recurrence of the earlier file-handle issue.
- Because both completed seeds pass the baseline gate, launched the third hard-CE seed (`2028`) using the same data, backbone, optimizer, worker settings, and export path:
  `/home/citybuster/Data/RCPS/work_dirs/logs/baseline_2018A_petcgdnn_hard_seed2028_gpu1.log`.
- No DPC/RCPS experiment is claimed or launched for 2018A yet. DPC remains gated on the completed three-seed hard-CE baseline summary.


## Iteration 111 - RadioML2018.01A PETCGDNN DPC teacher artifact prepared (2026-05-19 23:16:00 CST)

- Prepared the DPC teacher posterior artifact for RadioML2018.01A using the completed seed-2026 PETCGDNN hard-CE checkpoint:
  `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A/petcgdnn/hard-ce/seed_2026/best_accuracy_top1_epoch_176.pth`.
- Exported train-split predictions with SNR/sample-index metadata to:
  `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A/petcgdnn/hard-ce/seed_2026/predictions/train_snr_meta.pkl`.
- Built sample-posterior artifact:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201801A/petcgdnn_hard-ce_seed2026_train.npz`.
- Artifact validation: `1,277,952` train samples, `24` classes, unique sample-index coverage, probability sums within `[0.9999999, 1.0000001]`, and train prediction accuracy `63.10%`.
- This is preparation only. No DPC/RCPS 2018A training is launched until the hard-CE three-seed baseline gate completes.

## Iteration 112 - RadioML2018.01A PETCGDNN hard-CE baseline gate passed (2026-05-20 12:16:00 CST)

- The third `PETCGDNN + RadioML2018.01A + hard CE` seed completed naturally. Seed 2028 used `best_accuracy_top1_epoch_160.pth` and exported the large test split successfully with `num_workers=0`.
- Seed 2028 test overall: accuracy `62.4828`, NLL `1.1259`, ECE `0.0043`, Brier `0.3997`, mean confidence `0.6278`, mean entropy `1.1242`.
- Three-seed hard-CE summary: accuracy `61.9787±1.0954`, NLL `1.1428±0.0313`, ECE `0.0050±0.0013`, Brier `0.4052±0.0127`.
- The baseline gate threshold of `>=61.0%` mean test accuracy is passed. DPC-RCPS on RadioML2018.01A is now allowed under the baseline-first protocol.
- A non-fatal CRLF issue in the seed-2028 helper script produced `date
: command not found` after all metrics were already written; it did not affect the checkpoint, prediction file, or CSV.


## Iteration 113 - DPC-RCPS RadioML2018.01A PETCGDNN seed 2026 launched (2026-05-20 12:35:00 CST)

- After the three-seed hard-CE baseline gate passed, launched the first DPC-RCPS pilot on `PETCGDNN + RadioML2018.01A + seed 2026`.
- The DPC run uses the same backbone, data split, optimizer schedule, and export/analyze path as the admitted hard-CE baseline. Only the supervision target changes to the training-split sample-posterior artifact from the admitted seed-2026 teacher.
- Teacher artifact: `/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201801A/petcgdnn_hard-ce_seed2026_train.npz`.
- Work dir: `/home/citybuster/Data/RCPS/work_dirs/dpc_main/amc/deepsig201801A/petcgdnn_dpc-rcps/seed_2026`.
- Foreground monitor target log: `/home/citybuster/Data/RCPS/work_dirs/logs/dpc_2018A_petcgdnn_seed2026_gpu0.log`; launch PID `4073004`.
- No additional datasets, models, or RCPS variants are launched in this step.

## Iteration 114 - RadioML2018.01A PETCGDNN DPC-v1 boundary result (2026-05-20 18:45:00 CST)

- Stage: DPC-RCPS diagnostic pilot after PETCGDNN hard-CE baseline gate passed on RadioML2018.01A.
- Commit: ba6a96b.
- Run: PETCGDNN / DPC-RCPS v1 / seed 2026.
- Work dir: /home/citybuster/Data/RCPS/work_dirs/dpc_main/amc/deepsig201801A/petcgdnn_dpc-rcps/seed_2026.
- Best validation checkpoint: best_accuracy_top1_epoch_40.pth, validation top-1 = 62.1858%.
- Training ended by early stopping at epoch 90 after the monitored metric did not improve for 50 records.
- Test metrics path: /home/citybuster/Data/RCPS/work_dirs/dpc_main/metrics/deepsig201801A_petcgdnn_dpc-rcps_seed2026_test.csv.
- Overall test comparison against hard CE seed 2026:
  - DPC-RCPS v1: accuracy 62.1759%, NLL 1.1561, ECE 0.0162, Brier 0.4030, confidence 0.6071, entropy 1.2740.
  - Hard CE: accuracy 62.7312%, NLL 1.1236, ECE 0.0043, Brier 0.3962, confidence 0.6312, entropy 1.1208.
  - Delta DPC - hard: accuracy -0.555 pp, NLL +0.0325, ECE +0.0119, Brier +0.0068.
- Reliability-bin diagnosis:
  - At -20 dB, DPC gives a small uncertainty benefit: NLL -0.0204, ECE -0.0037, Brier -0.0014, but accuracy is essentially unchanged/slightly lower.
  - Averaged over low SNR bins (<=0 dB), DPC is worse: accuracy -0.384 pp, NLL +0.0298, ECE +0.0051, Brier +0.0037.
  - Averaged over high SNR bins (>=20 dB), DPC is also worse: accuracy -0.302 pp, NLL +0.0158, ECE +0.0093, Brier +0.0042.
- Decision:
  - Do not expand DPC-v1 on RadioML2018.01A to seeds 2027/2028.
  - Treat this as a boundary/diagnostic result: the sample-posterior target transferred from the seed-2026 teacher over-softens PETCGDNN on the larger 2018A setting.
  - Next candidate must use a more conservative target: stronger high-reliability retention, smaller epsilon, validation-constrained entropy/temperature, or a hard-CE warmup before posterior supervision.

## Iteration 115 - Conservative DPC-v2 low-only pilot launched (2026-05-20 18:49:23 CST)

- Motivation: DPC-v1 on RadioML2018.01A/PETCGDNN over-softened the model and degraded overall accuracy, NLL, ECE, and Brier relative to hard CE.
- Change: added configs/rcps/dpc/petcgdnn_dpc-lowonly-eps03_iq-snr-deepsig-201801A.py.
- Commit: cb98cdc.
- DPC-v2 target schedule:
  - psilon(type=low_reliability_power, max=0.3, gamma=2.0, cutoff=0.4) under the linear SNR map [-20, 30] -> [0, 1].
  - Effective epsilon: -20 dB = 0.300, -15 dB = 0.16875, -10 dB = 0.075, >=0 dB = 0.
  - Teacher posterior base is sharpened with temperature 0.75 and a smaller low-reliability prior blend 0.10.
- Run: PETCGDNN / RadioML2018.01A / seed 2026 / DPC-v2 low-only.
- Work dir: /home/citybuster/Data/RCPS/work_dirs/dpc_v2/amc/deepsig201801A/petcgdnn_dpc-lowonly-eps03/seed_2026.
- Log: /home/citybuster/Data/RCPS/work_dirs/logs/dpc_v2_2018A_petcgdnn_lowonly_eps03_seed2026_gpu0.log.
- Acceptance for expansion: must improve low-SNR NLL/Brier without degrading high-SNR accuracy by more than 1%; if overall metrics remain worse than hard CE, do not expand to more seeds.


## Iteration 116 - RadioML2018.01A PETCGDNN DPC-v2 low-only diagnostic result (2026-05-20 21:16:00 CST)

- Stage: conservative DPC-v2 pilot after DPC-v1 over-softened RadioML2018.01A/PETCGDNN.
- Commit: 11c0e69 for launch; config introduced in cb98cdc.
- Config: configs/rcps/dpc/petcgdnn_dpc-lowonly-eps03_iq-snr-deepsig-201801A.py.
- Schedule: epsilon(type=low_reliability_power, max=0.3, gamma=2.0, cutoff=0.4), so posterior supervision is active only below roughly 0 dB under the [-20, 30] SNR map.
- Intervention: stopped training after epoch 31 because validation collapsed for three consecutive epochs after the best point:
  - best epoch 28 validation accuracy 61.1180%;
  - epoch 29 12.8381%, epoch 30 23.3939%, epoch 31 25.1975%.
- Best checkpoint: /home/citybuster/Data/RCPS/work_dirs/dpc_v2/amc/deepsig201801A/petcgdnn_dpc-lowonly-eps03/seed_2026/best_accuracy_top1_epoch_28.pth.
- Test metrics path: /home/citybuster/Data/RCPS/work_dirs/dpc_v2/metrics/deepsig201801A_petcgdnn_dpc-lowonly-eps03_seed2026_test.csv.
- Overall test comparison against hard CE seed 2026:
  - DPC-v2 low-only: accuracy 61.1777%, NLL 1.1654, ECE 0.0056, Brier 0.4145, confidence 0.6159, entropy 1.1615.
  - Hard CE: accuracy 62.7312%, NLL 1.1236, ECE 0.0043, Brier 0.3962, confidence 0.6312, entropy 1.1208.
  - Delta DPC-v2 - hard: accuracy -1.554 pp, NLL +0.0418, ECE +0.0013, Brier +0.0184.
- Reliability-bin diagnosis:
  - Low bins (<=0 dB) average delta: accuracy -1.132 pp, NLL +0.0361, ECE +0.0036, Brier +0.0109.
  - High bins (>=20 dB) average delta: accuracy -0.968 pp, NLL +0.0262, ECE +0.0012, Brier +0.0147.
- Decision:
  - Do not expand DPC-v2 low-only on RadioML2018.01A to more seeds.
  - DPC-v2 is less over-soft than DPC-v1 in aggregate ECE, but it still loses accuracy, NLL, and Brier and is dynamically unstable.
  - Next algorithmic direction should not directly replace labels with teacher posterior targets on 2018A. Prefer a hard-CE primary objective plus a posterior-consistency regularizer or entropy-bounded auxiliary loss, with validation-gated activation and explicit stability checks.
- Housekeeping: cleaned CRLF-contaminated log filenames produced by the PowerShell-to-SSH launch script.


## Iteration 117 - DPC posterior-consistency pilot prepared (2026-05-20 22:42:00 CST)

- Motivation: DPC-v1 and DPC-v2 on RadioML2018.01A/PETCGDNN showed that directly replacing hard labels with sample posterior targets can over-soften or destabilize strong baselines.
- Algorithm change: added `DPCConsistencyLoss`, which keeps hard cross-entropy as the primary objective and adds a reliability-gated KL consistency term to the sample posterior teacher.
- Commit: e837e9f.
- Config: `configs/rcps/dpc/petcgdnn_dpc-consistency-eps03_iq-snr-deepsig-201801A.py`.
- Schedule: `epsilon(type=low_reliability_power, max=0.3, gamma=2.0, cutoff=0.4)` under the same [-20, 30] SNR map; `consistency_weight=0.1`.
- Safety checks passed before launch: config builds, Python compile passes, toy loss test confirms `consistency_weight=0` is exactly hard CE, and missing `sample_idx` raises a `KeyError`.
- Acceptance for expansion: this single-seed pilot must not reproduce the DPC-v2 instability, must keep high-SNR accuracy drop within 1%, and should improve at least a majority of low-SNR NLL/ECE/Brier metrics relative to hard CE or current DPC variants.


## Iteration 118 - DPC posterior-consistency pilot launched and early curve normal (2026-05-20 21:52:00 CST)

- Commit at launch: 4f88694.
- Run: PETCGDNN / RadioML2018.01A / DPC posterior-consistency / seed 2026.
- Work dir: `/home/citybuster/Data/RCPS/work_dirs/dpc_consistency/amc/deepsig201801A/petcgdnn_dpc-consistency-eps03/seed_2026`.
- Log: `/home/citybuster/Data/RCPS/work_dirs/logs/dpc_consistency_2018A_petcgdnn_eps03_seed2026_gpu0.log`.
- First-run observation: data/cache startup took several minutes on the large 2018A split, with heavy CPU/I/O and no file descriptor leak. Training then entered normal GPU execution.
- Early validation curve: epoch 1 = 41.8226%, epoch 2 = 48.4186%, epoch 3 = 52.0073%, epoch 4 = 54.8892%. No early collapse like DPC-v2 has appeared so far.
- Current decision: continue the single-seed pilot. Do not expand to more seeds or models until the run finishes and test reliability-bin metrics are compared against the admitted hard-CE seed-2026 baseline.


## Iteration 119 - DPC posterior-consistency diagnostic result (2026-05-20 22:22:00 CST)

- Stage: DPC posterior-consistency pilot after direct DPC-v1/v2 target replacement failed on RadioML2018.01A/PETCGDNN.
- Code commit: e837e9f; launch/docs commit: e038570.
- Config: `configs/rcps/dpc/petcgdnn_dpc-consistency-eps03_iq-snr-deepsig-201801A.py`.
- Intervention: stopped training during epoch 10 after a clear instability signal. Validation climbed normally through epoch 8 but then collapsed at epoch 9:
  - epoch 1: 41.8226%; epoch 2: 48.4186%; epoch 3: 52.0073%; epoch 4: 54.8892%; epoch 5: 56.6818%; epoch 6: 55.0036%; epoch 7: 58.1703%; epoch 8: 58.9172%; epoch 9: 37.5870%.
  - Training loss jumped at epoch 9 around iteration 1850 (4.1923), then remained elevated.
- Best checkpoint evaluated: `/home/citybuster/Data/RCPS/work_dirs/dpc_consistency/amc/deepsig201801A/petcgdnn_dpc-consistency-eps03/seed_2026/best_accuracy_top1_epoch_8.pth`.
- Test metrics path: `/home/citybuster/Data/RCPS/work_dirs/dpc_consistency/metrics/deepsig201801A_petcgdnn_dpc-consistency-eps03_seed2026_test.csv`.
- Overall test comparison against admitted hard-CE seed 2026:
  - DPC-consistency: accuracy 58.9566%, NLL 1.2235, ECE 0.0073, Brier 0.4399, confidence 0.5888, entropy 1.2412.
  - Hard CE: accuracy 62.7312%, NLL 1.1236, ECE 0.0043, Brier 0.3962, confidence 0.6312, entropy 1.1208.
  - Delta DPC-consistency - hard: accuracy -3.775 pp, NLL +0.1000, ECE +0.0030, Brier +0.0437.
- Reliability-bin diagnosis:
  - Low bins (<=0 dB) average delta: accuracy -2.142 pp, NLL +0.0871, ECE +0.0020, Brier +0.0194.
  - High bins (>=20 dB) average delta: accuracy -3.695 pp, NLL +0.0864, ECE +0.0108, Brier +0.0512.
- Decision:
  - Do not expand DPC-consistency on RadioML2018.01A.
  - Treat sample-posterior teacher injection as a diagnostic failure mode, not the paper's main algorithm.
  - Keep the degradation-posterior consistency theorem as a conceptual constraint, but practical RCPS should use validation-constrained class-level posterior bases, explicit high-reliability retention, and only include sample-adaptive teachers if a future stability mechanism passes the baseline gate.


## Iteration 120 - RadioML2018.01A class-level RCPS-Hybrid pilots prepared (2026-05-20 22:40:00 CST)

- Rationale: after DPC-v1/v2 and DPC-consistency failed on RadioML2018.01A/PETCGDNN, the next test should remove sample-level teacher bias and return to validation-constrained class-level posterior bases.
- Built reliability-conditioned class posterior base from admitted hard-CE seed-2026 validation predictions:
  `/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201801A/petcgdnn_hard-ce_seed2026_reliability_base.npz`.
- Source validation prediction file: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A/petcgdnn/hard-ce/seed_2026/predictions/validation.pkl`.
- Base shape: 26 SNR bins x 24 true classes x 24 posterior coordinates; SNR range is -20 to 30 dB.
- Added two conservative configs:
  - `configs/rcps/dpc/petcgdnn_rcps-hybrid-eps01_iq-snr-deepsig-201801A.py`: `retention_power(max=0.1, gamma=2.0, retain_min=0.8)`.
  - `configs/rcps/dpc/petcgdnn_rcps-hybrid-eps02_iq-snr-deepsig-201801A.py`: `retention_power(max=0.2, gamma=2.0, retain_min=0.8)`.
- Acceptance gate for expansion: relative to hard CE seed 2026, a pilot must improve validation/test NLL or Brier without high-SNR accuracy drop above 1%. If both fail, 2018A remains a boundary dataset and the paper should not claim the current RCPS family universally improves large-scale AMC.


## Iteration 121 - RadioML2018.01A class-level RCPS-Hybrid pilots launched (2026-05-20 22:56:00 CST)

- Commit at launch: 90194aa.
- Runs launched in parallel:
  - GPU0: PETCGDNN / RadioML2018.01A / `rcps-hybrid-eps01` / seed 2026.
  - GPU1: PETCGDNN / RadioML2018.01A / `rcps-hybrid-eps02` / seed 2026.
- Both runs use the same backbone, split, optimizer, 400-epoch budget, early stopping, and `num_workers=0`; only the RCPS target changes.
- Startup note: both processes spent several minutes in CPU/I/O while loading the large 2018A split, then entered normal GPU training with no traceback/OOM/file-handle errors.
- First validation checkpoint:
  - eps0.1 epoch 1 validation accuracy: 41.0245%.
  - eps0.2 epoch 1 validation accuracy: 39.4529%.
- Current action: keep both running. eps0.1 is early-stabler, but no pilot is admitted or rejected until validation/test reliability-bin CSVs are generated from the best checkpoints.

## Iteration 122 - 2018A RCPS-Hybrid foreground monitor and checkpoint-export fix (2026-05-21 09:38:00 CST)

- Stage: PETCGDNN / RadioML2018.01A / class-level RCPS-Hybrid / seed 2026 pilots.
- Runs remain healthy: no traceback, CUDA OOM, file-handle error, or data-loading error was detected.
- Current validation status at the intervention point:
  - `rcps-hybrid-eps01`: best validation accuracy 62.7927% at epoch 142.
  - `rcps-hybrid-eps02`: best validation accuracy 61.8594% at epoch 164.
- Post-processing risk found: the wrapper selected `ls -t best_accuracy_top1_epoch_*.pth epoch_*.pth | head -1`, which can choose a newer ordinary `epoch_*.pth` instead of the best checkpoint for prediction export.
- Recovery action: patched `/home/citybuster/Data/RCPS/work_dirs/logs/run_rcps_hybrid_2018A_pilot.sh` so prediction export prioritizes `best_accuracy_top1_epoch_*.pth` and falls back to ordinary epoch checkpoints only if no best checkpoint exists.
- Follow-up check: after metrics land, verify that validation/test predictions were generated from the best checkpoint. If the running shell used the old cached line, rerun collect/analyze from the best checkpoint and overwrite the CSVs. Training itself was not modified.

## Iteration 123 - Prediction export metadata tightened (2026-05-21 09:44:00 CST)

- Motivation: after finding the checkpoint-selection risk in the RCPS-Hybrid export wrapper, prediction artifacts should explicitly record which checkpoint produced them.
- Code change: `tools/rcps/collect_predictions.py` now prints the checkpoint path and stores `checkpoint` and `config` fields in each exported prediction pkl.
- Validation: `python3 -m py_compile tools/rcps/collect_predictions.py` passed.
- Commit: 3364028.
- Impact: this does not change training, model predictions, or metrics; it only improves traceability for validation/test CSVs generated after this commit.

## Iteration 124 - 2018A RCPS-Hybrid eps0.1 refreshes validation best (2026-05-21 12:16:00 CST)

- Stage: PETCGDNN / RadioML2018.01A / class-level RCPS-Hybrid / seed 2026 pilots.
- `rcps-hybrid-eps01` refreshed validation best at epoch 176: accuracy/top1 62.8029%, improving over the earlier epoch-142 best 62.7927% and remaining above the matched hard-CE validation plateau.
- `rcps-hybrid-eps02` remains on a weaker plateau around 61.85%-61.87%, so it is not a likely expansion candidate unless final posterior metrics unexpectedly compensate.
- No traceback, CUDA OOM, or file-handle error has appeared. Because eps0.1 refreshed its best checkpoint, early-stopping patience resets. Continue running to natural stop and evaluate only best-checkpoint validation/test CSVs.

## Iteration 125 - 2018A RCPS-Hybrid eps0.1 refreshes again (2026-05-21 12:47:00 CST)

- Stage: PETCGDNN / RadioML2018.01A / class-level RCPS-Hybrid / seed 2026 pilots.
- `rcps-hybrid-eps01` refreshed validation best again at epoch 181: accuracy/top1 62.8048%, improving over the previous epoch-176 best of 62.8029%.
- `rcps-hybrid-eps02` remains on a weaker plateau around 61.84%-61.86%; it is no longer a likely main candidate unless final NLL/Brier unexpectedly dominate.
- Continue to natural early stopping. If eps0.1 passes validation/test posterior metrics and high-reliability retention gates, expand eps0.1 to seeds 2027/2028. No paper claim is updated until test CSVs are available.

### 2026-05-21 14:33 CST - 2018A RCPS-Hybrid foreground recovery

- Foreground monitoring resumed after context interruption on branch `feature/rcps-experiments`, commit `a9d1ed8`.
- `rcps-hybrid-eps01` remains active on GPU0. Recent validation accuracies: epoch205 = 62.7978%, epoch206 = 62.7841%. No `Traceback`, CUDA OOM, or file-handle errors observed.
- `rcps-hybrid-eps02` was stopped as a diagnostic branch because validation plateaued around 61.85%-61.87%, below hard CE and eps0.1.
- During manual stop, eps0.2 appears to have been interrupted inside the best-checkpoint replacement window: previous best was removed and the new best file was not present. The only available checkpoint is `epoch_241.pth`.
- Export script `/home/citybuster/Data/RCPS/work_dirs/logs/export_rcps_hybrid_eps02_best.sh` was patched to prefer best checkpoints and fall back to the newest epoch checkpoint when best is absent. eps0.2 validation/test exports must therefore be treated as fallback-checkpoint diagnostics, not standard best-checkpoint results.

### 2026-05-21 14:47 CST - 2018A eps0.2 fallback diagnostic complete

- `rcps-hybrid-eps02` fallback export completed using `epoch_241.pth`; prediction pkl metadata records the checkpoint and config.
- Test overall vs matched hard CE seed2026: accuracy 61.7957 vs 62.7312 (delta -0.936 pp), NLL 1.1470 vs 1.1236 (delta +0.0234), ECE 0.00542 vs 0.00428 (delta +0.00114), Brier 0.4057 vs 0.3962 (delta +0.00958).
- Reliability-bin averages do not pass the retention gate: low-SNR accuracy delta -0.513 pp with NLL +0.0205 and Brier +0.00394; high-SNR accuracy delta -1.060 pp with NLL +0.0187 and Brier +0.0116.
- Decision: `epsilon_max=0.2` is too strong for RadioML2018.01A/PETCGDNN, even with class-level posterior/confusion base. Do not admit it to main results and do not expand to more seeds. Continue monitoring `eps0.1`.

### 2026-05-21 16:39 CST - 2018A eps0.1 seed2026 completed

- `rcps-hybrid-eps01` stopped after epoch231 and exported validation/test predictions. Both pkl metadata fields confirm checkpoint `best_accuracy_top1_epoch_181.pth`.
- Test overall vs matched hard CE seed2026: accuracy 62.7622 vs 62.7312 (delta +0.031 pp), NLL 1.12448 vs 1.12358 (delta +0.00090), ECE 0.00375 vs 0.00428 (delta -0.00053), Brier 0.39566 vs 0.39616 (delta -0.00050).
- Reliability-bin averages: low-SNR accuracy -0.107 pp, NLL +0.00420, ECE -0.00106, Brier +0.00098; mid-SNR accuracy +0.151 pp, NLL -0.00152, Brier -0.00181; high-SNR accuracy +0.103 pp, NLL -0.00154, Brier -0.00126.
- Decision: eps0.1 passes high-reliability retention and shows small ECE/Brier and mid/high-SNR posterior-quality gains, but overall NLL and low-SNR NLL/Brier are slightly worse. It is admitted only for stability checking with seeds 2027/2028, not as a standalone strong 2018A conclusion.

### 2026-05-21 16:43 CST - 2018A eps0.1 stability seeds launched

- Launched `rcps-hybrid-eps01` stability runs for RadioML2018.01A/PETCGDNN with the same config and wrapper as seed2026.
- GPU0: seed2027, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps_hybrid_2018A_petcgdnn_eps01_seed2027_gpu0.log`, launch PID 96643.
- GPU1: seed2028, log `/home/citybuster/Data/RCPS/work_dirs/logs/rcps_hybrid_2018A_petcgdnn_eps01_seed2028_gpu1.log`, launch PID 96644.
- Purpose: stability check only. The seed2026 result is marginal and cannot support a strong 2018A claim unless the remaining seeds consistently improve ECE/Brier while preserving high-SNR accuracy.

### 2026-05-21 18:56 CST - 2018A eps0.1 stability seeds mid-run note

- seed2027 shows substantial early validation volatility: best 60.9217% at epoch22, followed by epoch24/25/26 at 40.1288%/47.2510%/52.3086%, then recovery to 60.4406% by epoch31. No traceback, CUDA OOM, or dataloader/file-handle errors were observed.
- seed2028 reached best 61.2748% at epoch23, but later validation values include 54.9272% at epoch25, 51.6445% at epoch27, and 57.0485% at epoch29. No runtime errors were observed.
- Interpretation: RadioML2018.01A/PETCGDNN with `rcps-hybrid-eps01` is a stability-check setting, not yet a strong main-result setting. Final admission requires 3-seed test CSVs, high-SNR retention, and NLL/ECE/Brier gates. Best-checkpoint export remains the intended guardrail.

### 2026-05-21 22:02 CST - 2018A eps0.1 stability seeds recovered

- Foreground monitoring resumed after interruption. Both stability runs are still active with no runtime errors.
- seed2027 recovered from earlier validation volatility and reached best 62.6222% at epoch78.
- seed2028 reached best 62.9028% at epoch74, exceeding seed2026 validation best 62.8048%.
- Interpretation update: early validation volatility remains a diagnostic concern, but the two stability seeds are no longer stuck in low-quality regimes. Final admission still depends on best-checkpoint test CSVs and high-SNR/NLL/ECE/Brier gates.

### 2026-05-22 06:20 CST - RadioML2018.01A PETCGDNN RCPS-Hybrid eps0.1 stability run completed
- Stage: 2018A/PETCGDNN RCPS-Hybrid eps0.1 stability check after eps0.2 was rejected.
- Commit: `2be6428`.
- Config: `configs/rcps/dpc/petcgdnn_rcps-hybrid-eps01_iq-snr-deepsig-201801A.py`.
- Seeds: 2026, 2027, 2028; all validation/test predictions exported with best checkpoints.
- Checkpoints used: seed2026 epoch181, seed2027 epoch149, seed2028 epoch128.
- Test mean/std vs hard CE:
  - Hard CE accuracy/NLL/ECE/Brier: 61.978686 +/- 1.095368, 1.142802 +/- 0.031264, 0.005021 +/- 0.001265, 0.405207 +/- 0.012744.
  - RCPS-Hybrid eps0.1 accuracy/NLL/ECE/Brier: 62.932630 +/- 0.174679, 1.118974 +/- 0.005931, 0.003787 +/- 0.000191, 0.393793 +/- 0.001936.
  - Mean paired deltas: accuracy +0.953944 pp, NLL -0.023827, ECE -0.001234, Brier -0.011413.
- Interpretation: positive extension evidence on RadioML2018.01A with high-reliability retention satisfied in all inspected seeds, but paired delta variance is large because hard seed2027 is weak. Treat as encouraging but not standalone TPAMI-level proof; requires replication on another strong model/dataset pair before manuscript main claim is strengthened.
- Artifacts: `/home/citybuster/Data/RCPS/work_dirs/rcps_hybrid_2018A/summary/deepsig201801A_petcgdnn_rcps-hybrid-eps01_vs_hard_3seed_overall.csv`.

### 2026-05-22 06:25 CST - Launched MCformer RadioML2018.01A baseline gate
- Motivation: replicate the positive 2018A PETCGDNN RCPS-Hybrid evidence on a different model family before strengthening manuscript claims.
- Added config commit: `2d6da1b` with `configs/rcps/_base_/models/mcformer_iq-snr-deepsig-201801A.py` and `configs/rcps/mcformer/mcformer_hard-ce_iq-snr-deepsig-201801A.py`.
- Launched hard CE baseline seeds 2026/2027 on GPU0/GPU1 via `/home/citybuster/Data/RCPS/work_dirs/logs/run_mcformer_2018A_baseline_gate.sh`.
- Current status: processes initialized successfully, no shape/config errors; both are reading RadioML2018.01A `.npy` files during dataset cache warm-up. No RCPS run will be launched until baseline validation/test metrics are available.

### 2026-05-22 06:38 CST - Stopped MCformer RadioML2018.01A baseline gate due to infeasible runtime
- Observation: both MCformer 2018A hard CE seeds entered epoch 1 successfully with no shape/config/runtime errors, but each epoch has 3195 iterations at roughly 0.78--0.87 seconds/iteration, implying about 40+ minutes per epoch and an 11+ day 400-epoch ETA.
- Action: stopped seed2026/2027 before completing epoch 1; no metrics are used or reported.
- Interpretation: this is a resource/throughput diagnostic, not a model-quality failure. MCformer remains useful on 2016.10A/10B where existing 3-seed results are available. For 2018A cross-family replication, switch to a faster strong baseline such as CGDNet/FastMLDNN if its baseline gate is feasible.

### 2026-05-22 06:48 CST - CGDNet RadioML2018.01A smoke failed due to input-shape incompatibility
- Added config commit: `53b7387` for CGDNet 2018A with SNR metadata.
- Smoke run: seed2026, max_epochs=1.
- Result: initialization passed, but training failed before epoch metrics with `RuntimeError: shape [-1, 50, 472] is invalid for input of size 81120000`.
- Interpretation: CGDNet implementation is not directly compatible with the 1024-sample 2018A IQ shape under the current frame_length assumptions. This is a baseline parity issue; no RCPS conclusions are drawn. Do not use CGDNet on 2018A unless the architecture/input adapter is explicitly fixed and baseline-gated.

### 2026-05-22 06:55 CST - FastMLDNN RadioML2018.01A smoke failed due to head/output mismatch
- Added config commit: `5af0038` for FastMLDNN 2018A with SNR metadata.
- Smoke run: seed2026, max_epochs=1.
- Result: initialization passed, but training failed with `ValueError: Expected input batch_size (24) to match target batch_size (640)` from cross entropy.
- Interpretation: FastMLDNN needs its dedicated output/head handling before it can enter the RCPS baseline-gate matrix. This is a head/interface parity issue, not an RCPS result. Do not use FastMLDNN on 2018A until the baseline head is adapted and gated.

### 2026-05-22 07:20 CST - Speech Commands audio diagnostic consolidated
- Consolidated existing Speech Commands DS-CNN pilot metrics into `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_summary/summary/`.
- 20-epoch seed2026 matched comparison: hard CE accuracy/NLL/ECE/Brier = 82.6078 / 0.5252 / 0.0238 / 0.2434; fixed RCPS-retention eps0.05 = 82.6545 / 0.5324 / 0.0498 / 0.2432.
- Paired deltas: accuracy +0.047 pp and Brier -0.00023, but NLL +0.00718 and ECE +0.02606. At -10 dB, RCPS-retention worsens accuracy by -0.509 pp and worsens NLL/ECE/Brier.
- Interpretation: fixed uniform/retention smoothing is not an admissible audio main result. This diagnostic supports the manuscript claim that RCPS target broadening must be validation-constrained and posterior/base-aware; audio should stay diagnostic unless a posterior-base or sample-posterior audio variant passes validation gates.

### 2026-05-22 07:35 CST - RadioML2018.01A PETCGDNN SNR-region evidence generated
- Generated paired and aggregate SNR-bin summaries for `RadioML2018.01A / PETCGDNN / RCPS-Hybrid eps0.1` under `/home/citybuster/Data/RCPS/work_dirs/rcps_hybrid_2018A/summary/`.
- Three-seed region deltas versus hard CE: low SNR accuracy/NLL/ECE/Brier = +0.390 / -0.0218 / -0.00146 / -0.00277; mid SNR = +1.277 / -0.0270 / -0.000995 / -0.0152; high SNR = +0.837 / -0.0197 / -0.00131 / -0.0117.
- Interpretation: the 2018A improvement is not caused by sacrificing high-reliability recognition; high-SNR retention is satisfied in the region aggregate. The manuscript now reports these region values in the reliability-stratified AMC paragraph, while the detailed figure remains an auditable artifact rather than an additional main-text float.

### 2026-05-22 09:45 CST - MCLDNN RadioML2018.01A baseline smoke
- Added and parsed SNR-aware MCLDNN 2018A baseline-gate config in commit `e3b96f4`.
- Smoke with batch size 256 completed one epoch without shape/head/runtime errors; epoch time is about 8--9 minutes and validation top-1 after one epoch is 38.93%.
- Batch-size sweep: 1024 fails with CUDA OOM; 512 is stable but has similar epoch wall time to 256 and uses about 10.5 GB, so it provides no meaningful throughput advantage.
- Interpretation: MCLDNN is a feasible 2018A second-backbone candidate but full 3-seed baseline gating is expensive. It should enter at most a 50-epoch pilot first; do not launch RCPS variants until the hard CE pilot demonstrates competitive baseline behavior.

### 2026-05-22 09:50 CST - MCLDNN RadioML2018.01A 50-epoch hard CE pilot launched
- Launched a conservative single-seed hard CE pilot for MCLDNN on RadioML2018.01A to test whether it is worth admitting as a second 2018A backbone.
- Command wrapper: `/home/citybuster/Data/RCPS/work_dirs/logs/run_mcldnn_2018A_50ep_pilot.sh`; log: `/home/citybuster/Data/RCPS/work_dirs/logs/mcldnn_2018A_50ep_seed2026_gpu0.log`.
- Settings: seed 2026, batch size 256, max epochs 50, GPU0. This is a baseline feasibility pilot only; no RCPS variant will be launched unless the hard CE trajectory is competitive and stable.

### 2026-05-22 11:12 CST - MCLDNN 2018A 50-epoch pilot epoch-9 check
- Pilot remains healthy on GPU0 with no Traceback/OOM/Error.
- Validation trajectory reached epoch9: epoch8 top-1 = 50.9549%, epoch9 top-1 = 52.1887%; validation loss at epoch9 = 1.4688.
- Interpretation: MCLDNN is learning and remains a feasible second-backbone candidate, but it has not yet passed a baseline admission point. Continue to epoch20 before deciding whether to expand to multi-seed hard CE or stop this branch.

### 2026-05-22 11:24 CST - MCLDNN 2018A 50-epoch pilot epoch-10 check
- Pilot remains healthy on GPU0 with no Traceback/OOM/Error.
- Validation top-1 reached 52.4763% at epoch10; validation loss is 1.4571. Best checkpoint updated to epoch10.
- Interpretation: the baseline is learning but remains far below the PETCGDNN 2018A baseline level. Continue to epoch20 before go/no-go. If it remains below roughly 55%, do not expand MCLDNN 2018A to three seeds or RCPS variants.
### 2026-05-22 12:50 CST - MCLDNN RadioML2018.01A pilot stopped at epoch20
- MCLDNN / RadioML2018.01A hard CE pilot remained healthy through epoch20, with no Traceback/OOM/file-handle errors.
- Validation trajectory plateaued below the baseline gate: epoch18 = 54.1804%, epoch19 = 54.1639%, epoch20 = 54.3536%.
- Decision: stop the branch at epoch20 and keep it as a baseline feasibility diagnostic only. Do not expand MCLDNN 2018A to three seeds and do not launch RCPS variants on this backbone/dataset pair.
- Interpretation: this does not affect the admitted DPC-RCPS evidence. It confirms that the 2018A second-backbone path should not rely on the current MCLDNN configuration without additional architecture/input parity work.

### 2026-05-22 12:45 CST - Audio posterior-base interface and hard teacher export launched
- Added `rcps-confusion` support to `tools/rcps/run_crossmodal_audio.py`, plus optional validation/test posterior export through `--save-predictions`.
- Commit: `70705c8`.
- Launched DS-CNN Speech Commands hard teacher rerun on GPU1 with prediction export, work root `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_confusion_20ep`.
- Purpose: build a validation-only confusion/posterior base for a posterior-aware audio RCPS pilot. No audio positive claim will be made until this passes validation diagnostics.


## 2026-05-22 13:50 CST - Speech Commands posterior/confusion-aware audio pilot completed

- Completed Speech Commands / DS-CNN / seed2026 hard teacher and rcps-confusion pilot under /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_confusion_20ep.
- Hard CE test overall: accuracy 73.5977, NLL 0.7220, ECE 0.0199, Brier 0.3586.
- RCPS-Confusion test overall: accuracy 73.5017, NLL 0.7360, ECE 0.0183, Brier 0.3613.
- Paired deltas (RCPS-Confusion minus Hard): accuracy -0.0961 pp, NLL +0.0140, ECE -0.0016, Brier +0.0027.
- By SNR, RCPS-Confusion improves ECE in most bins and slightly improves low-SNR accuracy at -10 dB, but NLL worsens in every SNR bin and high-reliability accuracy/Brier degrade.
- Decision: this audio posterior/confusion-aware pilot is a diagnostic result, not main positive evidence. It supports the paper stability gate: ECE-only gains are insufficient when NLL/Brier and high-reliability retention fail.
- Summary CSV: /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_confusion_20ep/summary/speechcommands_ds_cnn_rcps_confusion_vs_hard_seed2026.csv.


## 2026-05-22 14:30 CST - Audio DPC-consistency pilots completed and rejected

- Completed two Speech Commands / DS-CNN / seed2026 DPC-consistency pilots with hard CE plus clean-to-noisy posterior consistency.
- Work roots:
  - /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_dpc_consistency_w005_20ep
  - /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_dpc_consistency_w010_20ep
- Compared with the same hard CE baseline, weight 0.05 gives overall deltas: accuracy -0.9281 pp, NLL +0.0318, ECE -0.0010, Brier +0.0111.
- Weight 0.10 gives overall deltas: accuracy -2.1237 pp, NLL +0.0599, ECE -0.0006, Brier +0.0276.
- Low-SNR and clean bins both degrade in accuracy/NLL/Brier. Therefore same-model clean-view consistency is not admitted as positive audio evidence.
- Decision: do not expand Speech Commands DPC-consistency. The result supports the theoretical boundary that DPC requires a reliable and calibrated teacher or validation-approved posterior projection; online same-backbone self-teaching can amplify bias.
- Summary CSV: /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_dpc_consistency_summary/speechcommands_ds_cnn_dpc_consistency_vs_hard_seed2026.csv.

## 2026-05-24 11:30 CST - Speech Commands logmel-resnet audio gate completed

- Completed a stronger Speech Commands / logmel-resnet audio diagnostic.
- Hard CE baseline over seeds 2026/2027/2028 is stable: accuracy 87.704±0.255, NLL 0.3714±0.0073, ECE 0.0058±0.0024, Brier 0.1729±0.0024.
- Low-weight online DPC consistency (weight 0.01) does not pass the gate: accuracy delta -0.149±0.291 pp, NLL delta +0.0044, ECE delta +0.0009, Brier delta +0.0025.
- Single-seed RCPS-retention with epsilon 0.05 improves accuracy and Brier but worsens NLL/ECE, so it remains diagnostic.
- Post-hoc phi gate using hard CE predictions confirms that scalar SNR is not a sufficient sample-level order parameter for audio:
  - SNR bins explain entropy/correctness R2 = 0.257±0.007 / 0.067±0.003.
  - SNR plus hand-crafted audio features explain 0.276±0.007 / 0.071±0.004.
  - Teacher confidence explains 0.949±0.001 / 0.346±0.010.
- Decision: audio is retained as a multidimensional-reliability boundary case. Do not claim Speech Commands as a positive RCPS/DPC result yet. The next audio direction, if pursued, is cross-fit teacher-confidence or learned phi-RCPS with validation gating.
- Summary CSV: /home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_summary/logmel_resnet_phi_gate/summary/logmel_resnet_phi_gate_summary.csv.

## 2026-05-24 12:15 CST - RadioML2018.01A CGDNet gate config repaired

- During the TPAMI evidence-gap audit, the 2018A second-backbone gap was traced to a configuration error rather than an RCPS/DPC algorithm issue.
- Previous `configs/rcps/_base_/models/cgdnet_iq-snr-deepsig-201801A.py` used `frame_length=128`, inherited from 2016-style inputs, while RadioML2018.01A uses 1024 complex samples per example.
- This caused the CGDNet GRU reshape to fail with `RuntimeError: shape '[-1, 50, 472]' is invalid...`.
- Fix: set `frame_length=1024` in the RCPS 2018A CGDNet base config and document why the value differs from 2016A.
- Direct random-tensor forward smoke now passes: `forward_ok (4, 24)` for input shape `(4, 1, 2, 1024)`.
- A full train-loop smoke was intentionally not used as proof because RadioML2018.01A dataloader/cache initialization can exceed a short timeout before the first batch.
- Next admissible step: launch a formal `CGDNet + RadioML2018.01A + Hard CE` baseline gate with seeds 2026/2027/2028 only after confirming expected runtime; no RCPS comparison is allowed on this pair before that gate passes.

## 2026-05-24 12:35 CST - RadioML2018.01A MCLDNN second-backbone prescreen launched

- Historical 2018A checkpoint audit was rejected: existing CNN4/DSCLDNN validation PKLs are near random (`~4.17%` on 24 classes), likely due to old checkpoint / class order / export mismatch, and must not enter the paper.
- A practical second-backbone path is MCLDNN on RadioML2018.01A: seed2026 20-epoch test export completed with overall accuracy `54.2846%` and high-SNR bins around `85%`.
- This is still a prescreen, not paper evidence, because it is a shorter 20-epoch budget and only one completed seed so far.
- Launched detached tmux runs for seed2027 and seed2028:
  - `rcps_mcldnn2018_s2027` on GPU0.
  - `rcps_mcldnn2018_s2028` on GPU1.
- Corrected a launch hygiene issue: an orphan duplicate seed2028 process from an earlier detach attempt was killed to prevent checkpoint contamination; only the tmux-managed seed2028 process remains.
- Next action after both runs finish: export test predictions, run reliability-bin metrics, and decide whether MCLDNN 2018A is strong enough for supplementary evidence or whether the paper still needs CGDNet/FastMLDNN full baseline gate.

## 2026-05-24 12:40 CST - MCLDNN 2018A export watcher installed

- Added runtime watcher `/home/citybuster/Data/RCPS/work_dirs/logs/watch_mcldnn_2018A_20ep_export.sh` outside the git repo.
- Watcher tmux session: `rcps_mcldnn2018_export`.
- Behavior: wait for seed2027/2028 training processes to finish, export test predictions with `num_workers=0`, run `analyze_reliability.py`, and aggregate seed2026/2027/2028 into:
  `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A_mcldnn_20ep/summary/deepsig201801A_mcldnn_hard-ce-20ep_3seed_summary.csv`.
- This watcher is operational glue only; it does not change models, losses, seeds, or evaluation rules.
## 2026-05-24 22:35 CST - Speech Commands phi-RCPS admitted

- Context:
  - Earlier scalar-SNR RCPS, posterior-confusion RCPS, and online DPC audio variants stayed diagnostic because they failed NLL/Brier or high-reliability retention gates.
  - A log-mel ResNet hard-label teacher passed the audio baseline gate, and teacher confidence explained substantially more sample-level entropy/correctness variation than scalar SNR.
- Code:
  - `tools/rcps/export_audio_teacher_predictions.py` committed as `b8d5e4b`.
  - `tools/rcps/run_crossmodal_audio_phi.py` committed as `709a246`.
- Experiment:
  - Dataset: Speech Commands noisy.
  - Model: log-mel ResNet.
  - Method: `phi-teacher`.
  - Seeds: `2026, 2027, 2028`.
  - Training: 20 epochs, `epsilon_max=0.05`, `gamma=1.0`, `retain_min=0.75`.
  - Teacher posterior source: train split only, one artifact per seed.
  - Test protocol: balanced, `test_max_per_label_snr=250`.
- Artifacts:
  - Metrics root: `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_phi_20ep_eps005_ret075/metrics`.
  - Summary: `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_phi_20ep_eps005_ret075/summary_phi_vs_hard_3seed.csv`.
  - Manifest: `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_phi_20ep_eps005_ret075/run_manifest_audio_phi.csv`.
- Three-seed mean delta relative to matched hard CE:
  - Accuracy: `+0.5368 pp`.
  - NLL: `-0.0140`.
  - Brier: `-0.0070`.
  - ECE: `+0.0021`, slightly worse.
  - Accuracy and NLL/Brier improve or stay non-negative in every SNR bin.
  - High-SNR and clean retention pass.
- Interpretation:
  - Audio is no longer only a failure-mode section.
  - Scalar SNR remains insufficient as a sample-level audio order parameter.
  - The admitted positive result supports a learned/model-adaptive reliability projection: teacher confidence plus a small posterior correction and retention gate.
  - The manuscript should keep scalar-SNR audio failures as diagnostics and report the phi-RCPS result as an admitted but teacher-dependent audio positive case.

## 2026-05-24 23:05 CST - RadioML2018.01A ResNet-AMR second-backbone gate opened

- Motivation: the TPAMI evidence audit still has one weak point: RadioML2018.01A has PETCGDNN as an admitted main backbone, but needs a second independent backbone that is not CGDNet/FastMLDNN after their failed parity branches.
- External reference: AMR-Benchmark includes a lightweight ResNet for RML2018.01A with Conv2D residual blocks, Adam lr 0.001, batch size 400, and early stopping.
- Code change:
  - Added `ResNetAMR` to `csrr/models/backbones/resnet_amr.py`.
  - Registered it in `csrr/models/backbones/__init__.py`.
  - Added ordinary and SNR-aware configs under `configs/resnet_amr/` and `configs/rcps/resnet_amr/`.
  - The backbone uses ordinary `ClsHead`, so RCPS comparisons can remain loss/target-only after the hard CE gate passes.
- QA:
  - Config parsing passed for hard CE, static LS, and RCPS-Hybrid configs.
  - Random-tensor forward passed for input layouts `N x L x 2`, `N x 2 x L`, `N x 1 x 2 x L`, `N x 2 x L x 1`, and `N x L x 2 x 1`.
  - Smoke annotation set with 800 train/validation/test samples was generated at `/home/citybuster/Data/RCPS/processed/smoke_annotations/deepsig201801A_800`.
  - Ann800 smoke train/validation/test/export/analyze passed and wrote `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A_smoke/metrics/deepsig201801A_resnet_amr_hard-ce-smoke_seed2026_test.csv`.
- Debug notes:
  - Initial smoke failed because the RCPS 2018A pipeline feeds `N x L x 2`, not `N x 1 x 2 x L`; the backbone now normalizes both layouts internally.
  - A full-dataset smoke with `cache=True` spent several minutes caching all 2018A samples; ann800 smoke is now used for fast chain validation only.
  - The first paired nohup launch started seed2026 but seed2027 inherited the wrong working directory; seed2027 was relaunched with an explicit `cd`.
- Running now:
  - `ResNetAMR + RadioML2018.01A + Hard CE + 20 epochs`, seed2026 on GPU0.
  - `ResNetAMR + RadioML2018.01A + Hard CE + 20 epochs`, seed2027 on GPU1.
- Gate rule unchanged: if 20-epoch hard CE is clearly below the MCLDNN diagnostic level or has unstable high-SNR plateau, ResNet-AMR is quarantined and the next candidate is DenseNet-AMR/DensCNN rather than forcing RCPS.
## 2026-05-25 03:45 CST - ResNet-AMR 2018A long-budget gate manifest update

- Added running manifest rows for `RadioML2018.01A / ResNetAMR / hard-ce-longbudget` seeds `2026` and `2027`.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A_resnet_amr_longbudget`.
- Early sanity through epoch 3: seed2026 val `41.2741%`, seed2027 val `37.4945%`.
- No RCPS paired comparison is allowed until hard CE gate passes.
- DenseNet/DensCNN fallback remains inactive until ResNet-AMR is formally quarantined.
## 2026-05-25 03:46 CST - ResNet-AMR 2018A long-budget epoch 4 sanity

- `RadioML2018.01A / ResNetAMR / hard-ce-longbudget` continues normally.
- Epoch 4 validation: seed2026 `44.5039%`, seed2027 `41.1745%`.
- Both seeds saved `best_accuracy_top1_epoch_4.pth`.
- Error scan is empty.
- Decision: continue hard CE parity gate; no RCPS paired comparison; no DenseNet/DensCNN fallback yet.
## 2026-05-25 04:08 CST - ResNet-AMR 2018A long-budget epoch 5--6 sanity

- Epoch 5 validation: seed2026 `46.2651%`, seed2027 `41.5323%`.
- Epoch 6 validation: seed2026 `46.9775%`, seed2027 `43.5235%`.
- Both seeds continue to refresh best checkpoints.
- seed2027 recovered from the slow epoch-5 increment, so no diagnostic intervention is triggered.
- Decision remains: continue hard CE gate; no RCPS paired comparison; no DenseNet/DensCNN fallback yet.
## 2026-05-25 04:30 CST - ResNet-AMR 2018A long-budget epoch 7--8 sanity

- Epoch 7 validation: seed2026 `46.7044%` with loss `1.6263`; seed2027 `43.9369%` with loss `1.7215`.
- Epoch 8 validation: seed2026 `48.4182%` with loss `1.6063`; seed2027 `44.6712%` with loss `1.7024`.
- Both seeds saved new best checkpoints at epoch 8.
- Error scan remains empty.
- Decision remains: continue hard CE parity gate; no RCPS paired comparison; no DenseNet/DensCNN fallback yet.
## 2026-05-25 04:52 CST - ResNet-AMR 2018A long-budget epoch 9--10 sanity

- Epoch 9 validation: seed2026 `48.8186%` / loss `1.5890`; seed2027 `46.8015%` / loss `1.6826`.
- Epoch 10 validation: seed2026 `50.0819%` / loss `1.5737`; seed2027 `47.1655%` / loss `1.6675`.
- Both seeds saved new best checkpoints at epoch 10.
- Error scan remains empty.
- Interpretation: long-budget training is correcting the short-budget prescreen weakness; keep running the hard CE parity gate. No RCPS and no fallback yet.
## 2026-05-25 05:14 CST - ResNet-AMR 2018A long-budget epoch 11--12 sanity

- Epoch 11 validation: seed2026 `49.1611%` / loss `1.5248`; seed2027 `48.1960%` / loss `1.6238`.
- Epoch 12 validation: seed2026 `50.6273%` / loss `1.4937`; seed2027 `48.8312%` / loss `1.5929`.
- Both seeds saved new best checkpoints at epoch 12.
- Error scan remains empty.
- Decision: keep hard CE gate running; ResNet-AMR is not quarantined; no RCPS until test/SNR-bin gate and seed plan are resolved.
## 2026-05-25 05:36 CST - ResNet-AMR 2018A long-budget epoch 13--14 sanity

- Epoch 13 validation: seed2026 `51.5885%` / loss `1.4683`; seed2027 `49.2461%` / loss `1.5672`.
- Epoch 14 validation: seed2026 `52.0778%` / loss `1.4518`; seed2027 `49.8390%` / loss `1.5458`.
- Both seeds saved new best checkpoints at epoch 14.
- Error scan remains empty.
- Interpretation: long-budget parity is materially stronger than the 20-epoch prescreen. Continue hard CE gate; no RCPS until test/SNR-bin and third seed decisions are resolved.



### 2026-05-25 05:47 CST - ResNet-AMR 2018A long-budget epoch 15 sanity

- Branch/commit during training: `feature/rcps-experiments` / training launched from `73c9e7f`; logging commit was `d631bdd`.
- Stage: `baseline_gate_2018A_resnet_amr_longbudget`, hard CE only.
- Epoch 15 validation:
  - seed 2026: accuracy `52.3008%`, loss `1.4369`, saved `best_accuracy_top1_epoch_15.pth`.
  - seed 2027: accuracy `49.0091%`, loss `1.5268`; best checkpoint remains epoch 14.
- Error scan: no `Traceback`, `CUDA out of memory`, `Too many open files`, `CalledProcessError`, `FileNotFoundError`, or `TypeError`.
- Decision: seed 2027 has an accuracy fluctuation but loss is still improving; continue hard CE parity. Do not start RCPS, and do not switch to DenseNet fallback.


### 2026-05-25 05:55 CST - ResNet-AMR 2018A long-budget go/no-go rule

This is a pre-specified admission rule for the RadioML2018.01A second-backbone gate, added while the long-budget hard-CE run is still in progress.

- Continue seed 2026/2027 until early stopping or a clear plateau.
- Advance to seed 2028 and paired RCPS only if best-checkpoint test mean is close to the MCLDNN diagnostic anchor, provisionally at least about `54.5%`, with normal high-SNR plateau (about `83%+` at the top SNR bins), no large NLL/Brier posterior-quality regression, and no seed collapse.
- Quarantine ResNet-AMR if the long-budget test mean remains more than about `2.5 pp` below the MCLDNN diagnostic anchor, or if the high-SNR plateau/export/analyze pipeline is abnormal.
- If quarantined, switch to the pre-registered DenseNet/DensCNN fallback, which must itself pass hard-CE gate before any RCPS comparison.
- No RCPS comparison is allowed before this hard-CE gate is resolved.


### 2026-05-25 05:56 CST - ResNet-AMR 2018A long-budget epoch 16 sanity

- Branch/commit during training: `feature/rcps-experiments`; training launched from `73c9e7f`; current logging commit before this entry was `21895df`.
- Stage: `baseline_gate_2018A_resnet_amr_longbudget`, hard CE only.
- Epoch 16 validation:
  - seed 2026: accuracy `52.3976%`, loss `1.4255`, saved `best_accuracy_top1_epoch_16.pth`.
  - seed 2027: accuracy `49.3833%`, loss `1.5108`; best checkpoint remains epoch 14.
- Error scan: empty.
- Decision: continue long-budget hard-CE parity. Seed 2027 recovered from the epoch-15 accuracy dip and is not a collapse. Do not start RCPS and do not switch to DenseNet/DensCNN fallback.


### 2026-05-25 06:07 CST - ResNet-AMR 2018A long-budget epoch 17 sanity

- Stage: `baseline_gate_2018A_resnet_amr_longbudget`, hard CE only.
- Epoch 17 validation:
  - seed 2026: accuracy `52.3509%`, loss `1.4137`; best checkpoint remains epoch 16.
  - seed 2027: accuracy `49.4679%`, loss `1.4981`; best checkpoint remains epoch 14.
- Error scan: empty.
- Decision: both seeds are showing accuracy fluctuations while validation loss continues to improve. This is not a runtime failure or immediate collapse. Continue hard-CE parity and wait for the epoch-20 moving-slope check and eventual best-checkpoint test. No RCPS and no DenseNet/DensCNN fallback yet.


### 2026-05-25 06:18 CST - ResNet-AMR 2018A long-budget epoch 18 sanity

- Stage: `baseline_gate_2018A_resnet_amr_longbudget`, hard CE only.
- Epoch 18 validation:
  - seed 2026: accuracy `52.2914%`, loss `1.4051`; best checkpoint remains epoch 16.
  - seed 2027: accuracy `49.1243%`, loss `1.4880`; best checkpoint remains epoch 14.
- Error scan: empty.
- Decision: top-1 accuracy is entering a plateau/fluctuation region while validation loss continues to improve. Continue to the epoch-20 moving-slope check before any go/no-go decision. No RCPS and no DenseNet/DensCNN fallback yet.

### ResNet-AMR long-budget epoch 19 sanity check: 2026-05-25 06:30 CST

- Dataset/model: RadioML2018.01A / ResNet-AMR / hard CE / long-budget gate.
- seed2026 epoch19: val top1 52.5559%, loss 1.3973; best checkpoint refreshed at epoch19.
- seed2027 epoch19: val top1 49.3637%, loss 1.4816; best remains epoch14 at 49.8390%.
- Last-5-epoch slopes: seed2026 acc +0.0638 pp/epoch, loss -0.0099/epoch; seed2027 acc +0.0887 pp/epoch, loss -0.0113/epoch.
- Decision: continue to epoch20 go/no-go; do not start RCPS; do not switch to DenseNet/DensCNN fallback yet.

### ResNet-AMR long-budget epoch20 gate decision: 2026-05-25 06:42 CST

- Dataset/model: RadioML2018.01A / ResNet-AMR / hard CE / long-budget gate.
- seed2026 epoch20: val top1 52.4893%, loss 1.3905; best epoch19 52.5559%.
- seed2027 epoch20: val top1 49.9479%, loss 1.4734; best epoch20 49.9479%.
- Two-seed best mean: 51.2519%, about 3.25 pp below the pre-registered ~54.5% admission line.
- Decision: ResNet-AMR is quarantined for 2018A main-table use; do not run ResNet-AMR RCPS paired comparisons; proceed to DenseNet/DensCNN hard-CE fallback gate.
- ResNet-AMR training processes were stopped after preserving logs and checkpoints.

### DensCNN fallback hard-CE gate launched: 2026-05-25 06:45 CST

- Dataset/model: RadioML2018.01A / DensCNN / hard CE / 20-epoch fallback prescreen.
- Config: configs/rcps/denscnn/denscnn_hard-ce_iq-snr-deepsig-201801A.py.
- Work root: /home/citybuster/Data/RCPS/work_dirs/baseline_gate_2018A_denscnn_20ep.
- seed2026: GPU0, pid 558587; seed2027: GPU1, pid 558583.
- CPU forward smoke passed for (N,1024,2), (N,2,1024), and (N,1,2,1024), all producing (N,24) logits.
- This is hard-CE baseline gate only; no DensCNN Static LS / RCPS before hard-CE gate passes.

### DensCNN fallback epoch 1 sanity check: 2026-05-25 07:14 CST

- seed2026: val top1 33.7616%, loss 1.8407; best epoch1 saved.
- seed2027: val top1 35.6392%, loss 1.8234; best epoch1 saved.
- No error signatures; both runs entered epoch2.
- Decision: DensCNN data/shape/loss path is sane; continue 20-epoch hard-CE prescreen; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 2 sanity check: 2026-05-25 07:30 CST

- seed2026: val top1 39.2895%, loss 1.7806; best epoch2 saved.
- seed2027: val top1 38.4306%, loss 1.7784; best epoch2 saved.
- No error signatures; both runs entered epoch3.
- Decision: DensCNN remains viable through epoch5/10/20; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 3 sanity check: 2026-05-25 08:00 CST

- seed2026: val top1 40.5962%, loss 1.7478; best epoch3 saved.
- seed2027: val top1 41.0680%, loss 1.7357; best epoch3 saved.
- No error signatures; early curve is comparable to ResNet-AMR and more seed-stable.
- Decision: continue hard-CE gate through epoch5/10/20; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 5/6 sanity check: 2026-05-25 08:41 CST

- seed2026: epoch4 val top1 41.8645%, loss 1.7148; epoch5 val top1 42.9346%, loss 1.6904; best epoch5 saved.
- seed2027: epoch4 val top1 42.2611%, loss 1.7091; epoch5 val top1 43.3550%, loss 1.6836; epoch6 val top1 44.7072%, loss 1.6640; best epoch6 saved.
- No error signatures. DensCNN keeps improving and has smaller seed gap than ResNet-AMR.
- Decision: continue 20-epoch hard-CE prescreen; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 9 sanity check: 2026-05-25 09:42 CST

- seed2026: epoch6 44.0768/1.6726; epoch7 45.0516/1.6551; epoch8 45.5700/1.6398; epoch9 45.5900/1.6272.
- seed2027: epoch6 44.7072/1.6640; epoch7 44.6398/1.6477; epoch8 45.5967/1.6324; epoch9 46.0555/1.6211.
- No error signatures. The curve is stable and seed gap is small, but the model is still weak relative to desired 2018A second-backbone gate.
- Decision: continue to 20 epochs before formal gate; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 11/12 mid-gate check: 2026-05-25 10:30 CST

- seed2026: epoch10 45.4823/1.6183; epoch11 47.6643/1.5836, best epoch11.
- seed2027: epoch10 47.3889/1.6094; epoch11 47.5237/1.5769; epoch12 47.7012/1.5529, best epoch12.
- DensCNN still learns and loss decreases, but remains weak for the 2018A second-backbone role so far.
- Decision: continue to epoch20 before formal gate; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 13/14 check: 2026-05-25 11:06 CST

- seed2026: epoch12 47.9692/1.5603; epoch13 49.0291/1.5390, best epoch13.
- seed2027: epoch13 48.2607/1.5369; epoch14 49.6278/1.5190, best epoch14.
- The curve is still improving and seed stability is better than ResNet-AMR, but it has not yet reached the 2018A second-backbone admission level.
- Decision: continue hard-CE gate to epoch20; no RCPS before hard-CE gate passes.

### DensCNN fallback epoch 14 follow-up: 2026-05-25 11:13 CST

- seed2026: epoch14 val top1 48.9820%, loss 1.5238.
- seed2027: epoch14 val top1 49.6278%, loss 1.5190, best epoch14.
- The run remains stable but is likely below the desired 2018A second-backbone admission line; continue to epoch20 for formal gate.
- Decision: no RCPS before hard-CE gate passes.

### DensCNN fallback epoch20 gate decision: 2026-05-25 13:03 CST

- seed2026: best epoch16 val top1 49.9291%, loss 1.4956; epoch20 val top1 49.1685%, loss 1.4643.
- seed2027: best epoch18 val top1 49.7477%, loss 1.4773; epoch20 val top1 49.5494%, loss 1.4624.
- Two-seed best mean: 49.8384%.
- Decision: DensCNN does not meet the ~54.5% 2018A second-backbone admission line and is weaker than ResNet-AMR long-budget. Quarantine DensCNN; do not run seed2028 or any DensCNN Static LS / RCPS paired comparison. Keep 2018A as PETCGDNN large-scale extension, not dual-backbone main evidence.



### Next-stage AMC supplement configs and smoke QA: 2026-05-25 23:31 CST

- Added RCPS-ready configs for HisarMod2019.1 with PETCGDNN and MCformer, plus PETCGDNN on RadioML2016.10B.
- Fixed the original MCformer Hisar config path typo (`datasets/datasets/hisar` -> `datasets/hisar`).
- New configs use `LoadIQFromFile` with `cache=False` for smoke/gate stability, avoiding full Hisar array caching.
- Static config parse passed for all new hard CE, Static LS, and RCPS-Hybrid configs.
- Forward smoke passed: Hisar PETCGDNN `(N,1024,2)->(N,26)`, Hisar MCformer `(N,2,1024)->(N,26)`, 10B PETCGDNN `(N,128,2)->(N,10)`.
- 1-epoch small-sample train/export/analyze smoke passed:
  - HisarMod2019.1 / PETCGDNN / hard CE / seed2026: validation CSV at `/home/citybuster/Data/RCPS/work_dirs/smoke_next_stage/metrics/hisar2019_petcgdnn_hard-ce_seed2026_validation.csv`.
  - HisarMod2019.1 / MCformer / hard CE / seed2026: validation CSV at `/home/citybuster/Data/RCPS/work_dirs/smoke_next_stage/metrics/hisar2019_mcformer_hard-ce_seed2026_validation.csv`.
  - RadioML2016.10B / PETCGDNN / hard CE / seed2026: validation CSV at `/home/citybuster/Data/RCPS/work_dirs/smoke_next_stage/metrics/deepsig201610B_petcgdnn_hard-ce_seed2026_validation.csv`.
- Decision: start formal hard-CE baseline gates for Hisar PETCGDNN and 10B PETCGDNN first. Hisar MCformer remains a second candidate after PETCGDNN gate starts cleanly. No RCPS paired runs before hard-CE gates pass.


### Next-stage hard-CE gate cache recovery: 2026-05-25 23:43 CST

- The first formal Hisar PETCGDNN and RadioML2016.10B PETCGDNN gates were initially too slow with per-sample non-cached `.npy` loading.
- Patched `LoadIQFromFile` to safely skip loading when `AMCDataset(cache=True)` has already placed `iq` in the sample dict, and switched the new Hisar/10B RCPS base configs back to cached loading.
- Commit: `0b238ca` (`Make RCPS IQ loading compatible with cached datasets`), pushed to `origin/feature/rcps-experiments`.
- Relaunched 20-epoch hard-CE baseline gates:
  - HisarMod2019.1 / PETCGDNN / seed2026 on GPU0, work root `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_next_stage_hisar_petcgdnn_20ep_cache`.
  - RadioML2016.10B / PETCGDNN / seed2026 on GPU1, work root `/home/citybuster/Data/RCPS/work_dirs/baseline_gate_next_stage_10B_petcgdnn_20ep_cache`.
- Both runs entered epoch 1 with normal training logs and no `Traceback`, `Too many open files`, or CUDA OOM. No Static LS / RCPS paired comparison before the hard-CE gates pass.


### Next-stage PETCGDNN hard-CE gate seed2026 complete: 2026-05-26 00:13 CST

- RadioML2016.10B / PETCGDNN / hard CE / seed2026 completed 20 epochs, exported validation/test predictions, and produced reliability-bin CSVs.
  - Validation overall accuracy: 62.2133%, NLL: 0.9433, ECE: 0.0235, Brier: 0.4186.
  - Test overall accuracy: 62.2181%, NLL: 0.9419, ECE: 0.0232, Brier: 0.4182.
- HisarMod2019.1 / PETCGDNN / hard CE / seed2026 completed 20 epochs, exported validation/test predictions, and produced reliability-bin CSVs.
  - Validation overall accuracy: 56.6346%, NLL: 1.0624, ECE: 0.0181, Brier: 0.4799.
  - Test overall accuracy: 56.7865%, NLL: 1.0575, ECE: 0.0140, Brier: 0.4785.
- Post-processing used `num_workers=0` and completed without the earlier file-handle failure.
- seed2027 runs have started for both gates. No Static LS / RCPS paired comparisons until all three hard-CE seeds finish and pass gate review.


### Next-stage PETCGDNN hard-CE gate seed2027 complete: 2026-05-26 00:45 CST

- RadioML2016.10B / PETCGDNN / hard CE / seed2027 completed and exported validation/test CSVs.
  - Validation overall accuracy: 62.5100%, NLL: 0.9366, ECE: 0.0102, Brier: 0.4141.
  - Test overall accuracy: 62.5683%, NLL: 0.9353, ECE: 0.0098, Brier: 0.4138.
  - Two-seed stability so far: seed2026 test 62.2181%, seed2027 test 62.5683%; this is a strong candidate for RadioML2016.10B second-backbone admission, pending seed2028.
- HisarMod2019.1 / PETCGDNN / hard CE / seed2027 completed and exported validation/test CSVs.
  - Validation overall accuracy: 55.7673%, NLL: 1.0917, ECE: 0.0104, Brier: 0.4894.
  - Test overall accuracy: 55.7196%, NLL: 1.0875, ECE: 0.0081, Brier: 0.4884.
  - This is slightly below seed2026 but still stable enough to continue seed2028 before deciding whether Hisar is admitted or diagnostic.
- seed2028 runs have started for both gates. No RCPS paired comparisons before seed2028 completes and gate review is logged.


### RadioML2016.10B PETCGDNN hard-CE gate review and paired-comparison launch: 2026-05-26 01:08 CST

- RadioML2016.10B / PETCGDNN / hard CE completed three seeds.
  - Test accuracy: seed2026 62.2181%, seed2027 62.5683%, seed2028 60.1402%.
  - Three-seed test mean/std: accuracy 61.6422±1.3125%, NLL 0.9538±0.0265, ECE 0.0102±0.0021, Brier 0.4250±0.0158.
- Gate decision: pass as RadioML2016.10B second-backbone candidate, with moderate seed variance explicitly tracked. This is not yet an RCPS result.
- Launched paired comparison on GPU1 under `/home/citybuster/Data/RCPS/work_dirs/paired_next_stage_10B_petcgdnn_20ep`:
  - Static LS with smoothing 0.1.
  - RCPS-Hybrid with epsilon max 0.1 and gamma 1.0.
  - Seeds 2026/2027/2028, 20 epochs, validation/test prediction export and reliability-bin analysis.
- HisarMod2019.1 / PETCGDNN / hard CE seed2028 continues on GPU0; no Hisar paired run before hard-CE gate review.


### HisarMod2019.1 PETCGDNN hard-CE gate review and paired-comparison launch: 2026-05-26 01:15 CST

- HisarMod2019.1 / PETCGDNN / hard CE completed three seeds.
  - Test accuracy: seed2026 56.7865%, seed2027 55.7196%, seed2028 55.3754%.
  - Three-seed test mean/std: accuracy 55.9605±0.7358%, NLL 1.0805±0.0204, ECE 0.0091±0.0017, Brier 0.4867±0.0075.
- Gate decision: pass as non-DeepSig AMC extension candidate, pending paired target comparison. This remains extension evidence unless RCPS paired metrics pass the NLL/Brier and retention gates.
- Launched paired comparison on GPU0 under `/home/citybuster/Data/RCPS/work_dirs/paired_next_stage_hisar_petcgdnn_20ep`:
  - Static LS with smoothing 0.1.
  - RCPS-Hybrid with epsilon max 0.1 and gamma 1.0.
  - Seeds 2026/2027/2028, 20 epochs, validation/test prediction export and reliability-bin analysis.


### Next-stage PETCGDNN paired comparison gate decisions: 2026-05-26 04:16 CST

- RadioML2016.10B / PETCGDNN paired comparison completed.
  - Hard CE test mean/std: Acc 61.6422±1.3125, NLL 0.9538±0.0265, ECE 0.0102±0.0021, Brier 0.4250±0.0158.
  - Static LS 0.1 test mean/std: Acc 62.1478±0.2859, NLL 1.0017±0.0088, ECE 0.0498±0.0042, Brier 0.4244±0.0052.
  - RCPS-Hybrid eps0.1 test mean/std: Acc 44.8975±30.2222, NLL 1.4029±0.7792, ECE 0.0139±0.0097, Brier 0.5771±0.2796.
  - Gate decision: reject RCPS-Hybrid for main/extension tables. The method improved over Static LS on the first two seeds but seed2028 collapsed to 10.0000% accuracy and NLL 2.3026. This pair remains a hard-label baseline anchor and target-gate stability diagnostic.

- HisarMod2019.1 / PETCGDNN paired comparison completed.
  - Hard CE test mean/std: Acc 55.9605±0.7358, NLL 1.0805±0.0204, ECE 0.0091±0.0017, Brier 0.4867±0.0075.
  - Static LS 0.1 test mean/std: Acc 55.7217±0.3486, NLL 1.1661±0.0123, ECE 0.0526±0.0041, Brier 0.4945±0.0039.
  - RCPS-Hybrid eps0.1 test mean/std: Acc 55.6467±0.3331, NLL 1.1262±0.0107, ECE 0.0229±0.0061, Brier 0.4925±0.0032.
  - Gate decision: reject RCPS-Hybrid for main/extension tables. Static LS is a clear negative control; RCPS-Hybrid is safer than Static LS but still worse than Hard CE in Acc/NLL/ECE/Brier. Hisar remains a non-DeepSig hard-label anchor and a target-gate diagnostic.

- Interpretation for the manuscript: these results strengthen the admission protocol rather than the positive table. They should be described as evidence that fixed smoothing and simple Hybrid targets are insufficient outside admitted settings. Follow-up should use Critical-RCPS or PosteriorBase-RCPS with validation target gates, not a wider fixed-epsilon Hybrid sweep.


### Critical-posterior PETCGDNN pilot decisions: 2026-05-26 05:05 CST

- Built reliability-conditioned posterior-table targets from hard-CE validation predictions and ran seed2028 pilots for RadioML2016.10B / PETCGDNN and HisarMod2019.1 / PETCGDNN.
- RadioML2016.10B / PETCGDNN / RCPS-Critical-Posterior seed2028 failed the target gate.
  - Test Acc/NLL/ECE/Brier: 10.0000 / 2.3026 / 0.0028 / 0.9000.
  - Paired vs Hard CE seed2028: Acc -50.1402 pp, NLL +1.3184, ECE -0.0097, Brier +0.4569.
  - Decision: reject and do not expand; record as soft-target collapse diagnostic for this model-dataset pair.
- HisarMod2019.1 / PETCGDNN / RCPS-Critical-Posterior seed2028 passed the single-seed pilot.
  - Test Acc/NLL/ECE/Brier: 56.5262 / 1.0671 / 0.0069 / 0.4810.
  - Paired vs Hard CE seed2028: Acc +1.1508 pp, NLL -0.0294, ECE -0.0041, Brier -0.0121.
  - Decision: expand only this Hisar critical-posterior setting to seeds 2026/2027. It remains pending three-seed admission and must not be written as a completed positive result yet.
- Manuscript interpretation: 10B/PETCGDNN strengthens the quarantine/target-gate narrative; Hisar critical-posterior is the current best non-DeepSig AMC extension candidate.

### Hisar critical-posterior three-seed expansion launch: 2026-05-26 05:08 CST

- Launched HisarMod2019.1 / PETCGDNN / RCPS-Critical-Posterior seeds 2026 and 2027 after seed2028 passed the single-seed pilot.
- Work root: `/home/citybuster/Data/RCPS/work_dirs/pilot_critical_posterior_hisar_20ep`.
- Logs: `pilot_critical_posterior_hisar_seed2026.log` and `pilot_critical_posterior_hisar_seed2027.log`.
- Admission rule remains unchanged: three-seed NLL/Brier majority improvement with high-reliability retention; otherwise diagnostic only.


### Hisar critical-posterior three-seed gate completed: 2026-05-26 05:40

`HisarMod2019.1 / PETCGDNN / RCPS-Critical-Posterior` seeds 2026/2027/2028 finished train, prediction export, and reliability analysis under `/home/citybuster/Data/RCPS/work_dirs/pilot_critical_posterior_hisar_20ep`.

Three-seed summary:

- Hard CE Acc/NLL/ECE/Brier: `55.9605±0.7358 / 1.0805±0.0204 / 0.0091±0.0017 / 0.4867±0.0075`.
- RCPS-Critical-Posterior: `56.2609±0.5987 / 1.0742±0.0194 / 0.0096±0.0023 / 0.4832±0.0058`.
- Paired deltas: Acc `+0.3004±0.7367 pp`, NLL `-0.0063±0.0203`, ECE `+0.0005±0.0041`, Brier `-0.0034±0.0075`.

Gate decision: not admitted to the main table. The mean trend is slightly positive, but NLL/Brier improvement is not present in the majority of seeds; seed2028 carries the gain while seeds 2026/2027 are slightly worse. Record as `promising but not admitted` non-DeepSig diagnostic.


### Audio cross-fit phi-RCPS launched: 2026-05-26 05:54

Purpose: test whether the `Speech Commands / Logmel-ResNet / phi-RCPS` evidence depends on same-seed teacher self-confirmation. Cross-fit teacher mapping: student seed2026 uses teacher checkpoint seed2027, seed2027 uses teacher checkpoint seed2028, and seed2028 uses teacher checkpoint seed2026.

Teacher artifact QA passed before training: each train posterior has 46200 samples, shape `(46200, 11)`, contiguous sample_idx, and normalized probabilities.

Launched seed2026 and seed2027 under `/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_phi_crossfit_20ep_eps005_ret075`; seed2028 will start after one GPU is free. Admission rule: NLL/Brier majority-seed improvement against hard CE, no material accuracy loss; ECE alone is insufficient.


### Audio cross-fit phi-RCPS gate completed: 2026-05-26 06:20

`Speech Commands / Logmel-ResNet / cross-fit phi-RCPS` completed for seeds 2026/2027/2028. Teacher mapping: student 2026 <- teacher 2027, student 2027 <- teacher 2028, student 2028 <- teacher 2026.

Three-seed overall test:

- Hard CE Acc/NLL/ECE/Brier: `87.7039±0.2545 / 0.3714±0.0073 / 0.0058±0.0024 / 0.1729±0.0024`.
- Cross-fit phi-RCPS: `87.9723±0.1696 / 0.3629±0.0060 / 0.0075±0.0023 / 0.1690±0.0017`.
- Paired deltas: Acc `+0.2684±0.1926 pp`, NLL `-0.0085±0.0017`, ECE `+0.0017±0.0009`, Brier `-0.0039±0.0014`.

Gate decision: admitted as audio learned-reliability evidence with ECE tradeoff. This strengthens the audio result against the same-seed teacher criticism, but it is not scalar SNR universality.


### Hisar / MCformer smoke passed: 2026-05-26 06:48

`HisarMod2019.1 / MCformer / Hard CE` seed2026 1-epoch smoke completed under `/home/citybuster/Data/RCPS/work_dirs/smoke_hisar_mcformer_1ep`. The run completed train, validation export, test export, and reliability-bin analysis without traceback/OOM/file-handle errors. Validation accuracy after one epoch was `33.1029%`; test accuracy was `33.16%`.

Decision: configuration and metadata path are valid, but this is not a baseline gate. MCformer on Hisar is a possible non-DeepSig second-backbone candidate; a 10--20 epoch single-seed pilot should precede any three-seed hard gate because one epoch is substantially heavier than PETCGDNN.


### Hisar / MCformer 10-epoch hard pilot launched: 2026-05-26 06:49

After the 1-epoch smoke passed, launched a single-seed 10-epoch hard-label pilot for `HisarMod2019.1 / MCformer` under `/home/citybuster/Data/RCPS/work_dirs/pilot_hisar_mcformer_hard_10ep`. This is not a main result; it is a cost-controlled check before deciding whether to spend a three-seed long-budget baseline gate.


### Hisar / MCformer 10-epoch hard pilot completed: 2026-05-26 09:25

`HisarMod2019.1 / MCformer / Hard CE` seed2026 10-epoch pilot completed under `/home/citybuster/Data/RCPS/work_dirs/pilot_hisar_mcformer_hard_10ep` with validation and test prediction export plus reliability-bin analysis.

Validation trajectory: epoch1 `32.9654%`, epoch2 `38.3567%`, epoch3 `40.6154%`, epoch4 `42.9192%`, epoch5 `45.1760%`, epoch6 `46.0404%`, epoch7 `46.7240%`, epoch8 `47.1269%`, epoch9 `46.8962%`, epoch10 `48.6481%`.

Overall test metrics from `hisar2019_mcformer_hisar_hard-ce_hard_seed2026_test.csv`: Acc/NLL/ECE/Brier `48.6900 / 1.3225 / 0.0069 / 0.5510`.

Gate decision: rejected as a weak second-backbone baseline for Hisar. The training/export/analyze chain is healthy, but the 10-epoch hard-label accuracy remains far below the Hisar PETCGDNN hard-label anchor around `56%`. Do not launch three-seed hard gate or RCPS paired comparison for this backbone unless a new independently stabilized recipe is introduced.

### Hisar / MCLDNN smoke failed: 2026-05-26 09:38

HisarMod2019.1 / MCLDNN / Hard CE seed2026 1-epoch smoke failed before training with `ValueError: cannot reshape array of size 2048 into shape (1,2,128)`. This is a config-parity/input-length issue: the current Hisar samples are 2x1024 while the legacy MCLDNN Hisar config uses 2x128 reshape. Do not count this as model or RCPS evidence. A corrected frame_length=1024 SNR-aware config is required before any baseline gate.

### Hisar / MCLDNN corrected smoke passed: 2026-05-26 09:48

Added a corrected SNR-aware MCLDNN Hisar config using `frame_length=1024` and `Reshape [1,2,1024]`, replacing the legacy 2x128 input assumption. The corrected 1-epoch hard-label smoke completed train, validation prediction export, test prediction export, reliability analysis, and efficiency export under `/home/citybuster/Data/RCPS/work_dirs/smoke_hisar_mcldnn_corrected_1ep`.

Results: validation accuracy `36.2548%`; test Acc/NLL/ECE/Brier `36.1054 / 1.8246 / 0.0109 / 0.6791`. This is not a baseline gate, but it is a valid chain and a stronger 1-epoch start than the MCformer Hisar smoke. Decision: commit the corrected config and run a single-seed 10-epoch hard-label pilot before considering any three-seed gate.

### Hisar / MCLDNN corrected 10-epoch hard pilot launched: 2026-05-26 09:52

After the corrected 1-epoch MCLDNN Hisar smoke passed, launched a single-seed 10-epoch hard-label pilot under `/home/citybuster/Data/RCPS/work_dirs/pilot_hisar_mcldnn_corrected_10ep`. This is a cost-controlled baseline-gate pilot, not a main result. Admission decision will compare its validation/test trajectory against the Hisar PETCGDNN hard-label anchor around 56% and the rejected MCformer pilot at 48.69%.
