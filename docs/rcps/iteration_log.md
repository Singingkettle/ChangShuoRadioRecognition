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
