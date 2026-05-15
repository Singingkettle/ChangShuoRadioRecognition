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
