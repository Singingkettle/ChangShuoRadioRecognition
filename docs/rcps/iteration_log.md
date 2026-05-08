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
