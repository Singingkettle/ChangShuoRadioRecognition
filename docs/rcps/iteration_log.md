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
