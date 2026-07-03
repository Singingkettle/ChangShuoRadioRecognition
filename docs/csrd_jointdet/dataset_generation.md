# CSRD (twc profile) dataset regeneration — noise fix + protocol

Companion docs: `README.md` / `paper_and_history_notes.md` in this directory
(JDM method, model-side expectations). This file documents the **data side**:
why the dataset was regenerated, the exact root cause of the
"noise added repeatedly" (SNR 重复添加噪声) defect, the fix, the generation
protocol, and the empirical SNR verification.

## 1. Generator

- Repo: <https://github.com/Singingkettle/ChangShuoRadioData>, `twc/` folder
  (the simulation code of the TWC paper, DOI `10.1109/TWC.2024.3450972`).
- Local working clone: `~/Projects/ChangShuoRadioData-twc`
  - upstream base: commit `c3eb8d0` (twc folder introduced in `3d38a7d`,
    2026-02-03)
  - **fix commit (local only, not pushed): `78b086b`**
    `fix(twc): store noise exactly once per frame to prevent repeated noise stacking`
- Toolchain: MATLAB R2024a (`~/Applications/MATLAB/R2024a`), headless
  (`matlab -batch`), CPU only. Requires Communications / Signal Processing /
  DSP System toolboxes (all present on this machine).

## 2. Root cause of the repeated-noise defect

Three generations of the script were audited (all in the generator repo's git
history):

1. **Original script** `ref/DataSimulationTool/generate.m` @ `edb0323`
   (2024-01). In the AWGN section (line 116) **every** sub-signal got its own
   `awgn(new_sub.data, dB)` call. The received frame is the *sum* of the
   sub-signals, so a frame with N signals accumulated N independent noise
   realizations: noise power ×N, i.e. effective SNR = label − 10·log10(N)
   (−6 dB for 4 signals). This is the literal "noise added repeatedly in the
   SNR section".
2. **Guarded revision** @ `0241d26` (2024-05) — the version that produced the
   OLD on-disk dataset (`/home/citybuster/Data/WirelessRadio/data/ChangShuo`,
   dated 2024-05-22). It added `if sub_signal_index == 1` around the `awgn`
   calls, so the wideband sum carries the noise exactly once (we verified this
   empirically, see §5). But it introduced/kept three other defects:
   - `real`/`real_awgn` versions: sub-signals 2..N were saved **without the
     fading channel** (the `c(...)`/clock-offset output was computed and then
     dropped because `new_sub.data` was only assigned inside the
     `sub_signal_index == 1` branch);
   - v104 (`real`): `awgn` was applied to the **pre-channel** signal, so the
     whole Rician/Rayleigh + clock-offset processing of sub-signal 1 was
     discarded;
   - `awgn(x, dB)` without `'measured'` assumes unit input power — wrong
     reference after a fading channel, so `real_awgn` SNR labels did not match
     the data (verified: post-channel sub-1 total power varies 0.33–9.8 where
     1 + noise was expected).
3. **Current upstream `twc/generate.m`** @ `3d38a7d` (2026-02) introduced
   `add_wideband_awgn` (noise realized once at the wideband level, correct
   `wideband_data`), **but still added the *same* noise vector to every
   sub-signal's saved `signal_data`** (pre-fix lines 120–123, 180–183,
   219–223). The toolbox's established consumption path composes the frame by
   summing `signal_data` (historical `tools/convert_datasets/cache_csrr.py`
   line 36 `np.sum(x, axis=0)`; current `LoadCSRDFrame`), which stacks the
   identical noise vector N times → noise amplitude ×N → **noise power ×N²**,
   effective SNR = label − 20·log10(N) (−12 dB for 4 signals). A latent
   second bug was also found: `add_clock_offset` uses `interp1`, which
   extrapolates the tail with NaN when the clock factor C < 1; one NaN
   poisoned the power computation and caused the saved `wideband_data` of the
   `real`/`real_awgn` versions to be **all zeros** (confirmed in a smoke run
   of the unfixed code).

## 3. The fix (generator commit `78b086b`)

Minimal changes in `twc/`:

- `generate.m`: the frame's single AWGN realization is stored **only** in
  `wideband_data` (= sum of post-channel sub-signals + noise). `signal_data`
  now holds the noise-free post-channel sub-signals, so no consumer can stack
  noise by summing. SNR labels are unchanged in meaning: per-signal power
  (average across the frame's sub-signals) over total wideband noise power.
- `add_clock_offset.m`: NaN tail from `interp1` replaced with zeros.
- `generate.m` became a function `generate(num_items, output_root)` with
  `rng(0, 'twister')` for reproducibility (defaults preserve old behavior).

Consumer-side change in this repo (same commit as this doc):
`csrr/datasets/transforms/csrd.py::LoadCSRDFrame` now prefers
`wideband_data` when present and only falls back to summing `signal_data`
(correct for the noise-free configurations and for the old export, where the
single noise realization lives inside sub-signal 1).

## 4. Generation protocol (per the paper / twc profile)

| parameter | value |
|---|---|
| sample rate | 150 kHz |
| frame length | 1200 samples (12000 synthesized, decimated ×10 by design) |
| modulations | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| samples per symbol | {10, 12, 15} (bandwidth diversity) |
| signals per frame | recursive placement with protect gap 2·BW, ≈2–5 |
| SNR grid | −8:2:30 dB (20 levels) |
| channel configs (124 versions) | v1 ideal; v2–v71 Rician (7 speeds × K=1..10); v72–v78 Rayleigh (7 speeds); v79–v98 AWGN (20 SNRs); v99–v103 clock offset (max 1,3,5,7,9 ppm); v104 "real" (random fading+offset+SNR); v105–v124 "real_awgn" (fading+offset, fixed SNR each) |
| frames per version | 1000 |

Output (do **not** delete the old dataset, kept for comparison):

- old: `/home/citybuster/Data/WirelessRadio/data/ChangShuo` (8.4 GB)
- new: `/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026`
  (~9–11 GB expected; 124 versions × 1000 items)

Layout per version (schema consumed by `csrr/datasets/csrd.py`):

```
v<k>/
  anno/000001.json ... 001000.json     # per-frame parallel arrays:
                                       # center_frequency, bandwidth, snr,
                                       # modulation, channel, sample_rate,
                                       # sample_num, sample_per_symbol, file_name
  sequence_data/iq/000001.mat ...      # signal_data  (num_signals, 2, 1200)  noise-FREE
                                       # wideband_data (1, 2, 1200)           received frame,
                                       #   present only for awgn-*/real/real_awgn-* versions
```

No split files are written; `CSRDDetectionDataset` / `CSRDModulationDataset`
apply the deterministic seeded 50/10/40 train/validation/test split per
version at load time (see `README.md`), so **no conversion step is required**
for the new export.

Launch command (durable, CPU-only):

```bash
cd ~/Projects/ChangShuoRadioData-twc/twc
nohup setsid matlab -batch \
  "generate(1000, '/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026')" \
  > /home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026/generate.log 2>&1 &
```

## 5. Empirical SNR verification

Method: for a frame with clean reference available, noise is reconstructed
exactly and compared with the label
(`tools/misc/verify_csrd_snr.py`).

**Old dataset** (all 1000 frames of v79, awgn −8 dB): noise present in
exactly one sub-signal per frame (histogram {1: 1000}); wideband-sum SNR −
label: mean +0.004 dB, std 0.13 → the old AWGN versions were *not*
double-noised on disk, but the `real`/`real_awgn`/v104 defects of §2.2 stand,
and any consumer that re-derived per-signal crops from `signal_data` saw
inconsistent noise (all noise in sub-signal 1, none in the others).

**New dataset** (smoke run, fixed generator; measured − label in dB):

| version | label | item 1 | item 2 | `signal_data` residual vs clean |
|---|---|---|---|---|
| v79 | −8 | −0.13 | +0.05 | 0 (noise-free ✓) |
| v84 | +2 | +0.02 | +0.03 | 0 |
| v89 | +12 | +0.22 | −0.05 | 0 |
| v94 | +22 | −0.16 | −0.19 | 0 |
| v98 | +30 | −0.02 | −0.07 | 0 |
| v105 (real_awgn) | −8 | +0.20 | −0.04 | n/a (measured post-channel ref) |
| v115 (real_awgn) | +12 | +0.04 | −0.02 | |
| v124 (real_awgn) | +30 | −0.05 | +0.05 | |

Before the `add_clock_offset` NaN fix the `real_awgn` rows were degenerate
(`wideband_data` ≡ 0); after the fix they match the labels within ±0.2 dB.
The residual ±0.2 dB spread is the natural per-realization variance of a
1200-sample noise estimate, not a bias.

After generation completes, re-run the full check:

```bash
python tools/misc/verify_csrd_snr.py \
  --data-root /home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026
```

## 6. Status / blockers

- Generation runs CPU-only (GPUs untouched, per the active training sweep).
- MATLAB present and licensed — no blocker.
- Disk: 7.1 TB free before generation — no blocker.
