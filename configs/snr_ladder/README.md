English | [简体中文](README_zh-CN.md)

# SNR-Ladder — Do SNR-Aware Training Gains Survive a Frozen-Model Readout?

> Anonymous authors, "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).

## Method in one paragraph

The paper audits SNR-aware training-time supervision for automatic modulation
classification against two references. The first is a pre-registered **null
ladder** fit on the *frozen* hard-baseline model's validation logits: per-SNR-bin
maps of increasing capacity, `F_shift ⊂ F_VS ⊂ F_aff` (per-bin constant shift ⊂
scale+shift ⊂ full affine). A training-time claim is admitted only if it
significantly beats the strictest applicable rung in the transition band with
high-SNR retention; otherwise the free readout collects the same gain without
retraining.
The second reference is an **exact per-SNR Bayes ceiling** computed on a
synthetic clean-paired AWGN benchmark whose generator is bit-exactly known
(factorized constellation likelihoods; a 40-state phase-lattice forward pass for
CPFSK; an unbiased sequential-importance-sampling correction for the generator's
frame normalization, with effective-sample-size certificates). The distance from
a frozen model to that ceiling decomposes into a decision deficit (removable for
free by the per-bin readout) and a representation deficit (owned by the
backbone); across the audited spectrum of SNR-aware routes, no method collects
more than the decision term.

## Paper section → code map

| paper | code |
| --- | --- |
| Null ladder, rungs and admission rule | `scripts/ladder/ladder_lib.py`, `scripts/ladder/ladder_audit.py` |
| Matched-pair audit (hard vs method) | `scripts/ladder/pair_ladder.py`; single-model margin: `scripts/ladder/ladder_only.py` |
| SNR-aware route spectrum (focal / curriculum / snr-weight / FiLM) | `p2/` configs + `scripts/ladder/p2_spectrum.py` |
| Feature-level probes (decision vs representation verdicts) | `scripts/ladder/collect_features.py`, `scripts/ladder/representation_probe_generic.py` |
| Exact Bayes ceiling with certificates | `scripts/ceiling/exact_alrt.py` (`run_tier_e.sh`, `run_sis.sh`) |
| Ceiling decomposition table | `scripts/decomp/decomp_table.py` (reads `results/ceiling_final.csv`) |
| Architecture-invariance check | `scripts/decomp/arch_invariance.py` |
| Label-free whitening ladder (V0–V3) | `scripts/decomp/whitening_ladder.py` |
| Mechanism statistics (S_drift / S_rot / plug-in deficit) | `scripts/decomp/familyd_mech.py` |
| Estimator sandwich (ensemble readout, 1-NN bounds) | `scripts/decomp/sandwich_run.py`, `scripts/decomp/merge_sandwich.py` |
| Proposition premise check (per-bin QDA − LDA) | `scripts/decomp/qda_lda_premise.py` |
| Soft-bin marginalized readout under SNR-estimate error | `scripts/deploy/run_softbin_scan.py`, `scripts/deploy/softbin_lib.py` |
| Paper figures (ceiling overlay, decomposition waterfall) | `scripts/figs/` (read `results/`) |
| Training configs: baselines and DPC pairs | `cgdnet/ cnn2/ denscnn/ dscldnn/ fastmldnn/ gru2/ mcformer/ mcldnn/ mldnn/ petcgdnn/ resnet_amr/ ucsd/ dpc/` |
| Synthetic AWGN anchor benchmark | `synthetic_awgn/` configs + `scripts/synthetic_awgn/` generator |
| DPC / RCPS losses, P2 losses, FiLM backbones | `csrr/models/losses/rcps_loss.py`, `csrr/models/losses/p2_losses.py`, `csrr/models/backbones/{petcgdnn,mcformer}_film.py`, `csrr/models/classifiers/snr_film.py` |

## Data

Public benchmarks follow the repository's standard layout under
`data/ModulationClassification/` (DeepSig RadioML2016.10A/B and 2018.01A,
UCSD RML22, HisarMod2019.1); see `docs/dataset/`.

The synthetic clean-paired AWGN anchor is regenerated from scratch:

```bash
# MATLAB (source of truth; ~minutes)
matlab -batch "cd configs/snr_ladder/scripts/synthetic_awgn; generate_synthetic_awgn_amc('data/synthetic_awgn_amc_v1', 1000, 128, 2026, '')"
# or the python fallback (numerically matched generator)
python configs/snr_ladder/scripts/synthetic_awgn/generate_python_fallback.py --output-root data/synthetic_awgn_amc_v1
python configs/snr_ladder/scripts/synthetic_awgn/validate_synthetic_awgn.py
```

Heavy artifacts are not committed and are regenerated: prediction pickles
(`{pps, gts, snrs}` per split) are dumped by running `tools/test.py` on each
seed's best checkpoint; penultimate-layer features come from
`scripts/ladder/collect_features.py`; DPC teacher posteriors are built from the
matched hard run's train-split predictions (`base.source` in the `dpc/`
configs); AWGN-posterior DPC targets come from
`scripts/synthetic_awgn/make_awgn_dpc_targets.py`.

## Train / evaluate

```bash
python tools/train.py configs/snr_ladder/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610B.py
python tools/test.py configs/snr_ladder/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610B.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
# ladder audit of a matched (hard, method) pair
python configs/snr_ladder/scripts/ladder/pair_ladder.py \
    --hard work_dirs/<hard-run-root> --method work_dirs/<method-run-root> \
    --out work_dirs/pair_ladder.csv --tag "<cell name>"
# exact Bayes ceiling on the synthetic anchor
bash configs/snr_ladder/scripts/ceiling/run_tier_e.sh
bash configs/snr_ladder/scripts/ceiling/run_sis.sh
```

## Results

Headline audit (three seeds per cell; per-bin transition-band accuracy; CIs use
Student-t with n−1 degrees of freedom on seed means):

| finding | measurement |
| --- | --- |
| Matched pairs where the method significantly beats the frozen model's per-bin affine reread | 0 of 19 (8 backbones, 7 dataset labels) |
| Pairs where the per-bin constant rung already matches the method | 15 of 19 |
| Exact Bayes ceiling `Acc*` (synthetic anchor, spliced exact/SIS) | 15.6 % at −20 dB, 67.6 % at 0 dB, 100.0 % at +18 dB (±0.2 pp) |
| In-band distance to the ceiling (PETCGDNN / MCformer / CGDNet) | 6.6 / 6.3 / 12.0 pp |
| ... of which decision deficit (hard→readout) | −0.1 / +0.0 / +2.1 pp |
| ... of which representation deficit (readout→ceiling) | +6.6 / +6.3 / +9.9 pp |
| Cross-architecture variance ratio after readout (five datasets) | 0.73–1.16 (no compression: the ceiling is model-independent, the distance is not) |

Per-bin curves and the exact ceiling table ship in `results/`
(`ceiling_final.csv`, `tier_e_ceiling.csv`, `sis_correction.csv`,
`decomp_synA.csv`); the audit scripts print the full per-cell tables.

## Documented deviations / notes

- **RML22 normalization**: RML22 IQ amplitudes are ~2 orders of magnitude below
  DeepSig's; MCformer/PETCGDNN collapse to chance without a per-sample
  `SelfNormalize` at the front of the pipeline. CNN4 trains without it and its
  pair was audited as-is. The collapsed v1 runs are recorded, not hidden.
- **MCformer on 2018.01A** uses `Reshape [2, 1024]` (the PETCGDNN-style
  transpose pipeline feeds it 1 channel and it cannot train).
- **t-CI correction**: `ladder_audit.py` uses `t(0.975, n−1)`; an earlier
  version used df=2 (or 1.96) for two-seed rows, understating those intervals.
- **FiLM scale**: `film_scale=1.0` collapsed 3/3 seeds on PETCGDNN/10B;
  the audited configuration uses `film_scale=0.1` (recorded honestly).
- **Not included**: the M2M4 blind-SNR closed loop (its server-side moment
  estimator is not curated here); Cover–Hart sandwich numbers are upper bounds
  only and are not extrapolated (uncalibrated 1-NN inversion runs +3–9 pp high).
- Configs are published in cleaned `_base_` form (content-equivalent to the runs);
  exploratory siege variants and framework-parity one-offs are excluded.
- This folder adds `scripts/` and `results/` subfolders (small, KB-scale
  measured tables consumed by the figure scripts) — a first for this
  repository's config dirs.
- `dscldnn/` keeps the upstream modality token `ap-iq` (it inherits
  `_base_/datasets/hisar/ap-iq-hisar2019.py`), unlike the `iq-ap` order used by
  `fastmldnn/` and `mldnn/`.
