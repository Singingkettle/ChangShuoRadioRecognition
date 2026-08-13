# CSRD / CRML23 Dataset Scale Audit

Date: 2026-07-06

Scope: compare the JDM paper (Xing et al., IEEE TWC 2024, arXiv:2405.00736) against
on-disk data at `data/ChangShuoTwc2026` and the
`twc` generator in `ChangShuoRadioData (twc/)`.

## Summary

**Scale matches the `twc` toolchain (124 versions × 1000 frames = 124k frames).**
Several **distribution and metadata mismatches** vs the paper text remain; they
affect direct numeric comparison to published figures but do not block training on
this export.

## Paper vs on-disk comparison

| Item | Paper (Sec. IV / Table I / VI-A) | Our data (`ChangShuoTwc2026`) | Match? |
|---|---|---|---|
| Dataset name | CRML23 (CSRD `twc` profile) | Same toolchain / profile | Yes |
| Channel configs | Rician + Rayleigh (+ extended conditions in code) | 124 versions: ideal×1, Rician×70, Rayleigh×7, AWGN×20, clock×5, real×1, real_awgn×20 | Yes (matches `twc/generate.m`) |
| Frames per version | Not stated explicitly; `twc` default **1000** | **1000** every version (v1–v124) | Yes |
| Total frames | Not stated; implied **124k** via generator | **124,000** | Yes |
| Total signals | Not stated | **434,124** (avg **3.50** / frame) | — |
| Frame length (stored I/Q) | **1200** samples (`2×1200`) | **1200** (`signal_data`: `(N, 2, 1200)`) | Yes |
| Sample rate | **150 kHz** | **150 kHz** | Yes |
| Modulations | BPSK, QPSK, 8PSK, 16QAM, 64QAM | Same 5 classes, ~uniform counts | Yes |
| Samples per symbol | 3 symbol-rate clusters (bandwidth diversity) | `{10, 12, 15}` in generator | Yes |
| SNR grid (Table I) | **[12:30:2] dB** (10 levels) | Generator **`−8:2:30`** (20 levels); AWGN v79–v98 + real_awgn v105–v124 use full grid | **Partial** — superset of paper Table I |
| Path delays / gains | `[0,1.8,3.4]×10⁻⁷`, `[0,−2,−10]` dB | Same in `generate.m` | Yes |
| K-factor (Rician) | `[1:10:1]` | K=1..10 per speed in v2–v71 | Yes |
| Max Doppler | 4 Hz (nominal); swept speeds 0:2:12 m/s | Same speed grid | Yes |
| Max clock offset | 5 ppm (Table I); v99–v103 use 1:2:9 ppm caps | Same | Yes |
| Signals per frame (Fig. 2c) | Top-3 counts: **4 (33.3%), 5 (59.0%), 6 (6.0%)**; 3 & 6 uncommon | **3 (49.6%), 4 (47.4%), 5 (1.9%)**; **no 6-signal or empty entries** | **No** |
| Bandwidth AP bins | `(0,110), (110,130), (130,150)` FFT bins | Empirical clusters **~96, ~120, ~146** bins | **Partial** — thresholds OK, cluster centers shifted |
| Anchor widths (detector) | 3 anchors aligned to bandwidth clusters / AP bins | Promoted **96/120/146** (empirical); paper text **110/130/150** | **Partial** |
| AR@k caps (VI-A) | **AR@4, AR@5, AR@6** from signal-count distribution | Should use **AR@3, AR@4, AR@5** for our distribution | **No** (metric definition) |
| Train/val/test split | **Not specified** in paper | Deterministic **50/10/40** per version, seed 0 | Unknown (repo convention) |
| Train frames | — | **62,000** (50.0%) | — |
| Val frames | — | **12,400** (10.0%) | — |
| Test frames | — | **49,600** (40.0%) | — |
| AMC train signals | — | **214,892** (49.5%) | — |
| AMC val signals | — | **43,772** (10.1%) | — |
| AMC test signals | — | **175,460** (40.4%) | — |
| `sample_num` metadata | — | Annotated **12000** (synthesis length); stored frames **1200** (÷10 decimation) | Stale field only |
| `wideband_data` | Not in paper | Present in **41/124** versions (AWGN, real, real_awgn); noise-free `signal_data` elsewhere | Generator fix (2026 regen) |
| Empty entries | Paper allows zero-signal entries | **0** empty entries in this export | Minor |

## Detailed findings

### Version inventory (124 total)

| Class | Versions | Count |
|---|---|---|
| Ideal | v1 | 1 |
| Rician (7 speeds × K=1..10) | v2–v71 | 70 |
| Rayleigh (7 speeds) | v72–v78 | 7 |
| AWGN (SNR −8:2:30 dB) | v79–v98 | 20 |
| Clock offset (1,3,5,7,9 ppm) | v99–v103 | 5 |
| Real (random fading + offset + SNR) | v104 | 1 |
| Real AWGN (fading + offset, fixed SNR) | v105–v124 | 20 |

Verified by scanning `anno/000001.json` channel tags and counting directories.

### Signal-count distribution mismatch (high impact on AR@k)

Paper VI-A selects AR caps from the **published** CRML23 histogram (4/5/6 dominate).
Our regenerated export shifts mass toward **3–4 signals per frame**:

```
Count   Paper (Fig. 2c)    Ours (124k frames)
  2          —               1.1%
  3       uncommon          49.6%
  4       33.3%             47.4%
  5       59.0%              1.9%
  6        6.0%              0%
  0      allowed            0%
```

Likely cause: same recursive band-filling code but different random seed / generator
revision (`rng(0)` in current `twc/generate.m`) or protect-gap behaviour. **AR@4/5/6
reported in the paper is not directly comparable** unless we reconfigure
`SignalDetectionMetric` max-detection caps to AR@3/4/5.

### SNR grid mismatch (moderate impact on SNR curves)

Table I lists **[12:30:2] dB**. The public `twc` generator uses **`snrs = -8:2:30`**
(20 levels). Low-SNR versions (v79–v84, v105–v110) exist on disk but are outside
the paper table. SNR-wise mAP curves should be annotated when comparing to Fig. 7/10/12.

Non-AWGN versions (ideal, Rician, Rayleigh, clock) carry per-signal `infdB` labels;
only AWGN/real/real_awgn versions contribute finite SNR bins to SNR curves.

### Bandwidth / anchor alignment (moderate impact on AP_small/medium/large)

Paper AP thresholds: **110 / 130 / 150** bins. Empirical GT bandwidths cluster at
**~96 / ~120 / ~146** bins (38.0% / 31.3% / 30.7% by paper thresholds). Using
paper anchor widths 110/130/150 vs empirical 96/120/146 is an open retune axis
(see `retune_campaign.md`).

### Split ratio (low–moderate impact)

Paper does not document train/val/test fractions. We use **50/10/40** with seed 0
(`csrr/datasets/csrd.py`). Test set is **40%** (49.6k frames) — larger than typical
70/15/15 or 60/20/20 splits. Published numbers may use a different partition;
absolute mAP levels are not strictly comparable without the authors' split files.

### Regenerated vs paper-era dataset

The 2026 export fixes wideband noise stacking (`wideband_data` policy, commit
`78b086b` in local `ChangShuoRadioData-twc`). The old 2024 export at
`the 2024-05 CSRD export` has the same 124×1000 scale and
similar bandwidth clusters but different noise/SNR semantics on some version classes.
**Prefer `ChangShuoTwc2026` for all new JDM work.**

## Impact on metric comparability

| Mismatch | Effect |
|---|---|
| Signal-count distribution | AR@4/5/6 numbers differ; joint/det recall caps misaligned |
| SNR −8..10 dB extra versions | Low-SNR tail on our curves has no paper counterpart |
| 50/10/40 split | Test size & composition differ from unknown paper split |
| Bandwidth cluster shift | AP_small/medium/large and anchor choice affect localization mAP |
| 5-epoch vs 30-epoch detector | Current best detector is under-trained vs paper protocol |
| Score fusion + proposal AMC | Our joint pipeline adds steps not in the paper baseline |

**Recommendation:** treat paper Fig. 8/10/13 as **qualitative** targets unless/until
we extract numeric values from the PDF and align AR caps + SNR subsets. Continue
retune on this export; document deviations in every results table.

## Audit commands (reproducible)

```bash
# Full scale + distribution scan
python \
  tools/jdm/retune_sweep.py --audit-dataset

# SNR verification (AWGN versions)
python tools/misc/verify_csrd_snr.py \
  --data-root data/ChangShuoTwc2026
```

## References

- `docs/csrd_jointdet/paper_and_history_notes.md` — paper protocol summary
- `docs/csrd_jointdet/dataset_generation.md` — regen protocol & noise fix
- `ChangShuoRadioData (twc/)/twc/generate.m` — ground-truth generator logic
- `csrr/datasets/csrd.py` — split & loading code
