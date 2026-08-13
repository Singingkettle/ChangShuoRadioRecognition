# JDM Retune Campaign

Date: 2026-07-06

Goal: close remaining gaps vs Xing et al. TWC 2024 (JDM) after the dataset scale
audit (`dataset_scale_audit.md`) and detector localization fixes documented in
`optimization_notes.md`.

## Architecture freeze policy

**Core principle (non-negotiable):** retunes must **not** change JDM network
architecture. Detector backbone (FFT stem, stride, channel widths), AMC backbone,
and joint head topology must match the paper / official `configs/jdm/` baseline.
Hyperparameters, initialization, training budget, loss weights, anchor widths,
and inference score fusion **are** allowed.

See also [`optimization_notes.md`](./optimization_notes.md) — score fusion
(`fuse_scores`), anchor widths (empirical 96/120/146 vs paper 110/130/150), and
bandwidth loss weight (20 vs 2) are **hyperparameters**, not architecture.
Same-padding grid vs valid-padding is a documented implementation deviation;
retune must **not** change backbone/head layer counts or conv/LSTM topology.

| Category | Allowed in retune | Forbidden |
|----------|-------------------|-----------|
| **Init** | `model.*.init_cfg` when present | Adding/removing conv or FC layers |
| **Optim / schedule** | lr, wd, `param_scheduler`, `train_cfg.max_epochs` | Changing detector stride or backbone depth |
| **Loss weights** | `loss_bw`, `loss_center`, AMC CE weight | Replacing head loss types structurally |
| **Anchors** | `model.head.anchor_widths` (e.g. 96/120/146 vs 110/130/150) | Changing number of anchor scales |
| **Training budget** | 30-epoch full detector retrain (paper protocol) | Shrinking AMC backbone width |
| **Inference** | `fuse_scores=True` (det × cls confidence) | NMS/backbone structural edits |
| **AMC domain** | Proposal-crop pipeline, epoch extension | New backbone for AMC |

`tools/jdm/retune_sweep.py` documents permitted `--cfg-options` keys.

### JDM-specific clarifications

- **Anchor widths 110/130/150 vs 96/120/146** — anchor *hyperparameters* tied to
  AP-bin calibration; OK to sweep (Wave 1 P0-A vs P0-B).
- **Bandwidth loss weight 20 vs 2** — loss *weight*; OK.
- **30-epoch detector retrain** — training *strategy* matching paper budget; OK.
- **Proposal-crop AMC 30 ep** — fine-tune schedule + data pipeline; backbone unchanged.

---

## Intervention catalog

### Allowed

| Category | Options | Config pattern |
|----------|---------|----------------|
| **Init** | Xavier / default PyTorch (when `init_cfg` set) | `model.backbone.init_cfg` |
| **LR / wd** | Adam lr `{5e-5, 1e-4, 2e-4}`, wd `{5e-5, 1e-4}` | `optim_wrapper.optimizer` |
| **Schedule** | CosineAnnealingLR `T_max`, `max_epochs` | `param_scheduler`, `train_cfg` |
| **Early stopping** | patience, `min_delta` (detector uses val mAP) | `custom_hooks` |
| **Batch size** | train/val dataloader batch | `train_dataloader.batch_size` |
| **Grad clip** | `max_norm` | `optim_wrapper.clip_grad` |
| **Loss weights** | `loss_bw` ×20, center BCE weight | `model.head.loss_*` |
| **Anchors** | empirical `(96,120,146)` or paper `(110,130,150)` | `model.head.anchor_widths` |
| **Training budget** | 30-ep detector; 30-ep proposal-crop AMC | dedicated retune configs |
| **Inference** | `fuse_scores=True` on joint test | `model.fuse_scores` or experiment cfg |
| **Warmup** | LinearLR ramp (if needed) | `param_scheduler` |

### Forbidden

| Change | Why blocked |
|--------|-------------|
| Detector/AMC backbone depth or channel widths | Paper architecture |
| Head conv/FC layer count, kernel sizes | Structural |
| Anchor *count* (must stay 3 scales) | Topology |
| Stride / grid cell count change | Documented same-padding deviation is fixed |
| Replacing `JDMFramework` submodule types | Architecture swap |

---

## Current best (2026-07-05)

| Component | Checkpoint | Test metric |
|---|---|---|
| Detector (5 ep) | `exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth` | mAP **0.7677**, AP75 **0.9182** |
| AMC GT-box | `jdm-amc_iq-csrd/best_accuracy_top1_epoch_60.pth` | val top1 ~**87%** |
| AMC proposal-crop (20 ep) | `exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth` | val top1 **78.09%** |
| Joint + fuse_scores | merged detprops 20 ep | class-aware mAP **0.5868** |

**Blockers to paper parity:** detector trained 5/30 epochs; anchor/AP-bin alignment;
AMC domain gap on detector crops; AR@k caps misaligned with our signal-count
distribution; unknown paper train/test split.

## Prioritized experiment queue

| P | ID | Experiment | Config | GPU est. | Expected impact |
|---|---|---|---|---|---|
| **P0** | `det_30ep_096146_bw20` | Full **30-epoch** detector, empirical anchors, bw loss ×20 | `retune/det_30ep_anchor096146_bw20.py` | ~2–3 h | +mAP/AP75 vs 5-ep; paper training budget |
| **P0** | `det_30ep_110130150_bw20` | Full 30-ep detector, **paper AP-bin anchors** 110/130/150 | `retune/det_30ep_anchor110130150_bw20.py` | ~2–3 h | Tests paper anchor hypothesis |
| **P1** | `amc_detprops_30ep` | Proposal-crop AMC **30 ep** (lr 1e-4, from 20-ep best) | `retune/amc_detprops_30ep.py` | ~45 min | +joint mAP via cls |
| **P1** | `amc_detprops_lr_sweep` | AMC lr `{5e-5, 1e-4, 2e-4}` × wd `{5e-5, 1e-4}` | manifest variants | ~3 h total | Close GT vs proposal val gap |
| **P2** | `joint_remerge_30ep` | Merge 30-ep det + best AMC → joint test + fuse | manual merge | ~40 min test | End-to-end headline number |
| **P2** | `per_class_crop_diag` | Per-modulation acc on detector crops at test | `tools/jdm/per_class_crop_diag.py` (TBD) | ~15 min | Target confused classes |
| **P3** | `ar_caps_345` | Re-eval with AR@3/4/5 caps matching our histogram | metric cfg-options | ~40 min | Apples-to-apples AR vs paper |
| **P3** | `paper_fig_extract` | Extract numeric targets from Fig. 8/10/13 PDF | manual + `tools/jdm/extract_paper_figs.py` (TBD) | — | Quantitative paper target table |
| **P3** | `snr_subset_12_30` | Replot SNR curves using SNR ≥ 12 dB only | test cfg filter | ~30 min | Match Table I SNR range |

## Goal mode usage

See [`goal_mode.md`](./goal_mode.md). Quick start:

```bash
# Status only (no GPU)
python tools/jdm/retune_sweep.py --goal-status

# Run wave until P0 goals met
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave1_manifest.json \
  --goal-mode --gpu 0,1 --max-parallel 2
```

Status artifact: `work_dirs/jdm/retune/GOAL_STATUS.json`

## Wave 1 manifest (launch now)

File: `configs/jdm/experiments/retune/wave1_manifest.json`

Runs via:

```bash
# Dry-run
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave1_manifest.json --dry-run

# Execute wave 1 on free GPUs
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave1_manifest.json \
  --gpu 0,1 --max-parallel 2
```

## Paper figure extraction plan (Fig. 8 / 10 / 13)

1. Download PDF from DOI / arXiv:2405.00736.
2. Fig. 8 (detection mAP/AR vs SNR): extract AWGN + simulated curves at SNR
   12, 16, 20, 24, 28, 30 dB for mAP, AP50, AP75, AR@4/5/6.
3. Fig. 10 (AMC per-modulation vs SNR): extract AWGN BPSK/QPSK/… curves at
   same SNR grid.
4. Fig. 13 (joint JDM): extract framework mAP + per-modulation joint accuracy.
5. Store in `docs/csrd_jointdet/paper_figure_targets.md` with digitized values
   and note our comparable subset (SNR ≥ 12, AR@3/4/5).

Tools: manual digitization (WebPlotDigitizer) or `tools/jdm/extract_paper_figs.py`
if figure assets are available locally.

## Success criteria (campaign)

| Metric | Current best | Target (paper-exact) | Paper source |
|---|---|---|---|
| Detector mAP (test) | **0.8113** mixed / **0.7899** AWGN v89–v98 | ≥ **0.91** (ideal) / floor 0.76 simul | Fig. 8(a) |
| Detector AP75 | **0.9182** mixed (5-ep) / **0.9501** AWGN | ≥ **0.96** (ideal) / floor 0.81 simul | Fig. 8(a) |
| Joint mAP (fuse) | **0.6686** mixed / **0.7621** AWGN | ≥ **0.85** (ideal) / floor 0.67 simul | Fig. 13(a) |
| AMC proposal val | **83.03%** | ≥ **90%** (high-SNR proxy) | Fig. 10 |

Digitized tables: [`paper_figure_targets.md`](./paper_figure_targets.md). Goals file:
`configs/jdm/retune/goals.json` (`campaign_mode: paper_exact`).

**Simulate floors are already met** on mixed `ChangShuoTwc2026`. Campaign continues
toward Fig. 8/13 **ideal** bars + Fig. 10 90% proxy. Honest ceiling: ideal-bar
and Fig. 12 per-mod precision need AWGN/`v1` protocol; AR@k remains mismatched
(see `dataset_scale_audit.md`).

## Status log

| Date | Action | Status |
|---|---|---|
| 2026-07-06 | Dataset scale audit completed | Done — see `dataset_scale_audit.md` |
| 2026-07-06 | Retune campaign plan created | Done |
| 2026-07-06 | Wave 1: `det_30ep_096146_bw20` + `det_30ep_110130150_bw20` | **Done** — see Wave 1 results |
| 2026-07-06 | Wave 1 post-train test (`test_post30`) | **Done** — both P0 detectors evaluated |
| 2026-07-08 | Wave 3 Track A (ft from 5-ep best) | **Done** — best mAP 0.7615 (below baseline) |
| 2026-07-11–12 | Wave 3B Track B fresh grid | **Done** — mAP 0.8113 PASS; AP75 0.8921 regress |
| 2026-07-14 | P1 AMC + joint fuse wave3b | **Done** — AMC prop 83.03%; joint fuse **0.6686** |
| 2026-07-14 | Digitize Fig. 8/10/13 → `paper_figure_targets.md` | **Done** (±0.03–0.04) |
| 2026-07-14 | Activate paper_exact goals + P2 joint | **Done** — 0/4 goals met vs ideal |
| 2026-07-14 | AP75 FT wave3b aborted (ep1 AP75 0.8707 → ep2 0.8945; still <0.9182) | **Stopped** — prefer 5-ep path |
| 2026-07-14 | AWGN v89–v98 det (`5ep` ckpt) + joint (`wave3b_amc`) | **Done** — det mAP **0.7899** / AP75 **0.9501**; joint mAP **0.7621** / AP75 0.8072 |
| 2026-07-14 | AP75 FT from 5-ep baseline (`det_paper_exact_ap75_ft_from_5ep_baseline`) | **Done (failed)** — best AP75 0.900 @ep1 then ES; ep6 AP75 0.886 < baseline 0.9182 |
| 2026-07-14→15 | Overnight stall: `paper_exact` waiter PID 2919652 self-matched `pgrep -f det_paper_exact_ap75_ft_…` (~15.5h); `jdm_amc_launched` blocked GPU1 | **Recovered 2026-07-15 ~08:50** — killed waiter; fixed `train_running()` / clear stale AMC flag |
| 2026-07-15 | Stall-hardening v2 (structural) — see **Stall classes + fixes** below | **Done** — waiter max-wait; `jdm_gpu1_live` ignores bash waiters; Tier-A GPU1 dispatch while GPU0 holds Tier-B; keepalive restart required after script edits |
| 2026-07-15 | Merge **5-ep baseline det** + `amc_wave3b` best → AWGN joint (`eval_awgn_snr12_30_joint_5ep_amc`) | **Done** — mAP **0.6887** / AP75 0.8262 (**worse** than wave3b-det AWGN joint **0.7621**); log `joint_awgn_5ep_amc.log` |
| 2026-07-15 | Skip further failed AP75 FT recipes; mixed-test joint 0.6686 vs AWGN joint 0.762 (wave3b) / 0.689 (5ep merge) | **Policy** — prefer wave3b det for AWGN joint; paper figures often AWGN/simul separated |

### Stall classes + fixes (2026-07-15)

| Stall class | Why it hung | Fix / guarantee |
|-------------|-------------|-----------------|
| A. Waiter self-`pgrep` | `pgrep -f det_paper_exact…` matched the waiter's own argv → loop forever after FT died | `tools/jdm/launch_paper_exact_keepalive.sh`: require `tools/train.py`, PID file, max-wait 2h + exit if train never seen |
| B. Stale `jdm_amc_launched` | Flag in `work_dirs/amr_benchmark_retune/scheduler_state.json` left `true` after AMC/JDM exited → GPU1 backfill skipped forever | `clear_stale_jdm_amc_flag` each tick; `jdm_gpu1_live` only counts real `train.py` / `test_det.py` / AMC python — **not** bash waiters |
| C. Tier-B deadlock | Primary `siege_queue` had Tier-A (HCGDNN) `pending` while ResNetAMR siege set `siege_orch=true`; backfill path required `pending_n==0` → GPU1 idle + Tier-A starved | Keepalive (+ watchdog) dispatch primary pending onto free GPU1 even when GPU0 has another siege |
| D. Stale keepalive process | Script on disk fixed but PID started days earlier still ran old bash functions | After editing keepalive/watchdog: **restart both daemons** (watchdog already restarts dead keepalive) |
| E. Queue zombies | Entries marked `exhausted`/`running` with no log/PID | `reset_false_exhausted_queue` + stale-`running` → `pending` |
| F. AMC relaunch thrash | Clearing `jdm_amc_launched` after eval made keepalive re-launch finished P1 AMC and re-set the flag | Skip AMC if `amc_wave3b_detprops_30ep/best_accuracy_*.pth` exists; set `jdm_amc_complete` |

## Wave 1 results (2026-07-06)

Both P0 detector runs completed 30 epochs (~15:07–15:11). Post-train test via
`tools/test_det.py` on `configs/jdm/jdm-det_fft-csrd.py`, work dir `test_post30/`.

| ID | Best ckpt (val mAP) | Val @ ep30 | Test mAP | Test AP50 | Test AP75 | Test AR | vs 5-ep baseline |
|---|---|---|---|---|---|---|---|
| `det_30ep_096146_bw20` | `best_detection_mAP_epoch_2.pth` (0.7183 @ ep2) | mAP 0.7010, AP75 0.8183 | **0.7227** | 0.9886 | **0.8969** | 0.8104 | mAP −0.0450, AP75 −0.0213 |
| `det_30ep_110130150_bw20` | `best_detection_mAP_epoch_27.pth` (0.7356 @ ep27) | mAP 0.7291, AP75 0.7457 | 0.7137 | 0.9807 | 0.7599 | 0.7940 | mAP −0.0540, AP75 −0.1583 |

**Takeaways**

- Empirical anchors (96/120/146) win on **test** mAP and AP75 vs paper anchors
  (110/130/150); paper-anchor run peaks val mAP at ep27 but test AP75 collapses.
- Neither run beats the 5-epoch baseline (mAP 0.7677, AP75 0.9182); interim
  target ≥0.78 mAP **not met** (`--goal-status` still FAIL on `detector_map`).
- Early-stop val best for 096146 is ep2 — full 30-ep budget did not improve val
  mAP beyond mid-training; consider P1 AMC / joint remerge with 096146 ckpt.

```bash
# Goal status after Wave 1 (2026-07-06)
python tools/jdm/retune_sweep.py --goal-status
# → detector_map FAIL (best=0.7677 baseline); amc_proposal_top1_pct FAIL
```

## Wave 2 plan (P0 retry)

**Conclusion (Wave 1):** Full 30-epoch detector retrain **without** early stopping
**regressed** vs the 5-epoch baseline. Best test mAP: 0.7227 (`det_30ep_096146_bw20`)
and 0.7137 (`det_30ep_110130150_bw20`) vs baseline **0.7677**. Val-best checkpoints
stuck at ep2 (096146) and ep27 (110130150) — clear overfitting / schedule mismatch.
**Interim best remains** `exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth`
(mAP 0.7677, AP75 0.9182).

**Wave 2 interventions** (architecture freeze unchanged):

| Knob | Wave 1 (failed) | Wave 2 |
|------|-----------------|--------|
| Max epochs | 30, no ES | **15** + `EarlyStoppingHook` on `detection/mAP` |
| ES patience | — | **3–5** |
| LR | 1e-3 (paper default) | **5e-4** |
| Scheduler | Cosine `T_max=30` | Cosine **`T_max=10` or `15`** |
| bw loss weight | ×20 only | Sweep **×20 vs ×2** (paper) |
| Anchors | 096146 + 110130150 | **096146 only** (empirical winner) |

**Goal:** beat 0.7677 test mAP; reach interim target **≥ 0.78**.

Manifest: `configs/jdm/experiments/retune/wave2_manifest.json` (4 variants).

```bash
# Dry-run wave 2 (no GPU)
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave2_manifest.json --dry-run

# Execute after AMR wave1 releases GPUs (or single-GPU interleave)
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave2_manifest.json \
  --goal-mode --gpu 0 --max-parallel 1
```

**Queue status (2026-07-06 ~16:07):** AMR cgdnet `es_patience30` on **GPU 0** (train child PID **241859**, sweep parent **241810**). JDM wave 2 on **GPU 1** (`retune_sweep.py` PID **246176**, first train child **246193**), log: `work_dirs/jdm/retune/wave2.log`.

| Date | Action | Status |
|---|---|---|
| 2026-07-06 | Wave 2 plan + manifest prepared | Done |
| 2026-07-06 | JDM wave2 `retune_sweep.py --gpu 1` launched | **Done** — sweep finished ~21:36 |
| 2026-07-06 | Wave 2 outcome documented | **FAILED** — see Wave 2 outcome + Wave 3 plan |

## Wave 2 outcome + Wave 3 plan (2026-07-06 ~22:07)

### Wave 2 conclusion

All four wave-2 detector variants finished (GPU 1, ~16:07–21:36). Post-train
test mAP on held-out test set:

| Variant | Test mAP | Test AP75 | vs 5-ep baseline |
|---|---|---|---|
| `det_wave2_es_pat3_lr5e4_bw20_tm10` | 0.6775 | 0.7804 | −0.0902 |
| `det_wave2_es_pat5_lr5e4_bw20_tm10` | 0.6527 | 0.7773 | −0.1150 |
| `det_wave2_es_pat5_lr5e4_bw20_tm15` | 0.6603 | 0.7049 | −0.1074 |
| `det_wave2_es_pat5_lr5e4_bw2_tm15` | **0.6924** | 0.7998 | −0.0753 |

**Verdict:** Early stopping + cosine schedule + lr 5e-4 **all regressed** vs the
5-epoch baseline. Best wave-2 test mAP 0.6924 (`bw×2`, T_max=15) is still
**0.0753 below** production best **0.7677**.

**Production best unchanged:**

`exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth` — mAP **0.7677**,
AP75 **0.9182**. Do **not** promote any wave-1/2 checkpoint.

`--goal-status` remains FAIL on `detector_map` (target ≥ 0.78).

### Wave 3 direction (architecture freeze)

Two complementary tracks; pick one or run both serially after AMR wave 1
releases GPUs. **Do not launch until AMR retune finishes or uses only one GPU.**

**Track A — fine-tune from 5-ep best (preferred)**

- `load_from` = `exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth`
- Short budget: **5–10 epochs** max
- lr **1e-4** (10× lower than winning recipe)
- Minimal schedule change: flat or gentle cosine `T_max=5`; optional ES patience 3
- Anchors / bw loss frozen at winning values: **96/120/146**, **bw×20**
- Goal: nudge mAP above 0.7677 without full retrain drift

**Track B — hyperparam grid around 5-ep winning recipe**

Small perturbations only; anchor **96/120/146**, **bw×20**, **lr 1e-3**, **5 ep**:

| Knob | Base | Sweep |
|---|---|---|
| lr | 1e-3 | `{5e-4, 1e-3, 2e-3}` |
| max_epochs | 5 | `{5, 8, 10}` |
| bw loss weight | ×20 | `{×20}` (fixed) |
| ES patience | off | `{off, 3}` |
| init | default | default only |

Manifest: `configs/jdm/experiments/retune/wave3_manifest.json` (Track A, 4 variants).

```bash
# Dry-run
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave3_manifest.json --dry-run

# Execute on GPU 1 only (AMR retains GPU 0)
nohup python \
  tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave3_manifest.json \
  --goal-mode --gpu 1 --max-parallel 1 \
  > work_dirs/jdm/retune/wave3.log 2>&1 &
```

**Queue status (2026-07-08 ~09:43):** AMR retune on **GPU 0** (unchanged). JDM wave 3 Track A on **GPU 1** (`retune_sweep.py` PID **910692**, log: `work_dirs/jdm/retune/wave3.log`).

| Date | Action | Status |
|---|---|---|
| 2026-07-08 | Wave 3 Track A manifest + configs prepared | Done |
| 2026-07-08 | JDM wave3 `retune_sweep.py --gpu 1` launched | **Done** — see Wave 3 outcome |

## Wave 3 outcome (2026-07-08)

Track A fine-tune from 5-ep best (`load_from` production checkpoint): all **4** variants
finished on GPU 1 (~09:43–13:57). None beat the 5-epoch baseline.

| Variant | Test mAP | Test AP75 | vs baseline |
|---|---|---|---|
| `det_wave3_ft_5ep_lr5e4_es3` | **0.7615** | 0.8961 | −0.0062 |
| `det_wave3_ft_10ep_lr1e4_es3` | 0.7612 | 0.8868 | −0.0065 |
| `det_wave3_ft_5ep_lr1e4_es3` | 0.7593 | 0.8815 | −0.0084 |
| `det_wave3_ft_8ep_lr1e4_es3` | 0.7593 | 0.8803 | −0.0084 |

**Verdict:** Fine-tuning from the 5-ep best at lower LR (1e-4 / 5e-4) **did not**
surpass baseline. Best wave-3 test mAP **0.7615** (`ft_5ep_lr5e4_es3`) remains
**0.0062 below** production best **0.7677**.

**Production best unchanged:**

`exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth` — mAP **0.7677**,
AP75 **0.9182**.

**Next direction:** Track B small hyperparam grid around the 5-ep winning recipe
(lr / epoch budget sweep), **or** accept 5-ep as ceiling pending paper figure
extraction (`paper_fig_extract` / Fig. 8 targets).

## Wave 3B outcome (2026-07-11 → 2026-07-12)

Track B fresh-train grid around the 5-ep winning recipe (lr / epochs / ES).
Best run: **`det_wave3b_5ep_lr1e3`**.

| Variant | Test mAP | Test AP75 | vs baseline mAP |
|---|---|---|---|
| **`det_wave3b_5ep_lr1e3`** | **0.8113** | 0.8921 | **+0.0436** |
| `det_wave3b_5ep_lr2e3` | 0.7777 | 0.8504 | +0.0100 |
| `det_wave3b_5ep_lr1e3_es3` | 0.6939 | 0.8149 | −0.0738 |
| `det_wave3b_8ep_lr1e3` | 0.6934 | 0.8152 | −0.0743 |
| `det_wave3b_5ep_lr5e4` | 0.5977 | 0.5989 | −0.1700 |

**mAP goal (interim ≥ 0.78):** **PASS** on mAP alone (`0.8113` ≥ `0.78`).

**AP75:** `0.8921` is **below** interim target `0.93` and **below** production
baseline AP75 **`0.9182`** — localization regresses despite mAP gain (AP50
stays high; AP75/AP band trade-off).

**Decision (2026-07-14):**

- **Do not** blindly replace the production detector
  (`exp_anchor096146_bw20_5ep` — mAP 0.7677 / AP75 0.9182).
- Use `det_wave3b_5ep_lr1e3` as an **experimental** detector for
  proposal-crop AMC / joint remerge trial only.
- Revisit production promotion only after joint mAP + AP75 review.

**Production baseline unchanged:**

| Component | Checkpoint | Metric |
|---|---|---|
| Detector (5 ep) | `exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth` | mAP **0.7677**, AP75 **0.9182** |
| AMC proposal-crop (20 ep) | `exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth` | val top1 **78.09%** |
| Joint + fuse_scores | merged detprops 20 ep | class-aware mAP **0.5868** |

**Experimental detector for P1 trial:**

`work_dirs/jdm/retune/det_wave3b_5ep_lr1e3/best_detection_mAP_epoch_5.pth`
(mAP **0.8113**, AP75 **0.8921**).

### P1 AMC PASS (2026-07-14)

`amc_wave3b_detprops_30ep` finished: best ckpt ep23, **test OA 83.03%**
(≥ P1 goal 80%). Source:
`work_dirs/jdm/retune/amc_wave3b_detprops_30ep`
(`best_accuracy_top1_epoch_23.pth`, `amc_test_metrics.json` / `res/paper.pkl`).

`--goal-status` previously showed baseline 78.09 FAIL because AMC
`tools/test.py` does not emit mmengine metric JSON. Fixed in
`tools/goal_mode_helpers.parse_jdm_metrics_json` (fallback to `paper.pkl` /
`overall accuracy` log) and `tools/test.py` now writes
`amc_test_metrics.json`.

**Next:** joint remerge + `fuse_scores` test — Track B det best
(`det_wave3b_5ep_lr1e3`) + new AMC best →
`configs/jdm/jdm-joint_iq-csrd.py` (`fuse_scores=True`).
Log: `work_dirs/jdm/retune/joint_wave3b_amc.log`.

**Keepalive note:** `jdm_trackb_launched` alone no longer blocks the next JDM
stage. After Track B, `gpu_keepalive.sh` may launch P1 AMC via
`tools/jdm/launch_wave_p1_amc.sh` (`jdm_amc_launched`). While that flag is set,
GPU1 AMR backfill is skipped so the watchdog does not steal the AMC slot.

```bash
# Goal status (discovers det_wave3b_* + AMC paper.pkl / amc_test_metrics.json)
python tools/jdm/retune_sweep.py --goal-status

# P2 joint: merge Track B det + wave3b AMC → fuse_scores test (GPU1)
# Log: work_dirs/jdm/retune/joint_wave3b_amc.log
PYTHON=python
$PYTHON tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/retune/det_wave3b_5ep_lr1e3/best_detection_mAP_epoch_5.pth \
  work_dirs/jdm/retune/amc_wave3b_detprops_30ep/best_accuracy_top1_epoch_23.pth \
  work_dirs/jdm/retune/jdm_joint_wave3b_amc.pth
CUDA_VISIBLE_DEVICES=1 $PYTHON tools/test_det.py \
  configs/jdm/jdm-joint_iq-csrd.py \
  work_dirs/jdm/retune/jdm_joint_wave3b_amc.pth \
  --work-dir work_dirs/jdm/retune/joint_wave3b_amc
```
| Date | Action | Status |
|---|---|---|
| 2026-07-11–12 | Wave 3B Track B grid (5 variants) | **Done** — best mAP 0.8113 |
| 2026-07-14 | `--goal-status` fix (scan `det_wave*`) + P1 AMC launch | **Done** — AMC test OA **83.03%** PASS |
| 2026-07-14 | AMC metrics discovery fix + joint merge+fuse on GPU1 | In progress |

## Commands (wave 1 — started 2026-07-06)

```bash
# P0-A: empirical anchors, 30 epochs (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup \
  python \
  tools/train.py configs/jdm/experiments/retune/det_30ep_anchor096146_bw20.py \
  > work_dirs/jdm/retune/det_30ep_anchor096146_bw20/train.log 2>&1 &

# P0-B: paper anchors, 30 epochs (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup \
  python \
  tools/train.py configs/jdm/experiments/retune/det_30ep_anchor110130150_bw20.py \
  > work_dirs/jdm/retune/det_30ep_anchor110130150_bw20/train.log 2>&1 &
```

After completion:

```bash
# Test both checkpoints
CUDA_VISIBLE_DEVICES=0 python tools/test_det.py \
  configs/jdm/experiments/retune/det_30ep_anchor096146_bw20.py \
  work_dirs/jdm/retune/det_30ep_anchor096146_bw20/best_detection_mAP_epoch_*.pth \
  --work-dir work_dirs/jdm/retune/det_30ep_anchor096146_bw20_test

# Merge best 30-ep detector + 20-ep proposal AMC → joint + fuse test
python tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/retune/det_30ep_anchor096146_bw20/best_detection_mAP_epoch_*.pth \
  work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth \
  work_dirs/jdm/retune/jdm_joint_30ep_det_20ep_amc.pth
```
