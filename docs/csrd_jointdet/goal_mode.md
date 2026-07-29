# JDM Goal Mode

Goal mode drives the JDM retune campaign until active P0/P1 numeric targets are
met or the manifest queue is exhausted.

**Orchestrator:** `tools/jdm/retune_sweep.py`  
**Goals config:** `configs/jdm/retune/goals.json`  
**Status artifact:** `work_dirs/jdm/retune/GOAL_STATUS.json`  
**Campaign plan:** [`retune_campaign.md`](./retune_campaign.md)

---

## Architecture freeze policy

Goal mode does **not** relax the architecture freeze. Detector/AMC backbone
depth, channel widths, and head topology must match the paper /
`configs/jdm/` baseline. Allowed: init, lr/wd, schedulers, ES, batch size, grad
clip, loss weights, anchor widths, `fuse_scores`, training epoch budget.

See [`retune_campaign.md` § Architecture freeze](./retune_campaign.md#architecture-freeze-policy).

---

## Two goal semantics (aligned with AMR)

As in [`../amr_benchmark/goal_mode.md`](../amr_benchmark/goal_mode.md), do not
conflate **internal tracking thresholds** with **campaign success**:

| Context | Rule |
|---------|------|
| **Tracking / regression guard** | Do not promote a variant that regresses vs documented baseline bests |
| **Campaign success** | Meet **paper-extracted** targets when available; otherwise interim targets in `goals.json` with an explicit `paper_pending` note |

Fig. 8/10/13 are digitized in [`paper_figure_targets.md`](./paper_figure_targets.md)
(2026-07-14, ±0.03–0.04). `goals.json` uses `campaign_mode: paper_exact` against
**ideal** bars / Fig. 10 90% proxy. Document waivers in `retune_campaign.md`.

---

## Measurable targets

| Goal | Priority | Current best | Paper-exact target | Paper source |
|------|----------|--------------|--------------------|--------------|
| Detector class-agnostic mAP | **P0** | see dual-protocol | ≥ **0.91** ideal (v1) / ≥ **0.76** simulate (Real+Real_awgn) | Fig. 8(a) |
| Detector AP75 | **P0** | see dual-protocol | ≥ **0.96** ideal / ≥ **0.81** simulate | Fig. 8(a) |
| Joint class-aware mAP (`fuse_scores`) | **P2** | see dual-protocol | ≥ **0.85** ideal / ≥ **0.67** simulate | Fig. 13(a) |
| AMC GT-box top1 | monitor | ~87% | ≥ 87% | inactive |
| AMC proposal-crop top1 | **P1** | 83.03% | ≥ **90%** (high-SNR proxy) | Fig. 10 |

Wave 1 manifest (`configs/jdm/experiments/retune/wave1_manifest.json`) covered
**P0 detector** (two 30-ep anchor variants) and **P1 AMC** (30-ep proposal-crop).
Joint P2 runs after detector + AMC remerge.

### Wave 1 outcome (2026-07-06)

P0 wave 1 is **exhausted without meeting** the interim detector mAP goal (≥ 0.78). Both
30-ep runs regressed vs the 5-ep baseline (best test mAP 0.7227 / 0.7137 vs
0.7677). Interim best checkpoint unchanged:
`exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth`.

**Wave 2** (`configs/jdm/experiments/retune/wave2_manifest.json`) is queued with
early-stop + cosine + lower-lr variants inheriting the 5-ep base. Launch is
**blocked** while AMR benchmark wave1 holds both GPUs (PID 241810); run wave 2
after AMR releases GPUs or interleave on a single free GPU.

---

## Stop conditions

| Mode | When tuning stops |
|------|-------------------|
| Default | After manifest experiments complete |
| `--goal-mode` | All **active** campaign goals met (paper-exact when defined) **or** manifest exhausted |
| `--goal-status` | Print checklist only; no training |

After each experiment, metrics are parsed from the latest mmengine test JSON
under the variant `work_dir` (`detection/mAP`, `accuracy/top1`, etc.) and
compared to `configs/jdm/retune/goals.json`.

---

## Auto-continue behaviour

With `--goal-mode`, the orchestrator walks the manifest in priority order. If an
experiment fails its module goal and more variants exist, it continues. Unmet
goals after queue exhaustion are recorded in `GOAL_STATUS.json` for escalation
in [`retune_campaign.md`](./retune_campaign.md).

Link to priority queue: see **Prioritized experiment queue** in
[`retune_campaign.md`](./retune_campaign.md#prioritized-experiment-queue).

---

## CLI flags

| Flag | Purpose |
|------|---------|
| `--goal-mode` | Loop manifest until active goals met or exhausted |
| `--until-pass` | Stop at first passing variant per module (when applicable) |
| `--stop-when-all-pass` | Stop when all active goals in checklist are met |
| `--goal-status` | Print goal checklist; no GPU |
| `--goals PATH` | Override goals JSON |

---

## Example commands

```bash
# Status only (no GPU)
python tools/jdm/retune_sweep.py --goal-status

# Run wave 2 until P0 goals met (after GPUs free)
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave2_manifest.json \
  --goal-mode --gpu 0 --max-parallel 1

# Full campaign stop when checklist complete
python tools/jdm/retune_sweep.py \
  --manifest configs/jdm/experiments/retune/wave2_manifest.json \
  --goal-mode --stop-when-all-pass
```

---

## Artifacts

### `GOAL_STATUS.json`

```json
{
  "active_goals": 2,
  "goals_met": 0,
  "campaign_complete": false,
  "checklist": [...],
  "last_experiment": {...},
  "updated_at": "2026-07-06T..."
}
```

### `retune_results.md`

Appends experiment rows with `goal_met` when goal mode is enabled.

---

## Related docs

- [`retune_campaign.md`](./retune_campaign.md) — wave plan, current bests, P0 blockers
- AMR goal mode: [`../amr_benchmark/goal_mode.md`](../amr_benchmark/goal_mode.md)
