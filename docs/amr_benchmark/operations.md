# AMR Benchmark Retune — Operations

Proactive monitoring and remediation for the siege + goal-mode retune campaign.

---

## Why issues were missed

Several recurring failure modes had no background detector until `health_watchdog.sh`
was added:

| Symptom | Root cause |
|---------|------------|
| GPUs idle for hours with pending queue | `gpu_keepalive.sh` logged `no action required` when queue entries were wrongly marked `exhausted` or stale `running` (orchestrator dead). No escalation path existed. |
| Keepalive silent for 4h | Daemon was running but queue parse skipped all entries; no self-test, no `STREAK_IDLE` counter, no cross-check against `health.log`. |
| Siege launches failed on workspace disconnect | Subagent shells died; no process survived to retry. `siege_hcgdnn` stayed `running` in queue with no orchestrator PID. |
| Progress only checked when user asks | No periodic `HEALTH_STATUS.json`, no `health.log` tail, no auto-relaunch on stale siege logs. |
| GPU1 idle while GPU0 trains | Goal-mode / single-GPU siege uses GPU0 only; JDM secondary slot waits 10 min. Without watchdog, GPU1 could sit at 0% indefinitely if keepalive tick logic missed the pending entry. |
| `siege_queue.json` pending=0 → silent `no action` | Keepalive only advanced the primary queue. When it was exhausted, `siege_queue_full.json` still had pending marginals (often `manifest: null`) and wave1 remaining fails, but ticks logged `no action` while GPU0 was busy / GPU1 idle (or both idle). |
| Tier-A pending + Tier-B siege on GPU0 → GPU1 idle forever | `siege_orch=true` blocked queue dispatch; `pending_n>0` blocked GPU1 backfill (`jdm_amc_launched` in `scheduler_state.json` made it worse). Fixed 2026-07-15: dispatch primary pending onto GPU1 while GPU0 busy. |
| Paper-exact waiter hung ~15h | Bare `pgrep -f <config>` matched the waiter itself; `jdm_gpu1_live` treated bash waiter as JDM owner so stale flag never cleared. Waiters now require `tools/train.py` + max-wait. |
| Promote wiped tracking → false `campaign_complete` → idle wave1 loop | `--promote` called `_update_tracking_md([one_pass])`, which **replaced** the entire auto table (was ~72 rows / ~23 pass / ~38 fail → 1 row). `GOAL_STATUS` then saw `fail_count=0` → `campaign_complete=true`. Keepalive's `wave1_goal_exhausted` stayed false (empty `exhausted_pairs`), so every tick re-launched wave1 with `--stop-when-all-pass`, which exited immediately. Both GPUs idle while `siege_queue_full` still had ~12 pending. Fixed 2026-07-16: `_update_tracking_md` **merges** by (model,dataset); `campaign_complete` requires a near-full matrix, not just zero fails. |
| Force-blocked head-of-queue starves GPU1 forever | `best_gpu1_backfill_target` / `next_pending_manifest_entry_full` returned the first pending by gap/priority. `pair_force_blocked` (exhausted/waived **or** ≥3 identical paper-exact fails) then caused `launch_gpu1_amr_backfill` to **return without trying the next entry**. Symptom: GPU0 training, GPU1 idle for hours, `full_pending=8`, scheduler.log spam `GPU1 backfill skip cgdnet … force-blocked`. Fixed 2026-07-16 evening: selectors **skip** force-blocked pairs; watchdog surfaces `IDLE_GPU1_full_pending_*` and auto-dispatches; tick interval 2 min; GPU1 fill threshold 5 min. |
| **Phantom pending + identical-fail skip → 7.7h both-GPU idle** (`phantom_pending_force_blocked`) | After MCLDNN@Hisar finished (~08:03 2026-07-18), `siege_queue_full` still showed `full_pending=6` (denscnn/resnetamr/gru2/cnn4/cnn1dpf/mcldnn) but every candidate was skipped: checkpoint-reuse re-logged the **same** fail metrics ≥3× so `pair_force_blocked` returned yes for the whole pair — including **passed** pairs (GRU2@2018) and Tier-B tracking passes (CNN4@10B). Keepalive looped `STREAK_IDLE` / `no post-siege work available (full queue + wave1 + wave4 exhausted)` while counting pending>0. Watchdog only `skip_stale_relaunch … force_blocked`. Fixed 2026-07-18: (1) selectors use **queue-terminal only**; (2) ≥3 identical fails only refuse `--force` and never block a pair that already has a `pass`; (3) `reconcile_phantom_pending.py` closes pending/stale-running → passed/exhausted so `full_pending` matches launchability; (4) when AMR queues truly empty, keepalive falls back to **JDM ideal v1** train/eval. |
| **`amr_queue_empty_no_jdm_fallback` → GPU1 idle ~94min** (2026-07-18 ~21:37–23:11) | `full_pending=0` and waves exhausted made `need_amr_fill=0`, so the tick **never entered** the block that calls `launch_jdm_ideal_fallback`. Instead it hit `SKIP JDM P1 AMC relaunch — amc_wave3b best ckpt already exists; leave GPU1 for AMR/backfill` and `no action required`. Ideal fallback itself also wrongly required `best_detection_mAP_epoch_30.pth` (never written; train ends with `epoch_30.pth` + `best_*_epoch_N`). Fixed: (1) when `full_pending=0` + GPU1 idle, **immediate** JDM ideal fall-through (also after AMC skip, and after 5min threshold); (2) train-complete via `epoch_30.pth`; (3) re-eval when `done.flag` older than best ckpt; (4) improved ideal anchors recipe; (5) watchdog issues `amr_queue_empty_no_jdm_fallback` and restarts keepalive. |
| **Wave6 pending instantly exhausted / skipped** (same evening) | (a) `reconcile_phantom_pending` used pair-level `identical_fail_ge3` → any new pending for FastMLDNN/ResNetAMR closed immediately; (b) `pair_queue_terminal` / selector `pair_is_force_blocked` returned terminal if **any** exhausted entry existed for the pair, so `siege_fastmldnn_10a` exhausted blocked `siege_fastmldnn_10a_wave6` pending. Fixed: reconcile only closes when **this entry's** manifest variants are recorded; terminal/blocked only when **no** pending/running entry remains for the pair. |
| **`jdm_fallback_false_exhausted_both_idle` → BOTH GPUs idle ~14.6h** (2026-07-19 ~06:03–20:40) | Keepalive *did* call `launch_jdm_ideal_fallback` every 2 min but it returned “no work left” while useful work existed: (1) shared `eval_ideal_v1_det/done.flag` made any later recipe look evaluated; (2) ckpt picker preferred base `det_ideal_v1_30ep` and **never** re-eval’d improved-anchors `best@ep4` (val **0.4747**, test never ran); (3) no follow-on ideal 60ep / AWGN-with-ideal path. Watchdog issue `amr_queue_empty_no_jdm_fallback` required `amr0≥1`, so **both-idle** only logged `siege_exhausted_sweep_idle_gpus_*` and restarted nothing useful. Fixed: per-ckpt eval dirs; 60ep lr5e4 + AWGN-ideal steps; free-GPU selection (fill GPU1 when GPU0 has JDM); watchdog raises `jdm_fallback_false_exhausted_both_idle` when both idle / GPU1 empty with `full_pending=0`. |
| **`all_waves_exhausted_no_next_seed` → BOTH GPUs idle ~36–38h** (2026-07-21 ~08:17–2026-07-22 ~20:30) | Wave-8 FastML+HCG+TierB exhausted; `auto_seed_next_near_miss_wave` **only queued existing** `siege_*_wave{N}.json` and printed `all_waves_exhausted_no_next_seed` when Wave-9 manifests were absent — never generated recipes. JDM ideal/AWGN/joint already had `done.flag`, so fallback spun empty. Watchdog restarted KA every ~2min (rate-unlimited) without creating work. Fixed: (1) seed Wave-9 Tier-A FastML+HCG + Tier-B buffer with NEW recipes; (2) `auto_seed` **generates** configs+manifests for `wave{N}` when missing, then queues; (3) auto-seed runs **pre-JDM** in `launch_post_siege_work`; (4) watchdog rate-limits KA restart (30min) and calls `GPU_KEEPALIVE_AUTO_SEED_ONCE=1`. Stall class unchanged. |

| **`all_waves_exhausted_no_next_seed` → BOTH GPUs idle ~14.7h** (2026-07-20 ~04:07–18:46) | Wave-7 FastMLDNN+HCGDNN exhausted; every JDM ideal/AWGN/joint step already had `done.flag`; fallback logged `amr_queue_empty_no_jdm_fallback` / `no action required` forever. Watchdog `restart_keepalive` **without killing** prior instances → ~873 zombie `gpu_keepalive.sh` + orphan `sleep 120`; concurrent ticks made `pgrep … test_det.py` **self-match** so fallback falsely skipped (“eval already live”). Fixed: (1) seed Wave-8 Tier-A near-miss + Tier-B buffer; (2) ideal joint measure even below target (ep7/impr + wave3b AMC); (3) `auto_seed_next_near_miss_wave` before declaring empty; (4) `restart_keepalive` replaces prior KA; (5) live-eval check via `/proc` python cmdline (no pgrep self-match). Stall class: `all_waves_exhausted_no_next_seed`. |

| **`auto_seed_wave_spam_phantom` + orphan waiter → BOTH GPUs idle ~7.5h** (2026-07-23 ~13:29–20:55) | (1) Orphan bash waiter PID **1116565** looped on `pgrep -f tools/train.py` while its own argv embedded `$PY … retune_model_siege.py`, so `orchestrator_running` kept `siege_orch=true` with **no** live train; (2) FastML W9 stuck fake `running` after siege parent died (FT80 val **61.26** trained, never tested); (3) `auto_seed_next_near_miss_wave` saw empty pending → generated wave10…81 with **reused bare variant names**; (4) `reconcile_phantom_pending` immediately exhausted each as “already recorded” → spam loop overwrote FastML workdirs. Fixed 2026-07-23 ~21:00: kill waiter; `orchestrator_running` requires real python `/proc/exe` (exclude bash); auto-seed **max 1 wave ahead**, **6h rate-limit**, refuse while pending/running, **wave-suffixed distinct variants only**, never reopen junk; reconcile protects `manual_careful` / `_wN` manifests; cleaned queue wave10–81 junk; tested FT80 (**test 60.78 / 91.98**); AMC 60ep FT + careful HCG W10 FT launched. Stall class: `auto_seed_wave_spam_phantom`. |

**Auto-fill guarantee (2026-07-16):** An idle GPU with remaining queue/full-queue/wave work must be filled **without a user ask**. Keepalive tick ≤2 min; GPU1 backfill ≤5 min idle (and **immediate** when primary pending=0 + GPU0 busy). Force-blocked / exhausted / stale `running` heads are skipped so the next launchable pending starts. Watchdog logs `IDLE_GPU1_*` as an issue and launches or restarts keepalive.

**Policy:** never leave both GPUs idle for more than **15 minutes** when the siege
queue has pending work. When **primary `siege_queue.json` pending=0**, keepalive
must still fill idle GPUs from **`siege_queue_full.json`** (synthesize ES/warmup
manifests if needed) or **wave1 remaining fails** — not sit on silent `no action`.
When **primary queue still has pending** and GPU0 is busy, **fill GPU1** with that
pending (prefer Tier-A FastMLDNN/HCGDNN over Tier-B ResNetAMR burning).
**When GPU0 is busy and GPU1 is empty with `full_pending>0`, fill GPU1 within 5 minutes** — never park behind a force-blocked head.

**Flag:** `jdm_amc_launched` lives in
`work_dirs/amr_benchmark_retune/scheduler_state.json`. Keepalive clears it when no
live JDM `train.py` / `test_det.py` / AMC python remains. Restart keepalive after
editing `gpu_keepalive.sh` — bash does not hot-reload functions.

---

## Daemons

Start both after a reboot or workspace reconnect:

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition

# GPU scheduler (5 min tick, launches next siege / goal-mode)
nohup bash tools/amr_benchmark/gpu_keepalive.sh \
  >> work_dirs/amr_benchmark_retune/scheduler.log 2>&1 &

# Health watchdog (3 min tick, detects + auto-fixes stuck state)
nohup bash tools/amr_benchmark/health_watchdog.sh \
  >> work_dirs/amr_benchmark_retune/health.log 2>&1 &
```

| Daemon | Log | Status artifact |
|--------|-----|-----------------|
| `gpu_keepalive.sh` | `work_dirs/amr_benchmark_retune/scheduler.log` | `scheduler_state.json` |
| `health_watchdog.sh` | `work_dirs/amr_benchmark_retune/health.log` | `HEALTH_STATUS.json` |

---

## How to monitor

```bash
# Live health + scheduler decisions
tail -f work_dirs/amr_benchmark_retune/health.log \
        work_dirs/amr_benchmark_retune/scheduler.log

# Machine-readable snapshot (issues + auto-actions)
cat work_dirs/amr_benchmark_retune/HEALTH_STATUS.json | jq .

# GPU + train processes
nvidia-smi
pgrep -af 'retune_model_siege|gpu_keepalive|health_watchdog|train.py.*amr_benchmark'
```

### HEALTH_STATUS.json fields

- `last_check` — UTC timestamp of last watchdog tick
- `gpu_util` — `[gpu0%, gpu1%]`
- `amr_jobs` — parent `train.py` count per GPU (AMR only)
- `issues[]` — detected problems this tick
- `auto_actions_taken[]` — remediation attempted

---

## Auto-remediation matrix

| Check | Alert | Auto-action |
|-------|-------|-------------|
| GPU idle | Both GPUs ≤1% util, no AMR `train.py`, idle >15 min, pending queue | Launch next siege from queue |
| Keepalive dead | `gpu_keepalive.sh` not in process list | Restart keepalive |
| Queue stuck | Pending entries, `siege_orch=false`, GPUs idle >15 min | Force launch with `--force` |
| Siege crashed | Siege log untouched >2 h, orchestrator dead | Relaunch same manifest |
| False exhausted | `exhausted` in queue but no `siege_<model>.log` | Reset entry to `pending` |
| Stale running | `running` in queue but orchestrator dead | Reset entry to `pending` |
| Disconnect recovery | r2/r3 manifest exists, variants incomplete, no/minimal log | Log `ACTION_NEEDED` in `health.log` |

### keepalive STREAK_IDLE

If `gpu_keepalive.sh` logs `no action required` for **3 consecutive ticks** while
both GPUs are idle and the queue has pending entries, the next launch uses
`--force` and writes `ERROR` to `health.log`.

---

## Manual recovery

```bash
# Force-relaunch current queue head (hcgdnn, lstm2, etc.)
PY=~/Applications/conda/envs/ChangShuoRadioRecognition/bin/python
LOGDIR=work_dirs/amr_benchmark_retune
nohup $PY tools/amr_benchmark/retune_model_siege.py \
  --queue configs/amr_benchmark/retune/siege_queue.json \
  --gpu 0,1 --max-parallel 2 --until-pass --paper-exact --promote --force \
  >> $LOGDIR/siege_hcgdnn.log 2>&1 &

# Reset falsely exhausted entries (also done automatically on daemon startup)
# Edit configs/amr_benchmark/retune/siege_queue.json — set status to "pending"
# for any entry lacking work_dirs/amr_benchmark_retune/siege_<model>.log
```

---

## 主动巡检 (proactive inspection)

Two layers: a **background watchdog** (continuous) and a **one-shot inspector**
(on-demand or cron).

| Layer | Script | Interval | Log / artifact |
|-------|--------|----------|----------------|
| Background watchdog | `tools/amr_benchmark/health_watchdog.sh` | **3 min** | `health.log`, `HEALTH_STATUS.json` |
| GPU scheduler | `tools/amr_benchmark/gpu_keepalive.sh` | **5 min** | `scheduler.log`, `scheduler_state.json` |
| One-shot inspection | `tools/amr_benchmark/run_inspection.sh` | on-demand / cron | stdout + `health.log`; optional `inspection.log` |

### What the watchdog checks every 3 min

- GPU utilization and AMR `train.py` parent count per GPU
- `gpu_keepalive.sh` and siege orchestrator alive
- Pending vs falsely `exhausted` / stale `running` entries in `siege_queue.json`
- Both-GPU idle duration (>15 min + pending queue → auto-launch siege)
- Stale siege logs (>2 h, orchestrator dead → relaunch manifest)
- Disconnect recovery hints (r2/r3 manifest incomplete, no log)

Results are written to `HEALTH_STATUS.json` each tick; remediation lines go to
`health.log` as `AUTO_ACTION:` / `ISSUE:`.

### What `run_inspection.sh` checks (on-demand)

- GPU util + memory, AMR train processes per GPU
- Keepalive, watchdog, and siege orchestrator PIDs
- `siege_queue.json` — pending, running, stuck (running without orchestrator)
- Latest `HEALTH_STATUS.json` and last 5 lines of `health.log`
- AMR `--goal-status` summary (`wave1_manifest.json`)
- JDM `--goal-status` summary

Human-readable report goes to **stdout** and is appended to `health.log`
(`INSPECTION:` prefix). Exit code **1** when critical: both GPUs idle (≤1% util,
no AMR trains) while the queue still has pending entries, or a daemon is dead.

### Log paths

| Path | Contents |
|------|----------|
| `work_dirs/amr_benchmark_retune/health.log` | Watchdog ticks, auto-actions, inspection summaries |
| `work_dirs/amr_benchmark_retune/scheduler.log` | Keepalive launch decisions, STREAK_IDLE |
| `work_dirs/amr_benchmark_retune/HEALTH_STATUS.json` | Machine-readable last watchdog snapshot |
| `work_dirs/amr_benchmark_retune/inspection.log` | Optional cron capture of `run_inspection.sh` |

### Auto-remediation (watchdog + keepalive)

See [Auto-remediation matrix](#auto-remediation-matrix) above. In short:

- Dead keepalive → restart `gpu_keepalive.sh`
- GPUs idle >15 min + pending queue → launch next siege (then `--force` if stuck)
- Primary `siege_queue.json` pending=0 + GPU1 idle (GPU0 busy or both idle) → dispatch from `siege_queue_full.json` / wave1 remaining (synthesize manifests if `manifest` is null); do **not** stop at silent `no action`
- False `exhausted` / stale `running` → reset queue entry to `pending`
- Stale siege log + dead orchestrator → relaunch manifest
- Keepalive STREAK_IDLE (3× “no action” while idle + pending) → force launch + `ERROR` in `health.log`

### Manual / cron inspection

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition

# One-shot report (stdout + health.log)
bash tools/amr_benchmark/run_inspection.sh

# Every 10 min via crontab (optional):
# */10 * * * * cd /home/citybuster/Projects/ChangShuoRadioRecognition && \
#   bash tools/amr_benchmark/run_inspection.sh \
#   >> work_dirs/amr_benchmark_retune/inspection.log 2>&1
```

After reboot or workspace reconnect, start **both** daemons (see [Daemons](#daemons));
startup lines record PIDs in `health.log` / `scheduler.log`.

---

## Two-Machine Goal Mode (2026-07-23)

Reproduction now runs across two boxes. Each box runs its **own** daemon over a
**partitioned queue** — two daemons must never share a writable `siege_queue*.json`
(they would race on `status: running`). Outcomes merge offline via
`docs/amr_benchmark/retune_results.md` + rsync, never a live-synced queue.

| Box | GPUs | Role | Daemon | Queue |
|-----|------|------|--------|-------|
| Remote `10.161.4.55` (kemove) | 4x H100 80G | Workhorse: Tier-A AMR paper-exact + JDM v1-fair training/eval | `tools/amr_benchmark/gpu_pool_keepalive.sh` | `configs/amr_benchmark/retune/siege_queue_remote.json` |
| Local | 2 GPU | Tier-B approximate (exhausted) + JDM v1-fair eval + plots/monitoring | `tools/amr_benchmark/gpu_keepalive.sh` + `health_watchdog.sh` | `siege_queue_full.json` (Tier-B); clean partition in `siege_queue_local_tierb.json` |

**N-GPU pool daemon** (`gpu_pool_keepalive.sh`): env-configurable
`REPO_ROOT`/`PYTHON`/`GPUS`/`AMR_GPUS`/`JDM_GPUS`/`QUEUE_REMOTE`. Keeps two
long-lived worker orchestrators alive over disjoint GPU lanes and **expands one
lane onto the other's freed GPUs when a lane's work is exhausted** (zero-idle
guarantee: no GPU idles while *either* lane has work). Default split is **AMR
0,1,2 + JDM 3** (not 2+2) because the JDM ideal-fair ladder is single-GPU
sequential, so a 2-GPU JDM lane would leave one GPU idle. Writes
`work_dirs/amr_benchmark_retune/POOL_STATUS.json` heartbeat.

- AMR lane: `retune_model_siege.py --queue siege_queue_remote.json --gpu <AMR_GPUS> --max-parallel <n> --until-pass --paper-exact --promote`.
- JDM lane: `tools/jdm/ideal_fair_ladder.py --train` (trains full-data det + AMC, then evaluates both protocols).

Deploy on the remote (disconnect-safe):
```
ssh citybuster@10.161.4.55 'cd /home/citybuster/Projects/ChangShuoRadioRecognition && \
  setsid nohup bash tools/amr_benchmark/gpu_pool_keepalive.sh \
  >> work_dirs/amr_benchmark_retune/pool.log 2>&1 < /dev/null &'
```

**JDM v1-fair root-cause fix**: paper "ideal" (Fig. 8/13, infdB) == CSRD version
`v1`. The fair ideal number is the **best FULL-data-trained detector evaluated on
the clean v1 test split** (`eval_ideal_v1_det_testonly.py` / `..._joint_testonly.py`
restrict ONLY the test dataloader to `v1`, training data unchanged). Retraining on
v1-only (`det_ideal_v1_30ep.py`) underfits to mAP ~0.31 — nonsense. Measured
2026-07-23: fair ideal det **mAP 0.8027 / AP75 0.836** (vs bogus 0.31; paper 0.91).
`configs/jdm/retune/goals.json` is now **dual-protocol**: ideal scored on v1
test-only, simulate on mixed test, with separate pass flags;
`campaign_complete` requires BOTH protocol sets met (`tools/goal_mode_helpers.py`
`jdm_goal_checklist`).

**Unified status**: `run_inspection.sh` now read-only SSHes the remote
(`REMOTE_SSH`/`REMOTE_REPO`/`REMOTE_ENABLE`) and reports both boxes' GPU util +
`POOL_STATUS.json`, warning when remote GPUs idle. Requires passwordless SSH
(`ssh-copy-id citybuster@10.161.4.55`).

### Steward (unattended, cron, every 30 min)

`tools/amr_benchmark/hourly_steward.py` runs **every 30 minutes via cron on the
local box** (installed by `tools/amr_benchmark/install_steward_cron.sh`; entry
`*/30 * * * * … hourly_steward.py`). It is the unattended safety net that stops the
"GPUs busy/idle but reproduction not advancing" failure. Chosen over a Cursor
cloud Automation deliberately: the H100 is a **private IP** and the code/GPUs are
local, so only an agent on this box can SSH in and act. Each hour it:

1. **Keeps daemons alive** — restarts local `gpu_keepalive.sh`/`health_watchdog.sh`
   and remote `gpu_pool_keepalive.sh` only when actually dead (pgrep-gated).
2. **Anti-idle** — reads remote `POOL_STATUS.json`; if a GPU is idle while work
   remains it nudges/relaunches the pool daemon and flags the verdict.
3. **Progress-driven strategy** — records the four JDM metrics + AMR pending each
   hour to `steward_history.jsonl`. On a **plateau** (no JDM metric moves for
   `PLATEAU_HOURS=3` samples while busy) it escalates *within the architecture
   freeze*: empty AMR queue → capped auto-seed of the next Tier-A wave; ideal-det
   still `< 0.91` with 30ep done → launch the longer
   `configs/jdm/experiments/retune/det_full_60ep_lr1e3.py` rung on the remote JDM
   GPU. Every action is stamped/idempotent (`steward_stamps.json`,
   ≥6h between repeats) so it can never spam waves — the class of bug behind
   prior stalls.
4. **Wave-12 follow-ups** (added 2026-07-27, `tools/amr_benchmark/wave12_followup.py`,
   executed on *both* boxes each tick): (a) post-train **tests** for any finished
   `*_w12*` variant that lacks `amc_test_metrics.json` (the wave-12 orchestrators
   were partly detached, so nothing else guarantees testing); (b) **stage
   chaining** — when FastMLDNN `author_stage1_ms3200_w12` completes, it writes and
   launches the author stage-2 fine-tune from the stage-1 best-val checkpoint
   (dp=0.07, beta=0.5, constant 1.054e-4); (c) **pulls remote test metrics** back
   into the local tree and logs `*** NEW BEST ***` when a wave-12 result beats the
   historical Tier-A best (61.02 / 63.31). All follow-ups are file-gated and
   idempotent; launches pick the least-loaded physical GPU with
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
5. Writes `work_dirs/amr_benchmark_retune/STEWARD_STATUS.json` (verdict + actions
   + both boxes' busy/idle + JDM metrics) and logs to `steward.log`.

Inspect: `cat work_dirs/amr_benchmark_retune/STEWARD_STATUS.json`. Transient
train→test/checkpoint troughs are expected and do **not** trigger escalation
(plateau needs a multi-hour flat window while busy).

---

## Related docs

- [`goal_mode.md`](./goal_mode.md) — siege vs goal-mode semantics
- [`retune_campaign.md`](./retune_campaign.md) — campaign plan and queue
- [`accuracy_tracking.md`](./accuracy_tracking.md) — pass/fail matrix
