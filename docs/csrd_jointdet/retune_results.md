# JDM Retune Results

Append-only log from ``tools/jdm/retune_sweep.py``.


## 2026-07-06 08:07:17

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-06 08:07:17 | det_wave2_es_pat3_lr5e4_bw20_tm10 | detector | `det_wave2_es_pat3_lr5e4_bw20_tm10` | done | `False` | map=0.6775, ap75=0.7804 | P0 retry: lr 5e-4, cosine T_max=10, ES patience 3, bw×20, max 15 ep |
| 2026-07-06 09:21:33 | det_wave2_es_pat5_lr5e4_bw20_tm10 | detector | `det_wave2_es_pat5_lr5e4_bw20_tm10` | done | `False` | map=0.6527, ap75=0.7773 | P0 retry: lr 5e-4, cosine T_max=10, ES patience 5, bw×20, max 15 ep |
| 2026-07-06 10:44:40 | det_wave2_es_pat5_lr5e4_bw20_tm15 | detector | `det_wave2_es_pat5_lr5e4_bw20_tm15` | done | `False` | map=0.6603, ap75=0.7049 | P0 retry: lr 5e-4, cosine T_max=15, ES patience 5, bw×20, max 15 ep |
| 2026-07-06 12:10:32 | det_wave2_es_pat5_lr5e4_bw2_tm15 | detector | `det_wave2_es_pat5_lr5e4_bw2_tm15` | done | `False` | map=0.6924, ap75=0.7998 | P0 retry: lr 5e-4, cosine T_max=15, ES patience 5, paper bw×2, max 15 ep |

## 2026-07-08 01:42:56

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-08 01:42:56 | det_wave3_ft_10ep_lr1e4_es3 | detector | `det_wave3_ft_10ep_lr1e4_es3` | done | `False` | map=0.7612, ap75=0.8868 | Track A: load 5-ep best, 10 ep, lr 1e-4, cosine T_max=10, ES pat 3, bw×20, anchors 96/120/146 |
| 2026-07-08 03:01:01 | det_wave3_ft_5ep_lr1e4_es3 | detector | `det_wave3_ft_5ep_lr1e4_es3` | done | `False` | map=0.7593, ap75=0.8815 | Track A: load 5-ep best, 5 ep, lr 1e-4, cosine T_max=5, ES pat 3, bw×20, anchors 96/120/146 |
| 2026-07-08 03:44:12 | det_wave3_ft_5ep_lr5e4_es3 | detector | `det_wave3_ft_5ep_lr5e4_es3` | done | `False` | map=0.7615, ap75=0.8961 | Track A: load 5-ep best, 5 ep, lr 5e-4, cosine T_max=5, ES pat 3, bw×20, anchors 96/120/146 |
| 2026-07-08 04:47:56 | det_wave3_ft_8ep_lr1e4_es3 | detector | `det_wave3_ft_8ep_lr1e4_es3` | done | `False` | map=0.7593, ap75=0.8803 | Track A: load 5-ep best, 8 ep, lr 1e-4, cosine T_max=8, ES pat 3, bw×20, anchors 96/120/146 |

## 2026-07-11 21:47:54

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-11 21:47:54 | det_wave3b_5ep_lr1e3 | detector | `det_wave3b_5ep_lr1e3` | done | `False` | map=0.8113, ap75=0.8921 | Track B: fresh 5 ep, lr 1e-3 (base winning recipe), ES off |
| 2026-07-11 22:31:44 | det_wave3b_5ep_lr5e4 | detector | `det_wave3b_5ep_lr5e4` | done | `False` | map=0.5977, ap75=0.5989 | Track B: fresh 5 ep, lr 5e-4, cosine T_max=5, ES off, bw×20, anchors 96/120/146 |
| 2026-07-11 23:16:30 | det_wave3b_8ep_lr1e3 | detector | `det_wave3b_8ep_lr1e3` | done | `False` | map=0.6934, ap75=0.8152 | Track B: fresh 8 ep, lr 1e-3, cosine T_max=8, ES off |
| 2026-07-12 00:16:04 | det_wave3b_5ep_lr1e3_es3 | detector | `det_wave3b_5ep_lr1e3_es3` | done | `False` | map=0.6939, ap75=0.8149 | Track B: fresh 5 ep, lr 1e-3, ES patience 3 |
| 2026-07-12 01:03:52 | det_wave3b_5ep_lr2e3 | detector | `det_wave3b_5ep_lr2e3` | done | `False` | map=0.7777, ap75=0.8504 | Track B: fresh 5 ep, lr 2e-3, ES off |

## 2026-07-14 01:08:25

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-14 01:08:25 | amc_wave3b_detprops_30ep | amc | `amc_wave3b_detprops_30ep` | done | `False` | — | P1: Track B det proposals + continue from 20-ep AMC best, 30 ep cosine |

## 2026-07-15 02:38:31

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-15 02:38:31 | amc_wave3b_detprops_30ep | amc | `amc_wave3b_detprops_30ep` | done | `False` | top1_pct=83.02633078764391 | P1: Track B det proposals + continue from 20-ep AMC best, 30 ep cosine |

## 2026-07-16 04:41:18

| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |
|---||---||---||---||---||---||---|
| 2026-07-16 04:41:18 | det_wave3b_5ep_lr1e3 | detector | `det_wave3b_5ep_lr1e3` | done | `False` | map=0.8113, ap75=0.8921 | Track B: fresh 5 ep, lr 1e-3 (base winning recipe), ES off |
| 2026-07-16 05:02:52 | det_wave3b_5ep_lr5e4 | detector | `det_wave3b_5ep_lr5e4` | done | `False` | map=0.5977, ap75=0.5989 | Track B: fresh 5 ep, lr 5e-4, cosine T_max=5, ES off, bw×20, anchors 96/120/146 |
| 2026-07-16 05:26:07 | det_wave3b_8ep_lr1e3 | detector | `det_wave3b_8ep_lr1e3` | done | `False` | map=0.6934, ap75=0.8152 | Track B: fresh 8 ep, lr 1e-3, cosine T_max=8, ES off |
| 2026-07-16 05:49:19 | det_wave3b_5ep_lr1e3_es3 | detector | `det_wave3b_5ep_lr1e3_es3` | done | `False` | map=0.6939, ap75=0.8149 | Track B: fresh 5 ep, lr 1e-3, ES patience 3 |
| 2026-07-16 06:11:55 | det_wave3b_5ep_lr2e3 | detector | `det_wave3b_5ep_lr2e3` | done | `False` | map=0.7777, ap75=0.8504 | Track B: fresh 5 ep, lr 2e-3, ES off |
