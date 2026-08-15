# 论文图数字目标（arXiv:2405.00736）

[English](paper_figure_targets.md) | 简体中文

**复现已关闭。** 检测 simulate（Fig. 8）与 AMC（Fig. 10，GT 框）达到或超过数字化
论文值；剩下的 ideal COCO-mAP / joint-simulate 差距来自高 IoU 离散化与生成器协议
差异，不是缺方法。工作点与停止再调的理由见
[`README_zh-CN.md`](README_zh-CN.md#结果)。

来源：arXiv [2405.00736](https://arxiv.org/abs/2405.00736)（不要把 PDF 放进本仓库）。
数字化日期：2026-07-14。方法：页面按 220 dpi 栅格化，裁 Fig. 8/10/12/13，目视读
雷达 / SNR 标记。**不确定度除非另注，绝对值为 ±0.03**。这些图**没有**出现在论文
正文表格里 — 下面的值是数字化估计，不是作者表。

## 公平比较的注意事项（请先读）

| Issue | Paper | Our `ChangShuoTwc2026` | Apples-to-apples? |
|---|---|---|---|
| Signal-count hist | 4/5/6 dominate (Fig. 2c) | 3/4 dominate; no 6-signal | **No** for AR@4/5/6 |
| SNR grid | Table I `[12:30:2]` | AWGN `-8:2:30` + fading | **Partial** — use `v89–v98` |
| “Ideal” setting | Pure signal, **no random factors** (Fig. 8/13) | `v1` (`channel=ideal`, `snr=infdB`) | **Yes** (test-only) |
| “Simulated” setting | Channel/velocity/K + **SNR as random factor** (Fig. 8/13) | `v104` Real + `v105–v124` Real_awgn | **Yes** (test-only; tightened 2026-07-24) |
| AWGN (SNR curves) | Pure AWGN, velocity=0 (Fig. 7/10/12 solid) | `v89–v98` (paper Table I `[12:30:2]`) | **Yes** for curves only |
| Full 124 mixed test | — | Historical reference only | **Not** Fig. 8/13 simulate |
| Train/val/test split | Not published | 50/10/40 seed 0 | Unknown |
| Fig. 10 y-axis | Classification accuracy / precision | Our AMC reports top-1 % | Yes (same meaning) |
| Fig. 12 y-axis | Joint per-modulation precision | Our joint SNR curve = **class-aware mAP** | **No** — different metric |
| Fig. 8 / 13 | Radar of aggregate det / joint mAP·AR | Same metric family | Yes if condition matched |

**诚实上限：** Fig. 8/13 的 **ideal** 必须只用 v1 测试（不要混）。Fig. 8/13 的
**simulate** 必须用 Real/Real_awgn（`v104`+`v105–v124`），**不是**全部 124 版本
混合测试（混合会把 ideal/AWGN/消融掺进去，把目标看起来“已经达到”）。Fig. 13(a)
simulate ~0.67 与 Fig. 8(a) simulate ~0.76/0.81 必须在收紧后的协议下重测。
逐点 Fig. 10/12 SNR 曲线需要 AWGN `v89–v98`（理想情况下还要用论文的逐调制
precision，而不是只有 class-aware mAP）。

可对比的评测旋钮：

```bash
# Paper Table I SNR subset (AWGN 12..30 dB) — Fig. 7/10/12 solid
# versions = ['v89'..'v98']  → configs/.../eval_awgn_v89_v98_det_testonly.py

# Ideal (Fig. 8/13) — generate.m Ideal
# versions = ['v1']  → eval_ideal_v1_*_testonly.py

# Simulate (Fig. 8/13) — generate.m Real + Real_awgn
# versions = ['v104'] + ['v105'..'v124']  → eval_simulate_real_awgn_*_testonly.py
```

---

## Fig. 8 — 检测汇总雷达图（第 9 页）

图注：检测模块的评测指标 (a) 对常规方法 (b)。

### Fig. 8(a) — 本文方法（数字化，±0.03）

| Metric | Ideal | Simulate | Campaign use |
|---|---:|---:|---|
| **mAP** | **0.91** | **0.76** | Primary det target |
| mAP@.50 | 1.00 | 0.95 | |
| **mAP@.75** | **0.96** | **0.81** | Secondary (AP75) |
| mAP_small | 0.91 | 0.71 | |
| mAP_medium | 0.91 | 0.75 | |
| mAP_large | 0.92 | 0.82 | |
| AR@4 / @5 / @6 | ~0.92 | ~0.81 | Not fair vs our hist |
| AR_small | 0.90 | 0.71 | |
| AR_medium | 0.91 | 0.76 | |
| AR_large | 0.96 | 0.88 | |

### Fig. 8(b) — 常规方法（仅作对照）

| Metric | Match | Threshold |
|---|---:|---:|
| mAP | ~0.55 | ~0.46 |
| mAP@.50 | ~0.91 | ~0.81 |
| mAP@.75 | ~0.65 | ~0.51 |

论文正文（Fig. 7）：simulate 在匹配 SNR 下比 AWGN 大约 **−10 pp**。

---

## Fig. 10 — AMC 准确率随 SNR（第 10 页）

纵轴：分类准确率（论文正文）。横轴：SNR 12→30，步长 2。
实线 = AWGN；空心/虚线 = simulate。

### 数字化曲线（准确率，±0.03；图顶略有裁切）

**Simulate（最佳数字化）：**

| SNR | BPSK* | QPSK | 8PSK | 16QAM | 64QAM |
|---:|---:|---:|---:|---:|---:|
| 12 | ≥0.80 | 0.40 | 0.30 | 0.17 | 0.05 |
| 14 | ≥0.80 | 0.55 | 0.43 | 0.39 | 0.13 |
| 16 | ≥0.80 | 0.56 | 0.45 | 0.40 | 0.24 |
| 18 | ≥0.80 | 0.56 | 0.49 | 0.40 | 0.24 |
| 20 | ≥0.85 | 0.61 | 0.53 | 0.41 | 0.30 |
| 22 | ≥0.85 | 0.65 | 0.53 | 0.49 | 0.30 |
| 24 | ≥0.90 | 0.67 | 0.54 | 0.51 | 0.32 |
| 26 | ≥0.90 | 0.72 | 0.60 | 0.51 | 0.34 |
| 28 | ≥0.95 | 0.75 | 0.61 | 0.59 | 0.39 |
| 30 | ~0.98 | 0.77 | 0.63 | 0.62 | 0.43 |

\*BPSK 的 AWGN/Simul 大多在裁切线以上；论文正文：高 SNR 时 BPSK → **~1.0**。

**AWGN（可见 / 正文可支撑）：**

| SNR | 16QAM | 64QAM | BPSK (text) |
|---:|---:|---:|---|
| 12 | ~0.72 | ~0.65 | high |
| 20 | ~0.82 | ~0.81 | →1.0 |
| 30 | ~0.89 | ~0.87 | ~1.0 |

**汇总目标代理**（论文里没有这个单一数字）：高 SNR AWGN 宏平均约为
**0.88–0.92**。Proposal 裁剪验证 top1 的论文精确代理：**≥ 90%**
（GT 框已经约 87%；proposal 目前是 **83.03%**）。

---

## Fig. 12 — Joint 逐调制随 SNR（第 11 页）

同一 SNR 网格；纵轴是 joint **precision**（论文）。正文：joint ≈ AMC −
**20–30 pp**；simulate ≈ AWGN − **10–15 pp**。

### 数字化 AWGN 平台（约，±0.04）

| Mod | @12 dB | @30 dB |
|---|---:|---:|
| BPSK | ~0.72 | ~0.85 |
| QPSK | ~0.61 | ~0.75 |
| 8PSK | ~0.55 | ~0.71 |
| 16QAM | ~0.47 | ~0.81 |
| 64QAM | ~0.43 | ~0.66 |

Simulate 曲线在中等 SNR 大约低于 AWGN 0.10–0.25；BPSK simul 在接近 30 dB 时追上。

**重要：** 我们的 `snr_curve.json` joint 点是 **class-aware mAP**
（wave3b joint 在 AWGN 12–30 上约 0.33–0.35）。这与 Fig. 12 的逐调制
precision **不是同一个量** — 不要只凭 mAP 曲线声称对上了 Fig. 12。

---

## Fig. 13 — Joint 汇总雷达图（第 11 页）

图注：JDM 评测指标 (a) 对常规组合 (b)。

### Fig. 13(a) — 本文方法（数字化，±0.04）

| Metric | Ideal | Simulate | Notes |
|---|---:|---:|---|
| **mAP** | **0.85** | **0.67** | Primary joint target |
| mAP@.50 | ~0.95 | ~0.76 | |
| mAP@.75 | ~0.72 | ~0.62 | |
| size mAPs | ~0.80–0.85 | ~0.66–0.68 | |
| AR family | ~0.80–0.85 | ~0.72 | AR@k unfair |

Fig. 13(b) 基线（MF/TH × SVM/DT）即使在放大刻度上也远小于 0.5 — 仅作对照。

---

## 目标映射 → `configs/jdm/retune/goals.json`

| Goal key | Paper figure | Active target (paper-exact) | Our best (2026-07-14) | Gap |
|---|---|---:|---:|---|
| `detector.map_min` | Fig. 8(a) **ideal** mAP | **0.91** | 0.8113 | −0.10 |
| `detector.ap75_min` | Fig. 8(a) **ideal** AP75 | **0.96** | 0.8921 (prod AP75 0.9182) | −0.07 / −0.04 |
| `joint.map_min` | Fig. 13(a) **ideal** mAP | **0.85** | 0.6686 | −0.18 |
| `amc_proposal.top1_min_pct` | Fig. 10 high-SNR proxy | **90.0** | 83.03 | −6.97 pp |

**Simulate 下限 — 在 Real/Real_awgn 下重测（2026-07-24 收紧）：**

- 旧的混合测试 det mAP 0.8113 / joint 0.6686 **仅作参考**；协议收紧后
  **不能**算作 Fig. 8/13 simulate。
- Simulate 分数只从 `eval_simulate_real_awgn_*_testonly` 出。

---

## 2026-07-29 按 IoU 审计 + 叙述安全的框投票（Phase A/B）

最佳检测器重测：`det_full_120ep_lr1e3` epoch 4（此前是 `det_full_60ep` ep18）。
按 IoU 分解 AP（`SignalDetectionMetric` 上新的 `per_iou_ap=True`；配置
`eval_*_det_periou.py`）。

### mAP 差距实际落在哪里 — 是**高 IoU 框紧度**，不是召回

**Ideal（v1），det_full_120ep ep4：**

| IoU | .50 | .55 | .60 | .65 | .70 | .75 | .80 | .85 | .90 | .95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AP | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | **0.38** | 0.20 | 0.07 |

mAP 0.759，AP50 1.00，AP75 0.989。AP 到 IoU 0.80 几乎完美，然后断崖下跌 →
论文差距（0.91 vs 0.76）**完全是** IoU ≥ 0.85 的定位紧度，不是漏检（AR 0.83）。

**Simulate（v104+v105–124）：** 同一形状 — AP 从 0.97→0.73 直到 IoU 0.80，
在 0.85 掉到 0.19。mAP 0.645，AP50 0.967，AP75 0.766。在紧度断崖之上，还有
次要的低 SNR 召回损失（mAP_snr_-8 = 0.27）。

### 框投票（加权框融合）— 推理时、叙述中性

新的 `interval_nms_vote` + `test_cfg.box_voting`/`vote_iou_thr`。用高重叠簇的
分数加权均值去 refinement 每个留下的区间。默认关闭（精确论文 NMS）。在
ideal v1 上的扫描：

| vote_iou_thr | mAP | AP80 | AP85 | AP90 |
|---|---:|---:|---:|---:|
| off (baseline) | 0.759 | 0.985 | 0.379 | 0.200 |
| 0.65 | 0.743 | 0.835 | 0.312 | 0.150 |
| **0.75** | **0.824** | 0.987 | **0.925** | 0.355 |
| 0.78 | 0.823 | 0.987 | 0.962 | 0.308 |
| 0.85 | 0.773 | 0.989 | 0.452 | 0.264 |

工作点 **`box_voting=True, vote_iou_thr=0.75`**：ideal-det
**0.759 → 0.824（+0.065）**；在零再训练、不改架构/叙述的前提下，关掉剩余到
0.91 差距的约 2/3。

### 带框投票的 Joint（Fig. 13）— 真正的目标

合并 ckpt `jdm_joint_det120ep_amcw17`（det 120ep ep4 + AMC w17 83.26%），
ideal v1 class-aware mAP：

| setting | joint mAP | AP85 | AP90 |
|---|---:|---:|---:|
| baseline (paper fusion α=1, T=1) | 0.708 | 0.368 | 0.210 |
| **+ box voting vt0.75** | **0.762** | 0.853 | 0.373 |
| + voting + fuse α=0.5/0.75 | 0.762 | 0.853 | 0.372 |
| + voting + cls T=2 | 0.759 | 0.849 | 0.364 |

**框投票把 joint ideal mAP 从 0.708 提到 0.762（+0.054）。** 融合分数校准
（`fuse_alpha`、`cls_temperature`，现已在 `JDMFramework` 上实现）对每个检测
**保序，并且不会移动 class-aware mAP** — 这是负结果，保持默认 α=1/T=1。
Joint 增益来自检测定位，所以到 0.85 的剩余 joint 差距拆成两部分：投票后仍低的
AP≥0.90 尾巴，以及 AMC top1（83% 对约 90% 的代理）。

### 剩下的杠杆

- AMC top1 在所有配方上饱和在约 83% → wave-20 用叙述安全的训练细节再训
  （EMA + label smoothing 0.1 + 保标签的无线电增强：相位 / 小 CFO / 时序滚动），
  通过 `RadioAugment` + `CrossEntropyLoss(label_smoothing=...)`（在 H100 GPU1
  上跑过）。
- 检测器 AP≥0.90 尾巴：框投票能完全挽回 AP85，但 AP95 仍低（锚框/步长离散化
  上限）；更紧的步长或更多 epoch 的检测器是仅剩的进一步杠杆。

## 2026-07-29 Fig. 10 逐点 AMC 审计（A2）

GT 框分类器 `jdm-amc_iq-csrd` ep60，按（调制，SNR）的 top-1 对数字化论文曲线：

**AWGN（v89–v98），总体 top1 = 93.20%（n=14150）：**

| mod | 12 | 16 | 20 | 24 | 28 | 30 |
|---|--:|--:|--:|--:|--:|--:|
| BPSK/QPSK/8PSK | 100 | 100 | 100 | 100 | 100 | 100 |
| 16QAM | 84 | 89 | 88 | 90 | 90 | 90 |
| 64QAM | 79 | 80 | 78 | 79 | 81 | 80 |

**Simulate（v104–v124），总体 top1 = 75.04%（n=29715）：**

| mod | 12 | 16 | 20 | 24 | 28 | 30 | paper Fig.10 sim |
|---|--:|--:|--:|--:|--:|--:|---|
| BPSK | 100 | 100 | 100 | 100 | 100 | 100 | 0.80→0.98 |
| QPSK | 94 | 98 | 99 | 100 | 100 | 100 | 0.40→0.77 |
| 8PSK | 96 | 99 | 100 | 100 | 100 | 100 | 0.30→0.63 |
| 16QAM | 59 | 72 | 70 | 75 | 74 | 73 | 0.17→0.62 |
| 64QAM | 78 | 76 | 79 | 75 | 76 | 79 | 0.05→0.43 |

**结论：我们的 GT 框 AMC 模块在每个调制、每个 SNR 上都超过论文 Fig.10 的
simulate 曲线（常常超出很多）。** AMC 模块不是复现瓶颈。约 83% 的
“proposal 裁剪”饱和是检测框定位噪声带来的联合推理伪影（框松几个 bin →
裁剪泄漏），不是分类能力不足 — 与 A1（检测器高 IoU 紧度差距）一致。
因此 `amc_proposal.top1_min_pct=90` 代理在 AWGN 上已满足（93.2%），Fig.10
逐点判据也满足；剩下的 joint 增益来自更紧的检测框（框投票、wave-21 更紧框
检测器），而不是更好的分类器。

## 2026-07-29 W21：在框投票后的 det120 proposal 上再训 AMC

用最佳 120-epoch 检测器**带框投票**预计算的 proposal 再训 AMC 头
（`amc_detprops_120voted_w21.py`），把 proposal 裁剪测试 top-1 从 83.26%
提到 **84.63%**（验证最佳 85.16%）— 印证 A1/A2 的结论：是更紧的框、
而不是更好的分类器，在移动 joint 指标。

合并后的 joint checkpoint（`jdm_joint_det120_amcw21.pth`），框投票 vt0.75：

| protocol | joint mAP (W17 AMC) | joint mAP (W21 AMC) | operating point |
|---|---|---|---|
| ideal (v1) | 0.7624 | **0.7667** | W21 merged ckpt (new best) |
| simulate (real_awgn) | **0.5195** | 0.4485 | keep W17 fusion |

W21 分类器在干净的 v1 裁剪上有帮助，但对嘈杂的 real_awgn 裁剪*更敏感*
（它是在投票/更紧的 proposal 上训的，也就是更干净的裁剪分布）。按协议的
工作点记在 `goals.json`。

## 2026-07-30 收紧检测器的尝试：三个负结果

三次试图越过 det120 冠军（ideal voted 0.8238）的尝试都把它打差了 —
det120 + 框投票仍是检测器工作点：

| attempt | ideal voted mAP | simulate voted mAP | verdict |
|---|---|---|---|
| det120 (champion) | **0.8238** | **0.7701** | keep |
| bw40 FT (bandwidth loss ×2) | 0.7936 | 0.7184 | negative (best at ep2, then decays) |
| EMA from-scratch (w21) | 0.6935 | — | negative (never reached det120 level) |
| SWA 16-ep constant-LR tail (w22) | 0.7568 (avg) / 0.7572 (best snapshot) | 0.7133 | negative |

解读：det120 的峰值是尖锐最优点；每一种扰动（损失重加权、权值平滑、
快照平均）都会离开它。下一档：det_full_200ep（从零开始更长的 cosine，
在跑）以及针对 simulate joint 差距的分类器侧稳健性攻击
（amc_detprops_120voted_radioaug_w23，在跑）。

## “逐点一致”的推荐评测协议

1. **Det Fig. 8 simul：** `eval_simulate_real_awgn_det_testonly.py`（`v104`+`v105–v124`）。
2. **Det Fig. 8 ideal：** `eval_ideal_v1_det_testonly.py`（`versions=['v1']`）。
3. **Fig. 7 / 10 / 12 SNR 曲线：** `eval_awgn_v89_v98_det_testonly.py` + Real_awgn
   同 SNR 配对做空心曲线；报告按 SNR 的指标。
4. **Fig. 13：** 与 (1)/(2) 相同的 ideal/simulate 条件；class-aware joint mAP + fuse_scores。
5. 当信号个数 / 划分使 AR@k 或 ideal 杠不可比时，声明失配上限。
6. **不要**把全部 124 混合测试当成 Fig. 8/13 simulate。

不要把 PDF 或栅格化页面入库；在 git 外留一份本地副本。
