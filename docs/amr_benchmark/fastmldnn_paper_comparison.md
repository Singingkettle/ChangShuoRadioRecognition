# FastMLDNN：论文 vs AMR-Benchmark vs CSRR PyTorch 对比分析

**分析日期：** 2026-07-08  
**论文：** *A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification*  
IEEE TCCN 2023, 9(6):1503–1518, doi [10.1109/TCCN.2023.3309010](https://doi.org/10.1109/TCCN.2023.3309010)

---

## 1. PDF 可访问性

| 尝试路径 | 结果 |
|---------|------|
| `/mnt/d/Documents/Personal/科研论文/03-正式出版/20-A_Fast_Multi-Loss_Learning_Deep_Neural_Network_for_Automatic_Modulation_Classification.pdf` | **未找到** |
| `/d/Documents/Personal/...`（同上） | **未找到** |
| `$HOME` 下 glob `*Fast*MLDNN*.pdf` / `*Fast_Multi-Loss*.pdf` | **未找到**（home 级搜索超时/无匹配） |
| 工作区内 `*.pdf` | 仅有 `work_dirs/performance/` 下的混淆矩阵/曲线图，**非论文原文** |

**结论：本次分析未能直接读取用户提供的 Windows PDF。** 论文数字与训练细节主要来自：

1. 仓库内 **原始论文复现配置** `configs/fastmldnn/paper/`（作者发布）
2. `docs/amr_benchmark/accuracy_targets.md` 中整理的锚点数字
3. ULNN 论文（Sci. Rep. 2024）对 FastMLDNN 的独立复现
4. IEEE / Semantic Scholar 摘要

---

## 2. 论文提取（间接来源）

### 2.1 网络结构

| 组件 | 论文 / 作者原始配置 | CSRR PyTorch (`csrr/models/backbones/fastmldnn.py`) |
|------|---------------------|---------------------------------------------------|
| 输入 | I/Q + A/P 双流拼接；`[iq: 2×128]` + `[ap: 2×128]` → 4 通道 | 同：`torch.concat([ap, iq], dim=1)`，shape `[B,4,128]` |
| 前端 CNN | **分组卷积** 3 层；`groups=(2,16,4)`；`hidden=256`；`kernel=3, stride=1` | **一致**（`channel_mode=True` 时 `Conv1d`） |
| 时序模块 | **Transformer Encoder**（1 head，`input_size=100`，`ffn=256`） | **一致**（`nn.TransformerEncoderLayer`） |
| 序列聚合 | `merge='sum'`（论文消融最优） | **一致** |
| 分类头 | `Linear(100→288)` → ReLU → Dropout(0.5) → `Linear(288→C, bias=False)` | **一致** |
| 参数量 | **≈159k**（论文 headline；ULNN/MDPI 引用） | 架构冻结，参数量应匹配 |
| Dropout | 主配置 `dp=0.5`；channel 预训练配置 `dp=0.07` | AMR 默认 `dp=0.5` |

**架构结论：CSRR PyTorch 骨干与论文描述一致，属冻结范围，不建议为追精度改结构。**

已知端口差异（文档化，非缺陷）：

- `configs/fastmldnn/paper/` 使用旧版 `FMLNet` + `channel_mode=False`（2D 分组卷积）与新版 `FastMLDNN`（1D channel 模式）并存；AMR 路径使用新版 `FastMLDNN` + `channel_mode=True`。
- 论文 stage2 配置 `fastmldnn_stage2_iq-ap-deepsig-201610A.py` 使用 `beta=0.5`（启用中心距离损失）；**AMR benchmark 配置 `beta=0`（关闭）**——见 §3。

### 2.2 多损失公式

论文提出 **Class Center Distance Expansion Loss（类中心距离扩展损失）** 与 **Cross-Entropy** 联合训练（`balance` 超参，论文 channel 配置 `balance=0.5`）。

CSRR 实现（`csrr/models/heads/fast_mldnn_head.py`）：

```python
loss_cls   = CE(cls_score, target)          # loss_weight 默认 0.1，AMR 配置 1.0
loss_center = CE(sim_matrix(W, W), arange(C))  # beta 控制权重
```

| 来源 | CE 权重 | 中心距离损失 (`beta` / `balance`) |
|------|---------|-----------------------------------|
| **论文**（`paper/fastmldnn_iq-ap-channel-deepsig201610A.py`） | 1.0（Focal 变体亦有） | **0.5** |
| **论文 stage2**（`fastmldnn_stage2_iq-ap-deepsig-201610A.py`） | 1.0 | **0.5** |
| **AMR benchmark**（`fastmldnn_iq-ap-deepsig-201610A.py`） | 1.0 | **0（关闭）** |

Wave-1 训练日志中 `loss_center: 0.0000` 恒定，证实中心距离项未参与优化。

### 2.3 训练超参

| 项目 | 论文 / 作者 `paper/` 配置 | AMR-Benchmark Keras（15 模型通用） | CSRR AMR FastMLDNN |
|------|---------------------------|-----------------------------------|-------------------|
| 优化器 | Adam | Adam | Adam |
| 学习率 | **4.4e-4**（`lr=0.00044`） | 1e-3 | **4.4e-4**（`schedules.py`） |
| LR 调度 | `MultiStepLR` milestones `[800,1200]`，`max_epochs=3200`；或 step `[20,80,400,600,760]` | `ReduceLROnPlateau(val_loss)` | `CosineAnnealingLR(T_max=150)` |
| Epochs | 3200（实际早停前收敛） | 10000 + ES patience 50 | **150** + ES patience **15** |
| Batch size | 80（iq-ap）/ **640**（iq-ap-channel） | 400 | **640** |
| Early stopping | 无（旧 runner） | `val_loss`, patience 50 | `accuracy/top1`, **min_delta=0.1, patience=15** |
| 数据划分 | `train_and_validation.json` 训练，`test.json` 作验证（**泄漏式**） | 6:2:2（论文协议） | **50/10/40**（`train/validation/test.json`） |
| 输入预处理 | 缓存 IQ/AP pkl；**无 per-sample L2** | 原始 IQ 尺度 | `MLDNNIQToAP`；IQ **无 L2 归一化** |
| 梯度裁剪 | 无 | 无 | 仅 201801A 配置有 `max_norm=5.0` |

ULNN 复现论文（非 FastMLDNN 原文，但引用了 FastMLDNN 数字）使用：Adam lr=0.001、batch=128、150 epochs、验证集准确率 10 epoch 无提升则 LR×0.8——与作者 `paper/` 配置亦不同。

### 2.4 论文报告精度（仅 RML2016.10A）

| 数据集 | Overall（SNR 平均） | Peak（单 SNR） | 备注 |
|--------|---------------------|----------------|------|
| **RML2016.10A** | **63.24%** | **≈92% @ 16 dB**（ULNN 复现 high-SNR avg 91.24%） | 159k params；论文主结果 |
| RML2016.10B | 未报告 | 未报告 | — |
| RML2018.01A | 未报告 | 未报告 | — |
| HisarMod | 未报告 | 未报告 | — |

ULNN 论文补充（FastMLDNN on 10A）：

- 低 SNR [-20, -2] dB 平均：**34.78%**
- 高 SNR [0, 18] dB 平均：**91.24%**
- 独立复现 overall：**63.01%**（与 63.24% 差 0.23 pp）

---

## 3. 三向对比总表

### 3.1 架构 & 训练

| 维度 | 论文 | AMR-Benchmark Keras | CSRR PyTorch（AMR） |
|------|------|---------------------|---------------------|
| 骨干 | 分组卷积 + Transformer + sum-merge | *不适用*（非 15 模型之一） | 与论文一致 |
| 多损失 | CE + 中心距离扩展（balance≈0.5） | *不适用* | **仅 CE（beta=0）** |
| LR | 4.4e-4 | 1e-3（基线模型） | 4.4e-4 |
| 调度 | MultiStep / 3200 ep | Plateau + 长 epoch | Cosine 150 ep + 严格 ES |
| Batch | 80–640 | 400 | 640 |
| Split | train+val 合并 / test 作 val | 6:2:2 | **50/10/40** |
| IQ 归一化 | 无 L2 | 无 L2 | 无 L2（`MLDNNIQToAP` 不归一化 IQ） |

### 3.2 精度对比（pp）

| 数据集 | 论文 target | CSRR 基线 sweep | MLDNN 对照（同 split） | Wave-1 `es_off150` | Wave-1 `lr2e4_warmup` |
|--------|-------------|-----------------|------------------------|--------------------|-----------------------|
| **RML2016.10A** overall | **63.24** | **39.32** (−23.9) | **62.31** (pass) | **51.89** (−11.4 vs paper) | **40.22** |
| **RML2016.10A** peak | **~92.0** | **62.61** (−29.4) | **92.73** (pass) | *待从 paper.pkl 解析* | *待解析* |
| RML2016.10B overall | n/a | 57.81 | 65.06 | — | — |
| RML2016.10B peak | n/a | 87.75 | 93.62 | — | — |
| RML2018.01A overall | n/a | 48.05 | 57.94 | — | — |
| RML2018.01A peak | n/a | 77.45 | 90.77 | — | — |
| HisarMod overall | n/a | **5.98** | 60.06 | — | — |
| HisarMod peak | n/a | **7.90** | 73.63 | — | — |

容差（one-sided，**仅用于 tracking 表**）：overall −1.5 pp，peak −1.0 pp。

**Campaign / siege 成功标准**（与 tracking 无关）：达到论文报告值，无容差扣减。
FastMLDNN @ RML2016.10A：**overall ≥ 63.24%**，**peak ≥ 92.0%**（见
`configs/amr_benchmark/retune/goals.json`）。

---

## 4. Gap 分析（中文摘要）

### 4.1 架构：已冻结 ✓

CSRR `FastMLDNN` 骨干与论文一致。`audit_changes.md` 架构冻结政策明确：不得为 retune 改通道数/层数/merge 方式。

### 4.2 训练细节不匹配（按影响排序）

| # | 不匹配项 | 影响评估 | 证据 |
|---|---------|---------|------|
| **1** | **`beta=0` 关闭类中心距离扩展损失** | **高** — 论文核心 multi-loss 未启用；stage2 配置用 `beta=0.5` | `fastmldnn_iq-ap-deepsig-201610A.py` L17；训练 log `loss_center=0` |
| **2** | **EarlyStopping 过早截断** | **高** — 基线在 epoch 26 达 val 39.3% 后 15 epoch 无 +0.1pp 即停（总 ~41 ep）；`es_off150` 训满 150 ep 后 test overall 升至 **51.89%** | `orchestrator.log` L599；`es_off150/retune.log` |
| **3** | **IQ 无 per-sample L2 归一化** | **中–高** — 同仓库 MCLDNN 在 tiny native IQ 尺度下需 `SelfNormalize` 才能从 57.8%→62.1%；FastMLDNN 共享 `MLDNNIQToAP` 且未做 L2 | `audit_changes.md` §RNN ceiling；MLDNN pass / FastMLDNN fail |
| **4** | **50/10/40 vs 论文合并 train+val** | **中** — 预期 1–3 pp overall 落差；不足以解释 24 pp | `accuracy_targets.md` protocol caveat |
| **5** | **Cosine 150ep vs MultiStep 3200ep** | **中** — `es_off150` 显示 150 ep 仍仅 51.9%，单独加长训练不够 | retune 结果 |
| **6** | **ES 监控指标** — `accuracy/top1` vs 论文/Keras `val_loss` | **低–中** | `audit_changes.md` |
| **7** | **201801A sum-merge 尺度** — 1024 帧使 sum 特征 ~8×，需 lr↓+warmup+clip | **仅 2018** — 已修；10A 不适用 | `audit_changes.md` 2026-07-01 |

### 4.3 `accuracy_tracking.md` 是否过时？

- **FastMLDNN 10A 行（39.32% / 62.61%）与当前 sweep 结果一致，不算 stale。**
- 用户提到的 **62.11% overall** 来自 `audit_changes.md` 中 **MCLDNN** 的 per-sample L2 + ES off 实验，**不是 FastMLDNN**。勿将 MCLDNN 修复结论误套到 FastMLDNN tracking 行。
- Wave-1 retune 结果（2026-07-08）**尚未写回** `accuracy_tracking.md`（该表由 orchestrator 更新，retune 在 `work_dirs/amr_benchmark_retune/`）。

### 4.4 Wave-1 Retune 结果

| 变体 | 配置要点 | Test overall (10A) | 相对基线 39.32% | 相对论文 63.24% |
|------|---------|-------------------|-----------------|-----------------|
| 基线 sweep | lr=4.4e-4, cosine 150, ES patience=15, beta=0 | **39.32%** | — | −23.9 pp |
| `es_off150` | 同上但 **关闭 ES**，训满 150 ep | **51.89%** | **+12.6 pp** | −11.4 pp |
| `lr2e4_warmup` | lr=2e-4, 5ep warmup, grad clip, ES patience=25 | **40.22%** | +0.9 pp | −23.0 pp |
| `lr1e3` | lr=1e-3 | **失败**（config `KeyError: append`） | — | — |

**解读：**

1. 关闭 ES 是最有效的单一杠杆（+12.6 pp），但仍距论文 11+ pp。
2. 降低 LR + warmup 对 10A **无益**（该组合为 201801A 发散修复设计）。
3. 下一步 retune 应优先：**启用 `beta=0.5`** + **IQ L2 归一化** + **ES off 或更大 patience**（均在架构冻结政策允许范围内）。

### 4.5 跨数据集

| 数据集 | 现象 | 可能原因 |
|--------|------|---------|
| 10B | 57.81% overall — 尚可 | 128 帧，无 sum 尺度问题 |
| 201801A | 48.05% — 偏低 | LR 修复后仍低；长序列 sum-merge；beta=0 |
| HisarMod | **5.98%** — 近随机 | 26 类 + 1024 帧 + beta=0 + 无 L2 + 可能 ES@epoch 1；需专项 stabilisation（参照 201801A 修复 + L2） |

---

## 5. 消融配置索引（论文原文实验，仓库 `configs/fastmldnn/paper/`）

| 配置文件 | 消融内容 |
|---------|---------|
| `abl-no-class-distance-expansion` | 去掉中心距离损失 |
| `abl-balance-{0.1,0.3,0.7,0.9}` | balance 超参扫描 |
| `abl-without-group-conv` | 去掉分组卷积 |
| `abl-merge-{sum,mean,max,min,last,std,quantile,median,logsumexp}` | 序列聚合方式 |
| `abl-focal-loss` / `abl-kl-divergence` / `abl-ghmc-loss` | 替代损失 |
| `abl-cross-entropy-center-loss` | CE + center loss 变体 |

这些配置保留论文原始 **train+val / test** 划分，**不用于** AMR benchmark sweep。

---

## 6. 建议的后续 retune 队列（不改架构）

1. **`beta=0.5`**（恢复 multi-loss）— 最高优先级  
2. **`SelfNormalize(norms=dict(iq={}))`** on IQ pipeline（与 MCLDNN 成功路径对齐）  
3. **ES off 或 patience≥50, min_delta=0** — 已验证 ES 伤害收敛  
4. 组合 1+2+3 单次实验，再评估是否需 paper 级 3200ep MultiStep  
5. Hisar / 201801A：在 10A 收敛后，叠加已有 201801A lr/warmup/clip 模板  

**→ 已落地为 Wave 2 manifest：** `configs/amr_benchmark/retune/wave2_fastmldnn_manifest.json`  
（4 变体，仅 deepsig201610A；执行计划见 [`retune_campaign.md` § FastMLDNN Wave 2 plan](./retune_campaign.md#fastmldnn-wave-2-plan-2026-07-08)）

---

## 7. 主要结论（一句话）

**PDF 本次不可读；架构已对齐论文，但 AMR 路径以 `beta=0` 关闭了论文的核心 multi-loss，再叠加严格 EarlyStopping 与无 IQ L2 归一化，导致 RML2016.10A 从论文 63.24% 跌至 39.32%；仅关闭 ES 可恢复至 51.89%，仍差 ~11 pp，需启用中心距离损失并尝试 L2 归一化才能接近论文。**
