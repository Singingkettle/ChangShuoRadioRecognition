[English](README.md) | 简体中文

# SNR-Ladder — SNR 感知训练的增益能否在冻结模型重读下幸存?

> 匿名作者, "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", 审稿中 (2026)。

## 方法一段话

本文用两把尺子审计自动调制识别中的 SNR 感知训练监督。第一把是预注册的**空阶梯
(null ladder)**:在*冻结的*硬标签基线模型的验证集 logits 上,按容量递增拟合逐
SNR-bin 的映射 `F_shift ⊂ F_VS ⊂ F_aff`(逐 bin 常数平移 ⊂ 缩放+平移 ⊂ 全仿
射)。训练时方法只有在过渡带内显著超过其适用的最严格一级、且保持高信噪比保留率
时才被承认;否则同样的增益由免费重读(readout)不重训练即可取得。第二把是在生
成器逐比特精确已知的自建 clean-paired AWGN 基准上计算的**精确逐 SNR Bayes 天花
板**(因子化星座似然;
CPFSK 的 40 状态相位格前向;针对生成器帧归一化的无偏序贯重要性采样修正,附有效
样本量证书)。冻结模型到天花板的距离分解为决策赤字(逐 bin 重读免费可收)与表征
赤字(归骨干所有);在被审计的全部 SNR 感知路线中,没有方法收到超出决策赤字的部
分。

## 论文章节 → 代码映射

| 论文 | 代码 |
| --- | --- |
| 空阶梯、梯级与承认规则 | `scripts/ladder/ladder_lib.py`, `scripts/ladder/ladder_audit.py` |
| 匹配对审计(hard vs method) | `scripts/ladder/pair_ladder.py`;单模型余量:`scripts/ladder/ladder_only.py` |
| SNR 感知路线谱(focal / curriculum / snr-weight / FiLM) | `p2/` 配置 + `scripts/ladder/p2_spectrum.py` |
| 特征层探针(决策层 vs 表征层裁决) | `scripts/ladder/collect_features.py`, `scripts/ladder/representation_probe_generic.py` |
| 带证书的精确 Bayes 天花板 | `scripts/ceiling/exact_alrt.py`(`run_tier_e.sh`, `run_sis.sh`) |
| 天花板分解表 | `scripts/decomp/decomp_table.py`(读 `results/ceiling_final.csv`) |
| 架构不变性检验 | `scripts/decomp/arch_invariance.py` |
| 无标签白化梯子(V0–V3) | `scripts/decomp/whitening_ladder.py` |
| 机制统计量(S_drift / S_rot / 插件赤字) | `scripts/decomp/familyd_mech.py` |
| 估计器夹逼(集成重读、1-NN 界) | `scripts/decomp/sandwich_run.py`, `scripts/decomp/merge_sandwich.py` |
| 命题前提检验(逐 bin QDA − LDA) | `scripts/decomp/qda_lda_premise.py` |
| SNR 估计误差下的软 bin 边际化重读 | `scripts/deploy/run_softbin_scan.py`, `scripts/deploy/softbin_lib.py` |
| 论文图(天花板叠加、分解瀑布) | `scripts/figs/`(读 `results/`) |
| 训练配置:基线与 DPC 匹配对 | `cgdnet/ cnn2/ denscnn/ dscldnn/ fastmldnn/ gru2/ mcformer/ mcldnn/ mldnn/ petcgdnn/ resnet_amr/ ucsd/ dpc/` |
| 自建 AWGN 锚点基准 | `synthetic_awgn/` 配置 + `scripts/synthetic_awgn/` 生成器 |
| DPC / RCPS 损失、P2 损失、FiLM 骨干 | `csrr/models/losses/rcps_loss.py`, `csrr/models/losses/p2_losses.py`, `csrr/models/backbones/{petcgdnn,mcformer}_film.py`, `csrr/models/classifiers/snr_film.py` |

## 数据

公开基准沿用仓库标准布局 `data/ModulationClassification/`(DeepSig
RadioML2016.10A/B 与 2018.01A、UCSD RML22、HisarMod2019.1),见 `docs/dataset/`。

自建 clean-paired AWGN 锚点基准从零再生:

```bash
# MATLAB(唯一事实源;数分钟)
matlab -batch "cd configs/snr_ladder/scripts/synthetic_awgn; generate_synthetic_awgn_amc('data/synthetic_awgn_amc_v1', 1000, 128, 2026, '')"
# 或 python 后备生成器(数值对齐)
python configs/snr_ladder/scripts/synthetic_awgn/generate_python_fallback.py --output-root data/synthetic_awgn_amc_v1
python configs/snr_ladder/scripts/synthetic_awgn/validate_synthetic_awgn.py
```

重资产不入库、全部可再生:预测 pickle(每 split 的 `{pps, gts, snrs}`)由各
seed 最优 checkpoint 跑 `tools/test.py` 导出;倒数第二层特征由
`scripts/ladder/collect_features.py` 导出;DPC teacher 后验由匹配 hard 运行的
训练集预测构建(见 `dpc/` 配置中的 `base.source`);AWGN 后验 DPC 目标由
`scripts/synthetic_awgn/make_awgn_dpc_targets.py` 生成。

## 训练 / 评估

```bash
python tools/train.py configs/snr_ladder/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610B.py
python tools/test.py configs/snr_ladder/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610B.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
# 匹配对的阶梯审计
python configs/snr_ladder/scripts/ladder/pair_ladder.py \
    --hard work_dirs/<hard-run-root> --method work_dirs/<method-run-root> \
    --out work_dirs/pair_ladder.csv --tag "<cell name>"
# 自建锚点上的精确 Bayes 天花板
bash configs/snr_ladder/scripts/ceiling/run_tier_e.sh
bash configs/snr_ladder/scripts/ceiling/run_sis.sh
```

## 结果

主审计(每 cell 三个 seed;过渡带逐 bin 精度;CI 用 seed 均值上自由度 n−1 的
Student-t):

| 发现 | 测量值 |
| --- | --- |
| 方法显著超过冻结模型逐 bin 仿射重读的匹配对 | 0 / 19(8 骨干、7 数据集标签) |
| 逐 bin 常数级已追平方法的对 | 15 / 19 |
| 精确 Bayes 天花板 `Acc*`(自建锚点,exact/SIS 拼接) | −20 dB 15.6%,0 dB 67.6%,+18 dB 100.0%(±0.2 pp) |
| 带内到天花板的距离(PETCGDNN / MCformer / CGDNet) | 6.6 / 6.3 / 12.0 pp |
| 其中决策赤字(hard→readout) | −0.1 / +0.0 / +2.1 pp |
| 其中表征赤字(readout→天花板) | +6.6 / +6.3 / +9.9 pp |
| 重读后的跨架构方差比(五个数据集) | 0.73–1.16(不收缩:天花板与模型无关,距离与模型有关) |

逐 bin 曲线与精确天花板表随 `results/` 提供(`ceiling_final.csv`、
`tier_e_ceiling.csv`、`sis_correction.csv`、`decomp_synA.csv`);审计脚本会打印
完整逐 cell 表格。

## 记录在案的偏离 / 说明

- **RML22 归一化**:RML22 的 IQ 幅度比 DeepSig 低约两个数量级;不在管线最前加
  逐样本 `SelfNormalize`,MCformer/PETCGDNN 会坍缩到随机。CNN4 无需它,其匹配对
  按原样审计。坍缩的 v1 运行有记录、未隐藏。
- **MCformer @ 2018.01A** 用 `Reshape [2, 1024]`(PETCGDNN 式转置管线只喂 1 个
  通道,无法训练)。
- **t-CI 修正**:`ladder_audit.py` 用 `t(0.975, n−1)`;旧版对两 seed 行用 df=2
  (或 1.96),低估了区间宽度。
- **FiLM 尺度**:`film_scale=1.0` 在 PETCGDNN/10B 上 3/3 seed 坍缩;被审计配置
  用 `film_scale=0.1`(如实记录)。
- **未包含**:M2M4 盲 SNR 闭环(其服务器侧矩估计器未整理入库);Cover–Hart 夹逼
  数字仅作上界、不外推(未标定的 1-NN 反演偏高 +3–9 pp)。
- 配置以清洗后的 `_base_` 形式发布(与实际运行内容等价);探索性围攻变体与框架
  对齐一次性配置未入库。
- 本目录新增 `scripts/` 与 `results/` 子目录(图脚本消费的小型 KB 级测量表),
  为本仓库配置目录的首例。
- `dscldnn/` 沿用上游模态 token `ap-iq`(继承
  `_base_/datasets/hisar/ap-iq-hisar2019.py`),与 `fastmldnn/`、`mldnn/` 的
  `iq-ap` 顺序不同。
