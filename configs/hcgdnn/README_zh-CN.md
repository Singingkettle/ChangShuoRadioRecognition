# HCGDNN — 基于分层分类头卷积门控深度神经网络的自动调制识别

[English](README.md) | 简体中文

> S. Chang 等, "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification," *IEEE Transactions on Wireless Communications*, 2022. [IEEE 9764618](https://ieeexplore.ieee.org/document/9764618)

## 方法简介

HCGDNN 从卷积表示和两层双向 GRU 表示构成分层特征。三个分类头联合训练，验证集预测用于求解非负且和为 1 的约束融合权重。

## 论文章节 → 代码映射

| 论文内容 | 代码 |
|---|---|
| CNN 与分层 GRU 骨干 | `csrr/models/backbones/hcgdnn.py` |
| 三个分类头和概率融合 | `csrr/models/heads/hcgdnn_head.py` |
| 融合目标与约束求解 | `csrr/evaluation/metrics/hcgdnn.py` |
| 验证选择配置 | `hcgdnn_iq-deepsig-201610a.py` |
| 全新 60% 最终配置 | `experiments/hcgdnn_iq-deepsig-201610a_final.py` |
| checkpoint 处理与执行器 | `release_utils.py`, `reproduce.py` |

## 数据

从 DeepSig 下载 RadioML.2016.10A，并转换到 `data/ModulationClassification/DeepSig/`。CSRR 按 modulation-SNR 分层生成 50% 训练集、10% 验证集和独立的 40% 测试集；`train_and_validation.json` 是前两者严格合并得到的 60%。设置 `cache=True` 后，每个进程会在训练前把完整数据划分加载到主机内存。

```bash
python tools/convert_datasets/convert_amc.py \
  --data_root data/ModulationClassification
python configs/hcgdnn/check_release.py --check-data
```

## 训练 / 评测

```bash
# 1. 安装实测环境，并禁止 CSRR 安装过程改写依赖。
python -m pip install -r requirements/hcgdnn.txt
python -m pip install -e . --no-deps

# 2. 执行验证选择、全新 60% 训练、单次测试和固定聚合。
python configs/hcgdnn/reproduce.py --devices 0 1 2

# 3. 也可用共享入口检查单次运行。
python tools/train.py configs/hcgdnn/hcgdnn_iq-deepsig-201610a.py
python tools/train.py \
  configs/hcgdnn/experiments/hcgdnn_iq-deepsig-201610a_final.py
python tools/test.py \
  configs/hcgdnn/experiments/hcgdnn_iq-deepsig-201610a_final.py \
  work_dirs/<run>/averaged_calibrated.pth
```

流程原子记录验证 top-1 最大且并列时最早的 epoch，以及验证集得到的融合权重。随后关闭验证，在 60% 并集上从头训练，等权平均最后三个保留 checkpoint，移植已冻结的融合权重，并拒绝在同一运行目录中进行第二次测试。

## 结果

| 数据集 | 论文 MAA | 复现 MAA | 状态 |
|---|---:|---:|---|
| RadioML.2016.10A | 63.75% | 63.7864% | 已复现 |

固定聚合规则：种子 31/37/41/43/47/53 分别对最终阶段最后三个保留 checkpoint 做等权参数平均，再对六份预测做等概率平均。全程不按测试集选择权重、删除成员或依结果重试。

## 已记录的差异 / 说明

复现等级：`statistical`。

50/10/40 两阶段协议移除了历史版本把测试集用于验证的行为。实测路径固定使用轻度优化稳定化和 checkpoint 平均，同时保持论文的优化器、学习率、批大小、1600 epoch 上限、三项损失、全样本融合目标和约束求解器。CUDA 内核可能导致 checkpoint 字节不同，因此按预先声明的六种子聚合 MAA 验收。

