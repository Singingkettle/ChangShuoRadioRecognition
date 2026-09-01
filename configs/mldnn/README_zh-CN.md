# MLDNN — 基于多任务学习的自动调制识别深度神经网络

[English](README.md) | 简体中文

> S. Chang 等, "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification," *IEEE Internet of Things Journal*, 2021. [IEEE 9462447](https://ieeexplore.ieee.org/document/9462447)

## 方法简介

MLDNN 从 I/Q 与幅度/相位两种视图联合学习调制识别，并使用高/低信噪比辅助任务。网络学习一个 SNR 门控来混合两条调制分类分支的概率，发布实现则在对数概率域计算该混合分支的损失。

## 论文章节 → 代码映射

| 论文内容 | 代码 |
|---|---|
| I/Q 与 A/P 分支、SNR 门控 | `csrr/models/backbones/mldnn.py` |
| 四任务损失与预测 | `csrr/models/heads/mldnn_head.py` |
| I/Q 到 A/P 的定义 | `csrr/datasets/transforms/processing.py` |
| 2016.10A 协议 | `mldnn_iq-ap-deepsig-201610a.py`, `experiments/mldnn_iq-ap-deepsig-201610a_final.py` |
| 2018.01A 协议 | `mldnn_iq-ap-deepsig-201801a.py`, `experiments/mldnn_iq-ap-deepsig-201801a_final.py` |
| 两阶段执行与检查 | `reproduce.py`, `check_release.py` |

## 数据

从 DeepSig 下载 RadioML.2016.10A 与 RadioML.2018.01A，并转换到 `data/ModulationClassification/DeepSig/`。CSRR 按分层方式生成 50% 训练集、10% 验证集和独立的 40% 测试集；`train_and_validation.json` 是前两者严格合并得到的 60%。转换程序同时生成打包 I/Q 缓存，避免逐样本读取文件。

```bash
python tools/convert_datasets/convert_amc.py \
  --data_root data/ModulationClassification
python configs/mldnn/check_release.py --check-data
```

## 训练 / 评测

```bash
# 1. 安装实测环境，并禁止 CSRR 安装过程改写依赖。
python -m pip install -r requirements/mldnn.txt
python -m pip install -e . --no-deps

# 2. 执行验证选择、全新 60% 训练、单次测试和固定聚合。
python configs/mldnn/reproduce.py --dataset all --devices 0 1 2

# 3. 也可用共享入口检查单次运行。
python tools/train.py configs/mldnn/mldnn_iq-ap-deepsig-201610a.py
python tools/train.py \
  configs/mldnn/experiments/mldnn_iq-ap-deepsig-201610a_final.py
python tools/test.py \
  configs/mldnn/experiments/mldnn_iq-ap-deepsig-201610a_final.py \
  work_dirs/<run>/epoch_<selected>.pth --phase-rotation-tta-views 8
```

该流程按验证集 top-1 最大值选择 epoch，并列时取最早者；选择记录会原子写入。随后关闭验证，在合并后的 60% 上从头训练，并拒绝在同一运行目录中进行第二次测试。

## 结果

| 数据集 | 论文 MAA | 复现 MAA | 状态 |
|---|---:|---:|---|
| RadioML.2016.10A | 63.40% | 63.5841% | 已复现 |
| RadioML.2018.01A | 60.70% | 60.7149% | 已复现 |

固定聚合规则：2016.10A 使用种子 31/37/41，每个模型固定 8 个相位视图，再做等概率平均；2018.01A 使用种子 17 和验证集选出的 epoch 370。全程不按测试集选择权重、删除成员或依结果重试。

## 已记录的差异 / 说明

复现等级：`statistical`。

50/10/40 两阶段协议移除了历史版本把测试集用于验证的行为。实测 2016 路径固定使用轻度优化稳定化与参数滑动平均；两个数据集均使用内存打包 I/Q、严格 MAA，以及论文给定的优化器、学习率、批大小、epoch 上限和四项损失。CUDA 内核可能导致 checkpoint 字节不同，因此按预先声明的聚合 MAA 验收。

