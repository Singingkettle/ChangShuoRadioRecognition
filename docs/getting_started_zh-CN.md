# 入门指南

[English](getting_started.md) | 简体中文

本页介绍 ChangShuoRadioRecognition 的基本用法。安装说明见 [install_zh-CN.md](install_zh-CN.md)。

## 准备数据集

在 CSRR 中，新增自有数据集时请按下面的规则存放：
```
# Your root data dir
/$YourDataDir$/TaskName/Organization/Version/

# Your folder to save the iq frames
/$YourDataDir$/TaskName/Organization/Version/iq

# Your annotation files
/$YourDataDir$/TaskName/Organization/Version/train.json
/$YourDataDir$/TaskName/Organization/Version/test.json
/$YourDataDir$/TaskName/Organization/Version/validation.json
/$YourDataDir$/TaskName/Organization/Version/train_and_validation.json
```
例如，若有一份调制分类演示数据，由 CitybusterLab 生成、版本为 2023.10.15，目录树如下：
```text
/$YourDataDir$/
├── ModulationClassification               ## TaskName
    ├── CitybusterLab                      ## Organization
        ├── 2023.10.15                     ## Version
            ├── iq                         ## iq frame folder, each file is array with shape [2, N], In CSRR, the .npy is commonly used to save the data
                ├── 000000.npy
                ├── 000001.npy
                ...
            ├── train.json                 ## annotation file for training
            ├── test.json                  ## annotation file for test
            ├── validation.json            ## annotation file for validation
            ├── train_and_validation.json  ## annotation file for training after all hyper-parameters have been selected in validation set
```
本仓库中的标注文件都采用如下格式：
```json
{
      "data_list": [
          {
            "file_name": "000000.npy",
            "modulation": "AM-DSB",
            "snr": 20
        },
        {
            "file_name": "000001.npy",
            "modulation": "QPSK",
            "snr": -6
        },
        ...
    ],
    "metainfo": {
        "author": "ShuoChang",
        "data_prefix": "iq",
        "date": "2023-10-15",
        "department": "CitybusterLab",
        "email": "changshuo@bupt.edu.cn",
        "modulations": [
            "8PSK",
            "AM-DSB",
            "BPSK",
            "CPFSK",
            "GFSK",
            "4PAM",
            "16QAM",
            "64QAM",
            "QPSK",
            "WBFM"
        ],
        "snrs": [
            -20,
            -18,
            -16,
            -14,
            -12,
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20
        ]
    }
}
```
标注 JSON 必须包含两个顶层键：`data_list` 与 `metainfo`。`data_list` 是 `iq` 目录中一部分帧的清单，每一项都要写全该帧所需信息，例如文件名、调制类型和信噪比。`metainfo` 是全局摘要，例如谁制作了数据集、制作日期、如何联系作者。其中 `modulations` 决定类别数量与名称，`snrs` 给出信噪比变化范围。

数据准备好后，请把数据集软链接到仓库：
```shell
cd ChangShuoRadioRecognition
ln -s /$YourDataDir$/ ./data
```

因此，对 DeepSig 这类公开数据集，必须先把原始文件转成上述布局。DeepSig、HisarMOD 和 UCSD 的转换脚本已经放在 **tools/convert_datasets**。主脚本是 `convert_amc.py`，可以这样运行：
```shell
cd tools/convert_datasets
python convert_amc.py
```
若使用自有数据集，可以参考这些转换脚本编写自己的处理代码，再用训练与评测脚本评估。

## 训练模型

ChangShuoRadioRecognition 同时支持分布式训练与非分布式训练，分别使用 `CSDistributedDataParallel` 与 `CSDataParallel`。

所有输出（日志与 checkpoint）会写到配置文件里 `work_dir` 指定的工作目录。

默认在每个 epoch 结束后于验证集上评估。若要改评估间隔，可在训练配置中设置 `interval`：

```python
evaluation = dict(interval=12)  # The model is evaluated per 12 training epoch.
```

**重要**：配置文件中的默认学习率对应 2 张 3090Ti。按照 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，若 GPU 数量或每卡样本数不同，学习率应与 batch size 成比例，例如 4 GPU × 2 sample/gpu 用 lr=0.01，16 GPU × 4 sample/gpu 用 lr=0.08。

### 单卡训练

例如，要在 [DeepSig 201610A 数据集](https://www.deepsig.ai/datasets) 上用 [CNN2](../configs/cnn2) 训练深度学习 AMC 分类器，可以运行：
```shell
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py
```
通用写法是：
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

若要在命令行指定工作目录，可加 `--work-dir ${YOUR_WORK_DIR}`。

### 多卡训练

```shell
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM}  --master_port=2905 tools/train.py ${CONFIG_FILE} --seed 0 --launcher pytorch
```

可选参数：

- `--no_validate`（**不建议**）：默认会在训练过程中每 k 个 epoch 评估一次（默认 k=1，可在配置中修改）。加 `--no-validate` 可关闭该行为。
- `--work_dir ${WORK_DIR}`：覆盖配置文件中的工作目录。
- `--resume_from ${CHECKPOINT_FILE}`：从已有 checkpoint 恢复。
- `--cfg_options 'Key=value'`：覆盖所用配置中的部分字段。

**说明**：

- `resume_from` 会同时加载模型权重与优化器状态，epoch 也从该 checkpoint 继承，通常用于意外中断后继续训练。
- 为避免混淆，原来的 `load_from` 已弃用，请改用 `--cfg_options 'load_from="path/to/you/model"'`。它只加载模型权重，训练 epoch 从 0 开始，通常用于微调。

### 在同一台机器上启动多个任务

若在同一台机器上启动多个任务，例如在 8 卡机器上跑 2 个 4 卡训练，需要为每个任务指定不同端口（默认 29500），以免通信冲突。

若用 `dist_train.sh` 启动训练，可在命令里设置端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```
