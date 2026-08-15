# DeepSig

[English](deepsig.md) | 简体中文

建议把数据集根目录软链接到 `$CSRR/data`。若你的目录结构不同，可能需要改配置文件里的对应路径。

```
ChangShuoRadioRecognition
├── configs
├── data
│   ├── ModulationClassification
│   │   ├── DeepSig
│   │   │   ├── 201610A
│   │   │   │   ├── train.json
│   │   │   │   ├── val.json
│   │   │   │   ├── test.json
│   │   │   │   ├── sequence_data
│   │   │   │   │   ├── iq
│   │   │   │   │   ├── ap
│   │   │   │   ├── constellation_data
│   │   │   │   │   ├── filter_size_0.010_stride_0.005
│   ├── SignalSeparation
│   │   ├── CSRR
│   │   │   ├── qpsk_16qam
│   │   │   │   ├── complex
│   │   │   │   │   ├── train_data.mat
│   │   │   │   │   ├── val_data.mat
│   │   │   │   │   ├── test_data.mat
│   │   │   │   ├── real
│   │   │   │   │   ├── train_data.mat
│   │   │   │   │   ├── val_data.mat
│   │   │   │   │   ├── test_data.mat

```

DeepSig 原始数据必须用 `tools/convert_datasets/convert_amc.py` 转成上述格式：
