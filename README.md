## Introduction

English | [简体中文](README_zh-CN.md)

CSRR (ChangShuoRadioRecognition) is an open source Automatic Modulation Classification (AMC) toolbox based on PyTorch and [MMEngine](https://github.com/open-mmlab/mmengine).

### Major Features

- **Config-Driven**: All experiments are configured via Python config files
- **Multiple AMC Methods**: Supports 20+ state-of-the-art AMC algorithms
- **Performance Analysis**: Built-in tools for generating publication-ready figures and tables
- **Minimal Dependencies**: Only depends on MMEngine, no other MM-family packages required

## Supported Methods

| Method | Paper |
|--------|-------|
| [CGDNet](configs/cgdnet) | [CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9349627) |
| [CLDNNW](configs/cldnnw) | [Deep architectures for modulation recognition](https://ieeexplore.ieee.org/abstract/document/7920754) |
| [CLDNNL](configs/cldnnl) | [Deep Neural Network Architectures for Modulation Classification](https://ieeexplore.ieee.org/document/8335483) |
| [CNN1DPF](configs/cnn1dpf) | [Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks](https://lirias.kuleuven.be/retrieve/546033) |
| [CNN2](configs/cnn2) | [Convolutional Radio Modulation Recognition Networks](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16) |
| [CNN4](configs/cnn4) | [Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels](https://ieeexplore.ieee.org/abstract/document/9128408) |
| [DAE](configs/dae) | [Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder](https://ieeexplore.ieee.org/abstract/document/9487492) |
| [DensCNN](configs/denscnn) | [Deep Neural Network Architectures for Modulation Classification](https://ieeexplore.ieee.org/document/8335483) |
| [DSCLDNN](configs/dscldnn) | [Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure](https://ieeexplore.ieee.org/document/9220797) |
| [FastMLDNN](configs/fastmldnn) | [A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/10239249) |
| [GRU2](configs/gru2) | [Automatic modulation classification using recurrent neural networks](https://ieeexplore.ieee.org/abstract/document/8322633) |
| [HCGDNN](configs/hcgdnn) | [A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/document/9764618) |
| [LSTM2](configs/lstm2) | [Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors](https://ieeexplore.ieee.org/abstract/document/8357902) |
| [IC-AMCNet](configs/icamcnet) | [CNN-Based Automatic Modulation Classification for Beyond 5G Communications](https://ieeexplore.ieee.org/abstract/document/8977561) |
| [MCformer](configs/mcformer) | [MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/9685815) |
| [MCLDNN](configs/mcldnn) | [A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9106397) |
| [MCNET](configs/mcnet) | [MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/8963964) |
| [MLDNN](configs/mldnn) | [Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/document/9462447) |
| [PET-CGDNN](configs/petcgdnn) | [An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation](https://ieeexplore.ieee.org/abstract/document/9507514) |
| [TRN](configs/trn) | [Signal Modulation Classification Based on the Transformer Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340) |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.8
- MMEngine >= 0.7.0

### Install

```bash
# Create conda environment
conda create -n csrr python=3.10 -y
conda activate csrr

# Install PyTorch (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install MMEngine
pip install mmengine

# Install CSRR
git clone https://github.com/Singingkettle/ChangShuoRadioRecognition.git
cd ChangShuoRadioRecognition
pip install -e .
```

## Quick Start

### 1. Training

```bash
# Train CNN2 on DeepSig RadioML.2016.10A
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py

# Train with custom work directory
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py --work-dir work_dirs/my_exp

# Override config options
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    --cfg-options train_cfg.max_epochs=100
```

### 2. Testing

After training, generate predictions for performance analysis:

```bash
# Test and save predictions
python tools/test.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    work_dirs/cnn2_iq-deepsig-201610A/best_accuracy_top1_epoch_*.pth \
    --work-dir work_dirs/cnn2_iq-deepsig-201610A
```

This saves `paper.pkl` containing predictions, ground truth, and SNR values.

### 3. Performance Analysis (Plotting)

Generate publication-ready figures by comparing multiple trained models:

**Step 1**: Configure method registry in `configs/_base_/performance_info/amc.py`:

```python
info = dict(
    work_dir='work_dirs',
    methods={
        'CNN2': 0,    # Index determines color/marker
        'CNN4': 1,
        'DensCNN': 2,
    },
    publish=dict(
        deepsig201610A=dict(
            CNN2='cnn2_iq-deepsig-201610A',      # method_name -> work_dir_name
            CNN4='cnn4_iq-deepsig-201610A',
            DensCNN='denscnn_iq-deepsig-201610A',
        ),
    ),
)
```

**Step 2**: Create a plot config (e.g., `configs/cnn2/cnn2_plot.py`):

```python
_base_ = ['../_base_/performance_info/amc.py']

performance = dict(
    type='Classification',
    Figures=[
        dict(
            type='SNRVsAccuracy',
            dataset=dict(
                deepsig201610A=dict(
                    comparison=['CNN2', 'CNN4', 'DensCNN'],
                ),
            ),
        ),
        dict(
            type='ConfusionMap',
            dataset=dict(deepsig201610A=['CNN2']),
        ),
    ],
)
```

**Step 3**: Generate figures:

```bash
python tools/analyze.py configs/cnn2/cnn2_plot.py
```

Output PDFs are saved to `work_dirs/performance/`.

#### Available Figure Types

| Type | Description |
|------|-------------|
| `SNRVsAccuracy` | Accuracy vs SNR curves (line plot + radar chart) |
| `ClassVsF1ScoreWithSNR` | Per-class F1 scores at each SNR |
| `ConfusionMap` | Confusion matrices per SNR |

## Project Structure

```
ChangShuoRadioRecognition/
├── configs/                 # Model and experiment configs
│   ├── _base_/             # Base configs (datasets, schedules, etc.)
│   ├── cnn2/               # CNN2 model configs
│   └── ...
├── csrr/                   # Core library
│   ├── datasets/           # Dataset classes
│   ├── models/             # Model implementations
│   ├── engine/             # Training hooks
│   ├── evaluation/         # Metrics
│   ├── performance/        # Plotting and analysis tools
│   └── ...
├── tools/                  # Command-line tools
│   ├── train.py           # Training script
│   ├── test.py            # Testing/inference script
│   └── analyze.py         # Performance analysis script
└── work_dirs/             # Output directory (created during training)
```

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@article{chang2021multi,
  title={Multi-task learning based deep neural network for automatic modulation classification},
  author={Chang, Shuo and Huang, Sai and Zhang, Ruiyun and Feng, Zhiyong and Liu, Liang},
  journal={IEEE Internet of Things Journal},
  year={2021},
  publisher={IEEE}
}

@article{chang2022hcgdnn,
  author={Chang, Shuo and Zhang, Ruiyun and Ji, Kejia and Huang, Sai and Feng, Zhiyong},
  journal={IEEE Transactions on Wireless Communications}, 
  title={A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification}, 
  year={2022},
  doi={10.1109/TWC.2022.3168884}
}

@article{chang2023fastmldnn,
  title={A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification},
  author={Chang, Shuo and others},
  journal={IEEE Transactions},
  year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

This project is maintained by Shuo Chang ([@Singingkettle](https://github.com/Singingkettle)).
