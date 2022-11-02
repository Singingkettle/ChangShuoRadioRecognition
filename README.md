## Introduction

English | [简体中文](README_zh-CN.md)

CSRR is an open source signal processing toolbox based on PyTorch. The framework of this project is based on
the [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmcv](https://github.com/open-mmlab/mmcv).

### Major features

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary AMC and SignalSeparation frameworks, *e.g.* CLDNN, TanhNet,
  etc.

## Changelog
v2.0.1 was released in 8/8/2022.

v1.0.1 was released in 5/9/2020.

## Benchmark and model zoo

Supported Automatic Modulation Classification methods:

- [x] [CNN2](configs/cnn2). ["Convolutional Radio Modulation Recognition Networks"](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16)
- [x] [CNN3](configs/cnn3). "Provided as Baseline"
- [x] [CNN4](configs/cnn4). "Provided as Baseline"
- [x] [DS-CLDNN](configs/dscldnn). ["Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure"](https://ieeexplore.ieee.org/document/9220797)
- [x] [DensCNN](configs/denscnn). ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [ResCNN](configs/rescnn). ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [CLDNN](configs/cldnn). ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [AlexNet](configs/alexnet). "Provided as Baseline in MLDNN"
- [x] [GoogleNet](configs/googlenet). "Provided as Baseline in MLDNN"
- [x] [ResNet](configs/resnet). "Provided as Baseline in MLDNN"
- [x] [VGGnet](configs/vggnet). "Provided as Baseline in MLDNN"
- [x] [CLDNN2](configs/cldnn2). "Provided as Baseline in MLDNN"
- [x] [CGDNN2](configs/cgdnn2). "Provided as Baseline in MLDNN"
- [x] [DecisionTree](configs/decisiontree). "Provided as Baseline in MLDNN"
- [x] [SVM](configs/svm). "Provided as Baseline in MLDNN"
- [x] [MLDNN](configs/mldnn). [**Multitask-Learning-Based Deep Neural Network for Automatic Modulation
  Classification**](https://ieeexplore.ieee.org/document/9462447)
- [x] [HCGDNN](configs/hcgdnn). [**A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification**](https://ieeexplore.ieee.org/document/9764618)
- [X] [FMLDNN](configs/fmldnn). 'New proposed by Shuo Chang'

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Getting Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of CSRR.

## Benchmark Results

Please see [summary.md](docs/summary.md) for the benchmark results.

## Version Control

For version control, we use the git. Please refer [**This tutorial**](docs/git_tutorial.md)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{csrr,
  title   = {{CSRR}: Open ChangShuo RadioRecognition Toolbox and Benchmark},
  author  = {Shuo Chang},
  journal= {coming soon},
  year={2020}
}

@article{chang2021multi,
  title={Multi-task learning based deep neural network for automatic modulation classification},
  author={Chang, Shuo and Huang, Sai and Zhang, Ruiyun and Feng, Zhiyong and Liu, Liang},
  journal={IEEE Internet of Things Journal},
  year={2021},
  publisher={IEEE}
}
@ARTICLE{9764618,
  author={Chang, Shuo and Zhang, Ruiyun and Ji, Kejia and Huang, Sai and Feng, Zhiyong},
  journal={IEEE Transactions on Wireless Communications}, 
  title={A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2022.3168884}}
```

## Acknowledgement

CSRR is an open source project that is contributed by ShuoChang. We appreciate all the contributors who implement their
methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could
serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their
own new signal processing algorithms.

![demo image](resources/pig.jfif)

## Contact

This repo is currently maintained by Shuo Chang ([@Singingkettle](https://github.com/Singingkettle)), 
