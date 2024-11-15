## Introduction

English | [简体中文](README_zh-CN.md)

CSRR (ChangShuoRadioRecognition) is an open source signal processing toolbox based on PyTorch. The framework of this project is based on
the [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmcv](https://github.com/open-mmlab/mmcv).

### Major features

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary AMC algorithms, *e.g.* CLDNN, TanhNet,
  etc.

## Changelog
v2.0.1 was released in 8/8/2022.

v1.0.1 was released in 5/9/2020.

## Benchmark and model zoo

Supported Automatic Modulation Classification methods:
- [x] [CGDNet](configs/cgdnet). [**CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition**](https://ieeexplore.ieee.org/abstract/document/9349627)
- [x] [CLDNNW](configs/cldnnw). [**Deep architectures for modulation recognition**](https://ieeexplore.ieee.org/abstract/document/7920754)
- [x] [CLDNNL](configs/cldnnl). [**Deep Neural Network Architectures for Modulation Classification**](https://ieeexplore.ieee.org/document/8335483)
- [x] [CNN1DPF](configs/cnn1dpf). [**Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks**](https://lirias.kuleuven.be/retrieve/546033)
- [x] [CNN2](configs/cnn2). [**Convolutional Radio Modulation Recognition Networks**](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16)
- [x] [CNN4](configs/cnn4). [**Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels**](https://ieeexplore.ieee.org/abstract/document/9128408)
- [x] [DAE](configs/dae). [**Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder**](https://ieeexplore.ieee.org/abstract/document/9487492)
- [x] [DensCNN](configs/denscnn). [**Deep Neural Network Architectures for Modulation Classification**](https://ieeexplore.ieee.org/document/8335483)
- [x] [DSCLDNN](configs/dscldnn). [**Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure**](https://ieeexplore.ieee.org/document/9220797)
- [x] [FastMLDNN](configs/fastmldnn). [**A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification**](https://ieeexplore.ieee.org/abstract/document/10239249)
- [x] [GRU2](configs/gru2). [**Automatic modulation classification using recurrent neural networks**](https://ieeexplore.ieee.org/abstract/document/8322633)
- [x] [HCGDNN](configs/hcgdnn). [**A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification**](https://ieeexplore.ieee.org/document/9764618)
- [x] [LSTM2](configs/lstm2). [**Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors**](https://ieeexplore.ieee.org/abstract/document/8357902)
- [x] [IC-AMCNet](configs/icamcnet). [**CNN-Based Automatic Modulation Classification for Beyond 5G Communications**](https://ieeexplore.ieee.org/abstract/document/8977561)
- [x] [MCformer](configs/MCformer). [**MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification**](https://ieeexplore.ieee.org/abstract/document/9685815)
- [x] [MCLDNN](configs/mcldnn). [**A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition**](https://ieeexplore.ieee.org/abstract/document/9106397)
- [x] [MCNET](configs/mcnet). [**MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification**](https://ieeexplore.ieee.org/abstract/document/8963964)
- [x] [MLDNN](configs/mldnn). [**Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification**](https://ieeexplore.ieee.org/document/9462447)
- [x] [PET-CGDNN](configs/petcgdnn). [**An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation**](https://ieeexplore.ieee.org/abstract/document/9507514)
- [x] [TRN](configs/trn). [**Signal Modulation Classification Based on the Transformer Network**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340)


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
