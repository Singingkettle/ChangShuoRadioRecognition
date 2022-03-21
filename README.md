## Introduction

WTISP is an open source signal processing toolbox based on PyTorch. The framework of this project is based on
the [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmcv](https://github.com/open-mmlab/mmcv).

### Major features

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary AMC and SignalSeparation frameworks, *e.g.* CLDNN, TanhNet,
  etc.

## Changelog

v1.0.1 was released in 5/9/2020.

## Benchmark and model zoo

Supported Automatic Modulation Classification methods:

- [x] [CNN2](configs/cnn2)
  . ["Convolutional Radio Modulation Recognition Networks"](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16)
- [x] [CNN3](configs/cnn3). "Provided as Baseline"
- [x] [CNN4](config/cnn4). "Provided as Baseline"
  . ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [DS-CLDNN](configs/ds_cldnn)
  . ["Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure"](https://ieeexplore.ieee.org/document/9220797)
- [x] [DensCNN](config/denscnn)
  . ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [ResCNN](config/rescnn)
  . ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [CLDNN](config/cldnn)
  . ["Deep Neural Network Architectures for Modulation Classification"](https://ieeexplore.ieee.org/document/8335483)
- [x] [alexnetco](config/alexnetco). "Provided as Baseline in MLDNN"
- [x] [googlenetco](config/googlenetco). "Provided as Baseline in MLDNN"
- [x] [resnetco](config/resnetco). "Provided as Baseline in MLDNN"
- [x] [vggnetco](config/vggnetco). "Provided as Baseline in MLDNN"
- [x] [CLDNN2](config/cldnn2). "Provided as Baseline in MLDNN"
- [x] [CGDNN2](config/cgdnn2). "Provided as Baseline in MLDNN"
- [x] [MLDNN](config/mldnn). "New proposed by Shuo Chang"
- [x] [HCGDNN](config/hcgdnn). "New proposed by Shuo Chang"
- [X] [FMLDNN](config/fmldnn). 'New proposed by Shuo Chang'

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Getting Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of WTISP.

## Benchmark Results

Please see [summary.md](docs/summary.md) for the benchmark results.

## Version Control

For version control, we use the git. Pleease refer the [**This tutorial**](docs/git_tutorial.md)

## Acknowledgement

WTISP is an open source project that is contributed by ShuoChang. We appreciate all the contributors who implement their
methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could
serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their
own new signal processing algorithms.

**Thanks WTI!**  
**Thanks 红の豚!**

![demo image](resources/pig.jfif)

## Contact

This repo is currently maintained by Shuo Chang ([@Singingkettle](https://github.com/Singingkettle)), 
