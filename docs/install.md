## Installation

### Requirements

- Ubuntu 22.04 LTS
- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8

### Install ChangShuoRadioRecognition

a. Upgrade version of g++/gcc to 5+ for building c++ files

b. Create a conda virtual environment and activate it.

```shell
conda create -n ChangShuoRadioRecognition python=3.10 -y
conda activate ChangShuoRadioRecognition
```

c. Install PyTorch and torchvision.
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

d. Install mmdet for radio detection and radio recognition.

```shell
pip install -U openmim
mim install mmengine
mim install mmdet
```

e. Clone the ChangShuoRadioRecognition repository.

```shell
git clone https://github.com/ChangShuoRadioRecognition.git
cd ChangShuoRadioRecognition
```

f. Install build requirements and then install ChangShuoRadioRecognition.

```shell
mim install -e .  # or "python setup_backup.py develop"
```

g. Install "Times New Roman" for matplotlib, following
the [link](https://blog.csdn.net/u014712482/article/details/80568540?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control)

```shell
sudo apt-get update
sudo apt-get install ttf-mscorefonts-installer
rm -rf ~/.cache/matplotlib
```

h. Install tsnecuda following the [link](https://github.com/CannyLab/tsne-cuda). 

