## Installation

### Requirements

- Ubuntu
- Python 3.8
- PyTorch 1.12.1+
- CUDA 11.6+
- g++/gcc 5+

### Install ChangShuoRadioRecognition

a. Upgrade version of g++/gcc to 5+ for building c++ files

b. Create a conda virtual environment and activate it.

```shell
conda create -n ChangShuoRadioRecognition python=3.8 -y
conda activate ChangShuoRadioRecognition
```

c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.

d. Clone the ChangShuoRadioRecognition repository.

```shell
git clone https://github.com/ChangShuoRadioRecognition.git
cd ChangShuoRadioRecognition
```

e. Install build requirements and then install ChangShuoRadioRecognition.

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v -e .  # or "python setup.py develop"
```

f. Install "Times New Roman" for matplotlib, following
the [link](https://blog.csdn.net/u014712482/article/details/80568540?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control)

```shell
sudo apt-get update
sudo apt-get install ttf-mscorefonts-installer
rm -rf ~/.cache/matplotlib
```

g. Install tsnecuda from source, following the [link](https://github.com/CannyLab/tsne-cuda). 
