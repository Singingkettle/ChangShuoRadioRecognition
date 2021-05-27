## Installation

### Requirements

- Ubuntu
- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+
- g++/gcc 5+

### Install WTISignalProcesssing

a. Upgrade version of g++/gcc to 5+ for building c++ files

b. Create a conda virtual environment and activate it.

```shell
conda create -n wtisignalprocessing python=3.7 -y
conda activate wtisignalprocessing
```

c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.

d. Clone the WTISignalProcesssing repository.

```shell
git clone SaiTeam@10.117.63.34:/home/SaiTeam/Git/wtisignalprocessing.git
cd wtisignalprocessing
```

e. Install MATLAB Engine API for Python following
the [official instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
, e.g.

f. Install build requirements and then install WTISignalProcesssing.

```shell
pip install -r requirements/build.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/docs.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/runtime.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/tests.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v -e .  # or "python setup.py develop"
```

g. Install "Times New Roman" for matplotlib, following
the [link](https://blog.csdn.net/u014712482/article/details/80568540?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control)

```shell
sudo apt-get update
sudo apt-get install ttf-mscorefonts-installer
rm -rf ~/.cache/matplotlib
```

h. Install machine learning package

```shell
pip3 install -U scikit-learn
```

i. Install cvx package

```shell
pip3 install cvxopt
```
