# Getting Started

This page provides basic tutorials about the usage of WTISignalProcesssing. For installation instructions, please
see [install.md](install.md).

## Prepare datasets

It is recommended to symlink the dataset root to `$WTISP/data`. If your folder structure is different, you may need to
change the corresponding paths in config files.

```
wtisignalprocessing
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
│   │   ├── WTISS
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

The deepsig data have to be converted into the specified format using `tools/convert_datasets/comvert_amc.py`:

## Train a model

WTISignalProcessing implements distributed training and non-distributed training, which uses `MMDistributedDataParallel`
and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the
config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by
adding the interval argument in the training config.

```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**\*Important\***: The default learning rate in config files is for 8 GPUs. According to
the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch
size if you use different GPUs or samples per GPU, e.g., lr=0.01 for 4 GPUs * 2 sample/gpu and lr=0.08 for 16 GPUs * 4
sample/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1,
  which can be modified in the config file) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options 'Key=value'`: Overide some settings in the used config.

**Note**:

- `resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified
  checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
- For more clear usage, the original `load-from` is deprecated and you can
  use `--cfg-options 'load_from="path/to/you/model"'` instead. It only loads the model weights and the training epoch
  starts from 0 which is usually used for finetuning.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to
specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

## Useful tools

We provide lots of useful tools under `tools/` directory.

### Analyze logs

You can plot loss/acc curves given a training log file. Run `pip install seaborn` first to install the dependency.


```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```

- Compare the classification acc of two runs in the same figure.

```shell
python tools/analyze_logs.py plot_curve log1.json log2.json --keys common/snr_mean_all --legend run1 run2
```

You can also compute the average training speed.

```shell
python tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```

The output is expected to be like the following.

```
-----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
slowest epoch 11, average time is 1.2024
fastest epoch 1, average time is 1.1909
time std over epochs is 0.0028
average iter time: 1.1959 s/iter

```

## Tutorials

Currently, we provide three tutorials for users to [add new dataset](tutorials/new_dataset.md)
, [design data pipeline](tutorials/data_pipeline.md) and [add new modules](tutorials/new_modules.md). We also provide a
full description about the [config system](config.md).

## Kill processes

```shell
sudo kill -s 9 $(ps aux | grep mldnn_*_201610A.py | awk '{print $2}')
sudo kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')
```

## GPU State

```shell
 watch -n 0.1 --color gpustat --c
```