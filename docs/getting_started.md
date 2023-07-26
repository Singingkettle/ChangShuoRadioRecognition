# Getting Started

This page provides basic tutorials about the usage of ChangShuoRadioRecognition. For installation instructions, please
see [install.md](install.md).

## Prepare datasets

In the CSRR, you should download and unzip the dataset, and store the datafile in the following rule:
/$YourDataDir$/TaskName/Organization/Version/raw_data

For example, when you download the DeepSig dataset RadioML.2016.04C from the [link](https://www.deepsig.ai/datasets), you
should put the data file in the following manner:
```html
ModulationClassification/
├── DeepSig
│   ├── RadioML.2016.04C
│   │   ├── 2016.04C.multisnr.pkl
```
Then, you should make a soft-link to the dataset by:
```shell
cd /YourProjectSavedFolder/ChangShuoRadioReognition
ln -s /$YourDataDir$/ ./data
```
After that, you must write a new data class to process the downloaded data file to generate a unified data format for the CSRR. 
Friendly, the python scripts to process datasets of DeepSig, and HisarMOD and UCSD  have been supplied. The corresponding python scripts can be 
found in the **tools/convert_datasets**. The main script is convert_amc.py, you can run by:
```shell
cd cd /YourProjectSavedFolder/ChangShuoRadioReognition/tools/convert_datasets
python conver_amc.py
```

## Train a model

ChangShuoRadioRecognition implements distributed training and non-distributed training, which uses `CSDistributedDataParallel`
and `CSDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the
config file.

By default, we evaluate the model on the validation set after each epoch, you can change the evaluation interval by
adding the interval argument in the training config.

```python
evaluation = dict(interval=12)  # The model is evaluated per 12 training epoch.
```

**Important**: The default learning rate in config files is for 2 3090Ti GPUs. According to
the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch
size if you use different GPUs or samples per GPU, e.g., lr=0.01 for 4 GPUs * 2 sample/gpu and lr=0.08 for 16 GPUs * 4
sample/gpu.

### Train with a single GPU

For example, when you want to train a DL AMC classifier using [CNN2](configs/cnn2) on [DeepSig 201610A dataset](https://www.deepsig.ai/datasets), you can run the command: 
```shell
python tools/train.py ./configs/cnn2/cnn2_iq-deepsig-201610A.py
```
The common style is:
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM}  --master_port=2905 tools/train.py ${CONFIG_FILE} --seed 0 --launcher pytorch  
```

Optional arguments are:

- `--no_validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1,
  which can be modified in the config file) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg_options 'Key=value'`: Override some settings in the used config.

**Note**:

- `resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified
  checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
- For more clear usage, the original `load_from` is deprecated and you can
  use `--cfg_options 'load_from="path/to/you/model"'` instead. It only loads the model weights and the training epoch
  starts from 0 which is usually used for finetuning.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to
specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```


