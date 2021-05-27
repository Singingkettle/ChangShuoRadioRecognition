#!/usr/bin/env bash

echo "Start ./configs/cnn3/cnn3_deepsig_201604C.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29512 tools/train.py ./configs/cnn3/cnn3_deepsig_201604C.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/cnn3/cnn3_deepsig_201610A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29513 tools/train.py ./configs/cnn3/cnn3_deepsig_201610A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/cnn3/cnn3_deepsig_201610B.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29514 tools/train.py ./configs/cnn3/cnn3_deepsig_201610B.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/cnn3/cnn3_deepsig_201801A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29515 tools/train.py ./configs/cnn3/cnn3_deepsig_201801A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &