#!/usr/bin/env bash

echo "Start ./configs/mldscldnn/mldscldnn_deepsig_201604C.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29549 tools/train.py ./configs/mldscldnn/mldscldnn_deepsig_201604C.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/mldscldnn/mldscldnn_deepsig_201610A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29550 tools/train.py ./configs/mldscldnn/mldscldnn_deepsig_201610A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/mldscldnn/mldscldnn_deepsig_201610B.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29551 tools/train.py ./configs/mldscldnn/mldscldnn_deepsig_201610B.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/mldscldnn/mldscldnn_deepsig_201801A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29552 tools/train.py ./configs/mldscldnn/mldscldnn_deepsig_201801A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &
