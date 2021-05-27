#!/usr/bin/env bash

echo "Start ./configs/crdnn/crdnn_gru_ap_deepsig_201604C.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29520 tools/train.py ./configs/crdnn/crdnn_gru_ap_deepsig_201604C.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/crdnn/crdnn_gru_ap_deepsig_201610A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29521 tools/train.py ./configs/crdnn/crdnn_gru_ap_deepsig_201610A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/crdnn/crdnn_gru_ap_deepsig_201610B.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29522 tools/train.py ./configs/crdnn/crdnn_gru_ap_deepsig_201610B.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &

echo "Start ./configs/crdnn/crdnn_gru_ap_deepsig_201801A.py"
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=29523 tools/train.py ./configs/crdnn/crdnn_gru_ap_deepsig_201801A.py --work_dir /home/citybuster/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch >/dev/null 2>&1 &
