#!/usr/bin/env bash

echo "Start ./configs/mcbldnn/con_128_ds10_4000_mcbldn.py"
export PYTHONPATH=/home/zry/Projects/wtisignalprocessing
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=29541 tools/train.py ./configs/mcbldnn/con_128_ds10_4000_mcbldn.py --work_dir /home/zry/Data/SignalProcessing/Workdir --seed 0 --launcher pytorch > /dev/null 2>&1 &

