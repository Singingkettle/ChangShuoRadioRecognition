# 1
nohup python tools/train.py ./configs/online/hcgdnn_abl-gru2_iq-x310-b210-0.25m.py --gpu-ids 0 --seed 0 > /dev/null 2>&1 &

# 2
nohup python tools/train.py ./configs/online/hcgdnn_abl-gru2_iq-x310-b210-1.2m.py --gpu-ids 1 --seed 0 > /dev/null 2>&1 &

# 3
nohup python tools/train.py ./configs/online/hcgdnn_abl-gru2_iq-x310-b210-3m.py --gpu-ids 2 --seed 0 > /dev/null 2>&1 &

# 4
nohup python tools/train.py ./configs/online/hcgdnn_abl-gru2_iq-x310-b210-line.py --gpu-ids 3 --seed 0 > /dev/null 2>&1 &

# 8
nohup python tools/train.py ./configs/online/hcgdnn_abl-gru2_iq-x310-x310-line.py --gpu-ids 7 --seed 0 > /dev/null 2>&1 &