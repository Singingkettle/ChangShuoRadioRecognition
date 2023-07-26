import copy

import numpy as np

with open('../fastmldnn_abl-merge-mean_iq-ap-channel-deepsig201610A.py', 'r') as f:
    contents = f.read()

gpu_index = 0

parallel_num = 8

gpu_00 = []
gpu_01 = []
for scale in np.arange(0.03, 1, 0.03):
    new_contents = copy.deepcopy(contents)
    new_contents = new_contents.replace('scale=1', f'scale={scale:.3f}')
    with open(f'fastmldnn_abl-merge-mean-{scale:.3f}_iq-ap-channel-deepsig201610A.py', 'w') as f:
        f.writelines(new_contents)
    if gpu_index == 0:
        gpu_00.append(f'export CUDA_VISIBLE_DEVICES={gpu_index} \npython tools/train.py '
                      f'./configs/fastmldnn/fastmldnn_abl-merge-mean-{scale:.3f}_iq-ap-channel-deepsig201610A.py '
                      f'--work_dir /home/citybuster/Data/WirelessRadio/work_dir --seed 0\n\n\n')
        gpu_index = 1
    else:
        gpu_01.append(f'export CUDA_VISIBLE_DEVICES={gpu_index} \npython tools/train.py '
                      f'./configs/fastmldnn/fastmldnn_abl-merge-mean-{scale:.3f}_iq-ap-channel-deepsig201610A.py '
                      f'--work_dir /home/citybuster/Data/WirelessRadio/work_dir --seed 0\n\n\n')
        gpu_index = 0

for scale in np.arange(5, 240, 4):
    new_contents = copy.deepcopy(contents)
    new_contents = new_contents.replace('scale=1', f'scale={scale}')
    with open(f'fastmldnn_abl-merge-mean-{scale:03d}_iq-ap-channel-deepsig201610A.py', 'w') as f:
        f.writelines(new_contents)
    if gpu_index == 0:
        gpu_00.append(f'export CUDA_VISIBLE_DEVICES={gpu_index} \npython tools/train.py '
                      f'./configs/fastmldnn/fastmldnn_abl-merge-mean-{scale:03d}_iq-ap-channel-deepsig201610A.py '
                      f'--work_dir /home/citybuster/Data/WirelessRadio/work_dir --seed 0\n\n\n')
        gpu_index = 1
    else:
        gpu_01.append(f'export CUDA_VISIBLE_DEVICES={gpu_index} \npython tools/train.py '
                      f'./configs/fastmldnn/fastmldnn_abl-merge-mean-{scale:03d}_iq-ap-channel-deepsig201610A.py '
                      f'--work_dir /home/citybuster/Data/WirelessRadio/work_dir --seed 0\n\n\n')
        gpu_index = 0


with open('run_all.sh', 'w') as af:
    for i in range(parallel_num):
        with open(f'train_mean_scale_configs_0{i:02d}.sh', 'w') as f:
            f.writelines(gpu_00[i:len(gpu_00):parallel_num])
        af.write(f'nohup bash configs/fastmldnn/train_mean_scale_configs_0{i:02d}.sh > /dev/null 2>&1 &\n\n\n')

    for i in range(parallel_num):
        with open(f'train_mean_scale_configs_1{i:02d}.sh', 'w') as f:
            f.writelines(gpu_01[i:len(gpu_01):parallel_num])
        af.write(f'nohup bash configs/fastmldnn/train_mean_scale_configs_1{i:02d}.sh > /dev/null 2>&1 &\n\n\n')





