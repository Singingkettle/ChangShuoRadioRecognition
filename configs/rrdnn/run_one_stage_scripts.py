import os

start = input('Enter the start version number!')
start = int(start)
for i in range(start, 55, 3):
    os.system(f'python.exe ./tools/train.py ./configs/rrdnn/rrdnn_first-stage_csrr2023_v{i:d}.py --seed 711981417')


