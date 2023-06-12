import copy

with open('rrdnn_second-stage_csrr2023_v1.py', 'r') as f:
    contents = f.read()

for i in range(2, 56):
    new_contents = copy.deepcopy(contents)
    new_contents = new_contents.replace('v1', f'v{i:d}')
    with open(f'rrdnn_second-stage_csrr2023_v{i:d}.py', 'w') as f:
        f.writelines(new_contents)

