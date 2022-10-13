gpu_num = 8
config_num = 5
for gpu_index in range(gpu_num):
    for config_index in range(gpu_index, config_num, gpu_num):
        print(config_index)
    print('===============')