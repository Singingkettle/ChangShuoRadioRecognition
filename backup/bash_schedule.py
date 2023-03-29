import time

import gpustat

gpustats = gpustat.new_query()

is_next = False

for script_name in range(1, 2):
    while not is_next:
        time.sleep(15)
        cur_state = 0
        for gpu_info in gpustats:
            if len(gpu_info) == 0:
                cur_state += 1
        if cur_state == len(gpustats):
            is_next = True

    time.sleep(30)
    print('Do something')
    is_next = False



