from csrr.common.utils import glob
from csrr.common.utils.config import load_json_log
from csrr.common.fileio import load as IOLoad
import pandas as pd

coco_metric_names = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
    'AR@4': 6,
    'AR@5': 7,
    'AR@6': 8,
    'AR_s@6': 9,
    'AR_m@6': 10,
    'AR_l@6': 11
}

keys = coco_metric_names.keys()
channels = []
all_data = []
versions = [
    # 1
    (4, 0.8481),
    # rician 7
    (10, 0.7818),
    (6, 0.7792),
    (7, 0.7775),
    (7, 0.7744),
    (12, 0.7726),
    (13, 0.7606),
    (11, 0.7586),
    # ray 7
    (8, 0.7796),
    (12, 0.777),
    (8, 0.7753),
    (14, 0.7711),
    (13, 0.7688),
    (12, 0.7548),
    (14, 0.7507),
    # awgn 10
    (9, 0.7781),
    (12, 0.799),
    (10, 0.8127),
    (2, 0.8185),
    (7, 0.8332),
    (3, 0.8314),
    (5, 0.8333),
    (7, 0.8302),
    (10, 0.8351),
    (6, 0.8385),

    # clockoffset 5
    (6, 0.8423),
    (5, 0.8341),
    (7, 0.8261),
    (19, 0.8166),
    (9, 0.8068),

    # real
    (13, 0.7665),

    # real different snrs
    (28, 0.6962),
    (10, 0.7207),
    (16, 0.7304),
    (21, 0.7405),
    (6, 0.7574),
    (8, 0.7617),
    (8, 0.7686),
    (8, 0.7713),
    (6, 0.7739),
    (13, 0.7753)
]
for i in range(1, 42):
    print(i - 1)
    work_dir = f'work_dirs/rrdnn_first-stage_csrr2023_v{i:d}/'
    json_paths = glob(work_dir, 'json')
    data_json = f'data/ChangShuo/v{i:d}/validation.json'
    if len(json_paths) > 0:
        log_dict = load_json_log(json_paths[0])
        info_dict = IOLoad(data_json)
        channels.append(info_dict['channels'][0])
        data = []
        for key in keys:
            data.append(log_dict[versions[i-1][0]][f'bbox_{key}'][0])
        all_data.append(data)

df = pd.DataFrame(all_data, index=channels, columns=list(keys))
df.to_excel("det.xlsx")