import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Build a class-prior base distribution from a CSRR annotation JSON file.')
    parser.add_argument('ann_file')
    parser.add_argument('--out', required=True)
    parser.add_argument('--laplace', type=float, default=1e-4)
    args = parser.parse_args()

    ann_path = Path(args.ann_file)
    with ann_path.open('r', encoding='utf-8') as f:
        ann = json.load(f)

    modulations = ann['metainfo']['modulations']
    counts = np.full(len(modulations), args.laplace, dtype=np.float64)
    index = {name: i for i, name in enumerate(modulations)}
    for item in ann['data_list']:
        counts[index[item['modulation']]] += 1.0
    prior = counts / counts.sum()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, prior.astype(np.float32))
    print(f'Saved prior to {out}: {prior}')


if __name__ == '__main__':
    main()
