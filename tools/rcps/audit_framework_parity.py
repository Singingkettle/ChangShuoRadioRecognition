import argparse
from pathlib import Path

from mmengine import Config


INIT_NOTES = [
    'Keras Conv2D defaults to kernel_initializer=glorot_uniform and bias_initializer=zeros.',
    'PyTorch Conv2d samples weights from a fan-in uniform distribution by default.',
    'AMC parity checks must also compare padding, channels order, Adam epsilon, BN momentum, dropout, data split, and epoch budget.',
]


def summarize_config(path):
    cfg = Config.fromfile(path)
    print(f'config: {path}')
    print(f"model: {cfg.get('model', {}).get('backbone', {}).get('type', 'unknown')}")
    train = cfg.get('train_dataloader', {})
    dataset = train.get('dataset', {})
    print(f"train batch_size: {train.get('batch_size')}")
    print(f"train ann_file: {dataset.get('ann_file')}")
    print(f"train data_root: {dataset.get('data_root')}")
    print(f"max_epochs: {cfg.get('train_cfg', {}).get('max_epochs')}")
    optimizer = cfg.get('optim_wrapper', {}).get('optimizer', {})
    print(f"optimizer: {optimizer}")
    print('pipeline:')
    for item in dataset.get('pipeline', []):
        print(f'  - {item}')
    print()


def main():
    parser = argparse.ArgumentParser(description='Print baseline parity audit notes and config summaries.')
    parser.add_argument('configs', nargs='*')
    args = parser.parse_args()

    print('Framework parity notes:')
    for note in INIT_NOTES:
        print(f'- {note}')
    print()
    for config in args.configs:
        summarize_config(Path(config))


if __name__ == '__main__':
    main()
