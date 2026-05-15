import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


CIFAR10C_URL = 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1'
DEFAULT_CORRUPTIONS = ('gaussian_noise', 'motion_blur', 'brightness')
REQUIRED_MEMBERS = ('CIFAR-10-C/labels.npy',) + tuple(
    f'CIFAR-10-C/{name}.npy' for name in DEFAULT_CORRUPTIONS
)
CLASS_NAMES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)


def archive_has_members(archive, members):
    if not archive.exists() or archive.stat().st_size <= 0:
        return False
    try:
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
    except tarfile.TarError:
        return False
    return all(member in names for member in members)


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_has_members(out_path, REQUIRED_MEMBERS):
        print(f'Using verified archive: {out_path}')
        return
    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + '.partial')
        print(f'Existing archive is incomplete; moving {out_path} -> {backup}')
        shutil.move(str(out_path), str(backup))
    print(f'Downloading {url} -> {out_path}')
    urllib.request.urlretrieve(url, out_path)
    if not archive_has_members(out_path, REQUIRED_MEMBERS):
        raise RuntimeError(f'Downloaded archive is incomplete: {out_path}')


def extract(archive, out_dir):
    marker = out_dir / 'CIFAR-10-C' / 'labels.npy'
    if marker.exists():
        print(f'Using existing extraction: {marker.parent}')
        return marker.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {archive} -> {out_dir}')
    with tarfile.open(archive) as tar:
        tar.extractall(out_dir)
    return marker.parent


def build_annotations(cifar10c_dir, out_dir, corruptions):
    labels = np.load(cifar10c_dir / 'labels.npy').astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    metainfo = {
        'classes': list(CLASS_NAMES),
        'corruptions': list(corruptions),
        'severities': [1, 2, 3, 4, 5],
        'reliability_definition': 'reliability=(5-severity)/4 for CIFAR-10-C test corruptions',
        'source': 'https://zenodo.org/records/2535967',
    }
    data_list = []
    for corruption in corruptions:
        array_path = cifar10c_dir / f'{corruption}.npy'
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        for severity in range(1, 6):
            start = (severity - 1) * 10000
            for local_index, label in enumerate(labels.tolist()):
                data_list.append({
                    'array_file': str(array_path),
                    'index': start + local_index,
                    'label': int(label),
                    'label_name': CLASS_NAMES[int(label)],
                    'corruption': corruption,
                    'severity': severity,
                    'reliability': float((5 - severity) / 4.0),
                })
    ann = {'data_list': data_list, 'metainfo': metainfo}
    out_path = out_dir / 'test.json'
    out_path.write_text(json.dumps(ann, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {out_path} with {len(data_list)} samples')


def main():
    parser = argparse.ArgumentParser(description='Prepare CIFAR-10-C annotations for RCPS.')
    parser.add_argument('--rcps-root', type=Path, default=Path('/home/citybuster/Data/RCPS'))
    parser.add_argument('--corruptions', nargs='+', default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    raw_dir = args.rcps_root / 'raw' / 'CIFAR-10-C'
    archive = raw_dir / 'CIFAR-10-C.tar'
    if args.download:
        download(CIFAR10C_URL, archive)
    if not archive_has_members(archive, REQUIRED_MEMBERS):
        raise FileNotFoundError(
            f'Missing or incomplete {archive}; rerun with --download.'
        )
    cifar10c_dir = extract(archive, raw_dir)
    out_dir = args.rcps_root / 'processed' / 'ReliabilityClassification' / 'Vision' / 'CIFAR-10-C'
    build_annotations(cifar10c_dir, out_dir, args.corruptions)


if __name__ == '__main__':
    main()
