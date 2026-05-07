import argparse
import json
import tarfile
import urllib.request
from pathlib import Path


SPEECH_COMMANDS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
TARGET_WORDS = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go')
SNR_BINS = (-10, -5, 0, 5, 10, 20)


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f'Using existing archive: {out_path}')
        return
    print(f'Downloading {url} -> {out_path}')
    urllib.request.urlretrieve(url, out_path)


def extract(archive, out_dir):
    marker = out_dir / 'SpeechCommands' / 'speech_commands_v0.02'
    if marker.exists():
        print(f'Using existing extraction: {marker}')
        return marker
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {archive} -> {out_dir}')
    with tarfile.open(archive) as tar:
        tar.extractall(marker)
    return marker


def read_list(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def split_for(rel_path, validation, testing):
    if rel_path in validation:
        return 'validation'
    if rel_path in testing:
        return 'test'
    return 'train'


def write_annotations(dataset_dir, out_dir, snr_bins):
    validation = read_list(dataset_dir / 'validation_list.txt')
    testing = read_list(dataset_dir / 'testing_list.txt')
    rows = {'train': [], 'validation': [], 'test': []}
    for wav_path in sorted(dataset_dir.glob('*/*.wav')):
        label = wav_path.parent.name
        if label == '_background_noise_':
            continue
        rel_path = wav_path.relative_to(dataset_dir).as_posix()
        split = split_for(rel_path, validation, testing)
        mapped_label = label if label in TARGET_WORDS else 'unknown'
        for snr in snr_bins:
            rows[split].append({
                'file_name': rel_path,
                'label_name': mapped_label,
                'snr': snr,
                'reliability': float((snr - min(snr_bins)) / (max(snr_bins) - min(snr_bins))),
                'degradation': {'type': 'additive_background_noise', 'snr_db': snr},
            })
        rows[split].append({
            'file_name': rel_path,
            'label_name': mapped_label,
            'snr': 'clean',
            'reliability': 1.0,
            'degradation': {'type': 'clean'},
        })

    classes = list(TARGET_WORDS) + ['unknown']
    metainfo = {
        'classes': classes,
        'snr_bins': list(snr_bins) + ['clean'],
        'source': SPEECH_COMMANDS_URL,
        'raw_root': str(dataset_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data_list in rows.items():
        ann = {'data_list': data_list, 'metainfo': metainfo}
        out_path = out_dir / f'{split}.json'
        out_path.write_text(json.dumps(ann, ensure_ascii=False), encoding='utf-8')
        print(f'Wrote {out_path} with {len(data_list)} noisy/clean entries')


def main():
    parser = argparse.ArgumentParser(description='Prepare Speech Commands reliability annotations for RCPS.')
    parser.add_argument('--rcps-root', type=Path, default=Path('/home/citybuster/Data/RCPS'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--snr-bins', nargs='+', type=int, default=list(SNR_BINS))
    args = parser.parse_args()

    raw_dir = args.rcps_root / 'raw' / 'SpeechCommands'
    archive = raw_dir / 'speech_commands_v0.02.tar.gz'
    if args.download:
        download(SPEECH_COMMANDS_URL, archive)
    if not archive.exists():
        raise FileNotFoundError(f'Missing {archive}; rerun with --download.')
    dataset_dir = extract(archive, raw_dir)
    out_dir = args.rcps_root / 'processed' / 'ReliabilityClassification' / 'Audio' / 'SpeechCommands-v0.02'
    write_annotations(dataset_dir, out_dir, tuple(args.snr_bins))


if __name__ == '__main__':
    main()
