import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


ERROR_PATTERN = re.compile(
    r'Traceback|CalledProcessError|ERROR conda|CUDA out of memory',
    re.IGNORECASE)


def run_text(cmd, cwd=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False)
    return proc.stdout.strip()


def count_csv(path):
    root = Path(path)
    if not root.exists():
        return 0
    return len(list(root.glob('*.csv')))


def latest_csv(path, limit):
    root = Path(path)
    if not root.exists():
        return []
    files = sorted(root.glob('*.csv'), key=lambda p: p.stat().st_mtime)
    return [p.name for p in files[-limit:]]


def scan_errors(log_dir, patterns):
    hits = []
    for pattern in patterns:
        for path in sorted(Path(log_dir).glob(pattern)):
            try:
                lines = path.read_text(errors='replace').splitlines()
            except OSError as exc:
                hits.append({'file': str(path), 'line': 0, 'text': f'read failed: {exc}'})
                continue
            for line_no, line in enumerate(lines, start=1):
                if ERROR_PATTERN.search(line):
                    hits.append({'file': str(path), 'line': line_no, 'text': line[-300:]})
    return hits[-50:]


def git_info(repo):
    return {
        'branch': run_text(['git', 'branch', '--show-current'], cwd=repo),
        'commit': run_text(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo),
        'status_short': run_text(['git', 'status', '--short'], cwd=repo),
    }


def collect_snapshot(args):
    calibration_metrics = Path(args.work_root) / 'calibration_10ep' / 'metrics' / 'calibration'
    confusion_metrics = Path(args.work_root) / 'calibration_10ep_confusion' / 'metrics' / 'calibration'
    main_metrics = Path(args.work_root) / 'main_10ep_3seed' / 'metrics' / 'main'
    log_dir = Path(args.work_root) / 'logs'

    snapshot = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'repo': git_info(args.repo),
        'processes': run_text([
            'pgrep', '-af',
            'rcps_calibration_watchdog|monitor_experiments|run_calibration_grid.py|tools/train.py'
        ]).splitlines(),
        'gpus': run_text([
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader'
        ]).splitlines(),
        'metrics': {
            'calibration_10ep': {
                'count': count_csv(calibration_metrics),
                'expected': args.expected_calibration,
                'latest': latest_csv(calibration_metrics, args.latest),
            },
            'calibration_10ep_confusion': {
                'count': count_csv(confusion_metrics),
                'expected': args.expected_confusion,
                'latest': latest_csv(confusion_metrics, args.latest),
            },
            'main_10ep_3seed': {
                'count': count_csv(main_metrics),
                'expected': args.expected_main,
                'latest': latest_csv(main_metrics, args.latest),
            },
        },
        'artifacts': {
            'selected_static_uniform': str(
                calibration_metrics / 'selected_static_uniform.csv'),
            'selected_static_uniform_exists': (
                calibration_metrics / 'selected_static_uniform.csv').exists(),
            'selected_confusion': str(confusion_metrics / 'selected_confusion.csv'),
            'selected_confusion_exists': (
                confusion_metrics / 'selected_confusion.csv').exists(),
            'prior': str(Path(args.work_root) / 'priors' / 'deepsig201610A.npy'),
            'prior_exists': (
                Path(args.work_root) / 'priors' / 'deepsig201610A.npy').exists(),
            'confusion_base': str(
                Path(args.work_root) / 'confusion_bases' / 'deepsig201610A.npy'),
            'confusion_base_exists': (
                Path(args.work_root) / 'confusion_bases' / 'deepsig201610A.npy').exists(),
        },
        'errors': scan_errors(
            log_dir,
            [
                'calibration_10ep_gpu*.log',
                'calibration_10ep_confusion_gpu*.log',
                'main_10ep_3seed_gpu*.log',
                'rcps_calibration_watchdog.log',
                'rcps_main_10ep_watchdog.log',
            ]),
    }

    snapshot['status'] = 'error' if snapshot['errors'] else 'running'
    if (snapshot['metrics']['calibration_10ep']['count'] >= args.expected_calibration
            and snapshot['metrics']['calibration_10ep_confusion']['count'] >= args.expected_confusion):
        snapshot['status'] = 'complete' if not snapshot['errors'] else 'error'
    if snapshot['metrics']['main_10ep_3seed']['count'] > 0:
        snapshot['status'] = 'main-running'
        if snapshot['metrics']['main_10ep_3seed']['count'] >= args.expected_main:
            snapshot['status'] = 'main-complete'
        if snapshot['errors']:
            snapshot['status'] = 'error'
    return snapshot


def append_jsonl(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + '\n')


def append_markdown(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = snapshot['metrics']['calibration_10ep']
    conf = snapshot['metrics']['calibration_10ep_confusion']
    main = snapshot['metrics']['main_10ep_3seed']
    with path.open('a', encoding='utf-8') as f:
        f.write(f'\n## {snapshot["timestamp"]} - {snapshot["status"]}\n\n')
        f.write(f'- Commit: `{snapshot["repo"]["commit"]}` on `{snapshot["repo"]["branch"]}`\n')
        f.write(f'- Calibration CSV: {cal["count"]}/{cal["expected"]}\n')
        f.write(f'- Confusion CSV: {conf["count"]}/{conf["expected"]}\n')
        f.write(f'- Main CSV: {main["count"]}/{main["expected"]}\n')
        f.write(f'- Active processes: {len(snapshot["processes"])}\n')
        f.write(f'- GPUs: {"; ".join(snapshot["gpus"])}\n')
        if cal['latest']:
            f.write(f'- Latest calibration CSV: `{cal["latest"][-1]}`\n')
        if conf['latest']:
            f.write(f'- Latest confusion CSV: `{conf["latest"][-1]}`\n')
        if main['latest']:
            f.write(f'- Latest main CSV: `{main["latest"][-1]}`\n')
        if snapshot['errors']:
            f.write('- Error hits:\n')
            for hit in snapshot['errors'][-5:]:
                f.write(f'  - `{hit["file"]}:{hit["line"]}` {hit["text"]}\n')
        else:
            f.write('- Error hits: none\n')


def main():
    parser = argparse.ArgumentParser(description='Record RCPS experiment status snapshots.')
    parser.add_argument('--repo', default='/home/citybuster/Projects/ChangShuoRadioRecognition')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs')
    parser.add_argument('--jsonl', default='/home/citybuster/Data/RCPS/work_dirs/logs/rcps_monitor_snapshots.jsonl')
    parser.add_argument('--markdown', default='/home/citybuster/Data/RCPS/work_dirs/logs/rcps_monitor_snapshots.md')
    parser.add_argument('--expected-calibration', type=int, default=68)
    parser.add_argument('--expected-confusion', type=int, default=48)
    parser.add_argument('--expected-main', type=int, default=96)
    parser.add_argument('--latest', type=int, default=10)
    args = parser.parse_args()

    snapshot = collect_snapshot(args)
    append_jsonl(args.jsonl, snapshot)
    append_markdown(args.markdown, snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
