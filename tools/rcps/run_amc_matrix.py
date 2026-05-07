import argparse
import subprocess
from pathlib import Path


CONFIGS = {
    ('cnn2', 'hard-ce'): 'configs/rcps/cnn2/cnn2_hard-ce_iq-snr-deepsig-201610A.py',
    ('cnn2', 'static-ls'): 'configs/rcps/cnn2/cnn2_static-ls_iq-snr-deepsig-201610A.py',
    ('cnn2', 'confidence-penalty'): 'configs/rcps/cnn2/cnn2_confidence-penalty_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-uniform'): 'configs/rcps/cnn2/cnn2_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-prior'): 'configs/rcps/cnn2/cnn2_rcps-prior_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-confusion'): 'configs/rcps/cnn2/cnn2_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'hard-ce'): 'configs/rcps/mcldnn/mcldnn_hard-ce_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'static-ls'): 'configs/rcps/mcldnn/mcldnn_static-ls_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'rcps-uniform'): 'configs/rcps/mcldnn/mcldnn_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'rcps-confusion'): 'configs/rcps/mcldnn/mcldnn_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'hard-ce'): 'configs/rcps/cgdnet/cgdnet_hard-ce_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'static-ls'): 'configs/rcps/cgdnet/cgdnet_static-ls_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-uniform'): 'configs/rcps/cgdnet/cgdnet_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-confusion'): 'configs/rcps/cgdnet/cgdnet_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'hard-ce'): 'configs/rcps/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'static-ls'): 'configs/rcps/petcgdnn/petcgdnn_static-ls_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-uniform'): 'configs/rcps/petcgdnn/petcgdnn_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-confusion'): 'configs/rcps/petcgdnn/petcgdnn_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('mcformer', 'hard-ce'): 'configs/rcps/mcformer/mcformer_hard-ce_iq-snr-deepsig-201610A.py',
    ('mcformer', 'static-ls'): 'configs/rcps/mcformer/mcformer_static-ls_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-uniform'): 'configs/rcps/mcformer/mcformer_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-confusion'): 'configs/rcps/mcformer/mcformer_rcps-confusion_iq-snr-deepsig-201610A.py',
}


def run(cmd, execute):
    print(' '.join(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate or execute the RCPS AMC experiment matrix.')
    parser.add_argument('--models', nargs='+', default=['mcldnn', 'cgdnet', 'petcgdnn', 'mcformer'])
    parser.add_argument('--methods', nargs='+', default=['hard-ce', 'static-ls', 'rcps-uniform', 'rcps-confusion'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--dataset', default='deepsig201610A')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--epsilon-max', type=float, default=None)
    parser.add_argument('--epsilon-gamma', type=float, default=None)
    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    for model in args.models:
        for method in args.methods:
            config = CONFIGS[(model, method)]
            for seed in args.seeds:
                work_dir = Path(args.work_root) / 'amc' / args.dataset / f'{model}_{method}' / f'seed_{seed}'
                cfg_options = [f'randomness.seed={seed}', f'work_dir={work_dir.as_posix()}']
                if args.max_epochs is not None:
                    cfg_options.append(f'train_cfg.max_epochs={args.max_epochs}')
                if method.startswith('rcps-'):
                    if args.epsilon_max is not None:
                        cfg_options.append(f'model.head.loss.epsilon.max={args.epsilon_max}')
                    if args.epsilon_gamma is not None:
                        cfg_options.append(f'model.head.loss.epsilon.gamma={args.epsilon_gamma}')
                if method == 'static-ls' and args.label_smoothing is not None:
                    cfg_options.append(f'model.head.loss.smoothing={args.label_smoothing}')
                if args.num_workers is not None:
                    cfg_options.extend([
                        f'train_dataloader.num_workers={args.num_workers}',
                        f'val_dataloader.num_workers={args.num_workers}',
                        f'test_dataloader.num_workers={args.num_workers}',
                    ])
                    if args.num_workers == 0:
                        cfg_options.extend([
                            'train_dataloader.persistent_workers=False',
                            'val_dataloader.persistent_workers=False',
                            'test_dataloader.persistent_workers=False',
                        ])
                run(['python', 'tools/train.py', config, '--cfg-options', *cfg_options], args.execute)
                if args.test:
                    matches = sorted(work_dir.glob('best_accuracy_top1_epoch_*.pth'))
                    checkpoint = matches[-1] if matches else work_dir / 'best_accuracy_top1_epoch_*.pth'
                    test_cmd = ['python', 'tools/test.py', config, checkpoint.as_posix(), '--work-dir',
                                work_dir.as_posix()]
                    if args.num_workers is not None:
                        test_cmd.extend([
                            '--cfg-options',
                            f'test_dataloader.num_workers={args.num_workers}',
                        ])
                        if args.num_workers == 0:
                            test_cmd.append('test_dataloader.persistent_workers=False')
                    run(test_cmd, args.execute)


if __name__ == '__main__':
    main()
