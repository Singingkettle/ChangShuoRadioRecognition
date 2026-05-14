import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    ('cnn2', 'hard-ce'): 'configs/rcps/cnn2/cnn2_hard-ce_iq-snr-deepsig-201610A.py',
    ('cnn2', 'static-ls'): 'configs/rcps/cnn2/cnn2_static-ls_iq-snr-deepsig-201610A.py',
    ('cnn2', 'confidence-penalty'): 'configs/rcps/cnn2/cnn2_confidence-penalty_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-uniform'): 'configs/rcps/cnn2/cnn2_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-retention'): 'configs/rcps/cnn2/cnn2_rcps-retention_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-prior'): 'configs/rcps/cnn2/cnn2_rcps-prior_iq-snr-deepsig-201610A.py',
    ('cnn2', 'rcps-confusion'): 'configs/rcps/cnn2/cnn2_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('cnn4', 'hard-ce'): 'configs/cnn4/cnn4_iq-deepsig-201610A.py',
    ('gru2', 'hard-ce'): 'configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py',
    ('lstm2', 'hard-ce'): 'configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py',
    ('cldnnl', 'hard-ce'): 'configs/cldnnl/cldnnl_iq-deepsig-201610A.py',
    ('cldnnw', 'hard-ce'): 'configs/cldnnw/cldnnw_iq-deepsig-201610A.py',
    ('dscldnn', 'hard-ce'): 'configs/dscldnn/dscldnn_ap-iq-deepsig-201610A.py',
    ('hcgdnn', 'hard-ce'): 'configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py',
    ('mcnet', 'hard-ce'): 'configs/mcnet/mcnet_iq-deepsig-201610A.py',
    ('denscnn', 'hard-ce'): 'configs/denscnn/denscnn_iq-deepsig-201610A.py',
    ('mcldnn', 'hard-ce'): 'configs/rcps/mcldnn/mcldnn_hard-ce_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'static-ls'): 'configs/rcps/mcldnn/mcldnn_static-ls_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'rcps-uniform'): 'configs/rcps/mcldnn/mcldnn_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'rcps-retention'): 'configs/rcps/mcldnn/mcldnn_rcps-retention_iq-snr-deepsig-201610A.py',
    ('mcldnn', 'rcps-confusion'): 'configs/rcps/mcldnn/mcldnn_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('mldnn', 'hard-ce'): 'configs/rcps/mldnn/mldnn_hard-ce_iq-ap-snr-deepsig-201610A.py',
    ('mldnn', 'static-ls'): 'configs/rcps/mldnn/mldnn_static-ls_iq-ap-snr-deepsig-201610A.py',
    ('mldnn', 'rcps-uniform'): 'configs/rcps/mldnn/mldnn_rcps-uniform_iq-ap-snr-deepsig-201610A.py',
    ('mldnn', 'rcps-retention'): 'configs/rcps/mldnn/mldnn_rcps-retention_iq-ap-snr-deepsig-201610A.py',
    ('mldnn', 'rcps-confusion'): 'configs/rcps/mldnn/mldnn_rcps-confusion_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'hard-ce'): 'configs/rcps/fastmldnn/fastmldnn_hard-ce_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'static-ls'): 'configs/rcps/fastmldnn/fastmldnn_static-ls_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'rcps-uniform'): 'configs/rcps/fastmldnn/fastmldnn_rcps-uniform_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'rcps-retention'): 'configs/rcps/fastmldnn/fastmldnn_rcps-retention_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'rcps-confusion'): 'configs/rcps/fastmldnn/fastmldnn_rcps-confusion_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'rcps-entropy'): 'configs/rcps/fastmldnn/fastmldnn_rcps-entropy_iq-ap-snr-deepsig-201610A.py',
    ('fastmldnn', 'rcps-posterior'): 'configs/rcps/fastmldnn/fastmldnn_rcps-posterior_iq-ap-snr-deepsig-201610A.py',
    ('cgdnet', 'hard-ce'): 'configs/rcps/cgdnet/cgdnet_hard-ce_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'static-ls'): 'configs/rcps/cgdnet/cgdnet_static-ls_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-uniform'): 'configs/rcps/cgdnet/cgdnet_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-retention'): 'configs/rcps/cgdnet/cgdnet_rcps-retention_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-confusion'): 'configs/rcps/cgdnet/cgdnet_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-entropy'): 'configs/rcps/cgdnet/cgdnet_rcps-entropy_iq-snr-deepsig-201610A.py',
    ('cgdnet', 'rcps-posterior'): 'configs/rcps/cgdnet/cgdnet_rcps-posterior_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'hard-ce'): 'configs/rcps/petcgdnn/petcgdnn_hard-ce_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'static-ls'): 'configs/rcps/petcgdnn/petcgdnn_static-ls_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-uniform'): 'configs/rcps/petcgdnn/petcgdnn_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-retention'): 'configs/rcps/petcgdnn/petcgdnn_rcps-retention_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-confusion'): 'configs/rcps/petcgdnn/petcgdnn_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-entropy'): 'configs/rcps/petcgdnn/petcgdnn_rcps-entropy_iq-snr-deepsig-201610A.py',
    ('petcgdnn', 'rcps-posterior'): 'configs/rcps/petcgdnn/petcgdnn_rcps-posterior_iq-snr-deepsig-201610A.py',
    ('mcformer', 'hard-ce'): 'configs/rcps/mcformer/mcformer_hard-ce_iq-snr-deepsig-201610A.py',
    ('mcformer', 'static-ls'): 'configs/rcps/mcformer/mcformer_static-ls_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-uniform'): 'configs/rcps/mcformer/mcformer_rcps-uniform_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-retention'): 'configs/rcps/mcformer/mcformer_rcps-retention_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-confusion'): 'configs/rcps/mcformer/mcformer_rcps-confusion_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-entropy'): 'configs/rcps/mcformer/mcformer_rcps-entropy_iq-snr-deepsig-201610A.py',
    ('mcformer', 'rcps-posterior'): 'configs/rcps/mcformer/mcformer_rcps-posterior_iq-snr-deepsig-201610A.py',
}


def run(cmd, execute):
    print(' '.join(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def loss_prefixes(model):
    if model == 'mldnn':
        return ['model.head.loss_amc_merge', 'model.head.loss_amc_ap', 'model.head.loss_amc_iq']
    return ['model.head.loss']


def main():
    parser = argparse.ArgumentParser(description='Generate or execute the RCPS AMC experiment matrix.')
    parser.add_argument('--models', nargs='+', default=['mldnn', 'fastmldnn', 'cgdnet', 'petcgdnn', 'mcformer'])
    parser.add_argument('--methods', nargs='+', default=['hard-ce'])
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
            prefixes = loss_prefixes(model)
            for seed in args.seeds:
                work_dir = Path(args.work_root) / 'amc' / args.dataset / f'{model}_{method}' / f'seed_{seed}'
                cfg_options = [f'randomness.seed={seed}', f'work_dir={work_dir.as_posix()}']
                if args.max_epochs is not None:
                    cfg_options.append(f'train_cfg.max_epochs={args.max_epochs}')
                if method.startswith('rcps-'):
                    for prefix in prefixes:
                        if args.epsilon_max is not None:
                            cfg_options.append(f'{prefix}.epsilon.max={args.epsilon_max}')
                        if args.epsilon_gamma is not None:
                            cfg_options.append(f'{prefix}.epsilon.gamma={args.epsilon_gamma}')
                if method == 'static-ls' and args.label_smoothing is not None:
                    for prefix in prefixes:
                        cfg_options.append(f'{prefix}.smoothing={args.label_smoothing}')
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
                run([sys.executable, 'tools/train.py', config, '--cfg-options', *cfg_options], args.execute)
                if args.test:
                    matches = sorted(work_dir.glob('best_accuracy_top1_epoch_*.pth'))
                    checkpoint = matches[-1] if matches else work_dir / 'best_accuracy_top1_epoch_*.pth'
                    test_cmd = [sys.executable, 'tools/test.py', config, checkpoint.as_posix(), '--work-dir', work_dir.as_posix()]
                    if args.num_workers is not None:
                        test_cmd.extend(['--cfg-options', f'test_dataloader.num_workers={args.num_workers}'])
                        if args.num_workers == 0:
                            test_cmd.append('test_dataloader.persistent_workers=False')
                    run(test_cmd, args.execute)


if __name__ == '__main__':
    main()
