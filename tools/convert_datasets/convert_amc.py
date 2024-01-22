# Copyright (c) Shuo Chang. All Rights Reserved.
import argparse

from datasets import DeepSigA, DeepSigB, DeepSigC, DeepSigD, HisarMod2019, UCSDRML22

DatasetToClass = {
    # 'RML22': UCSDRML22,
    # 'RadioML.2016.04C': DeepSigA,
    # 'RadioML.2016.10A': DeepSigB,
    # 'RadioML.2016.10B': DeepSigC,
    # 'RadioML.2018.01A': DeepSigD,
    'HisarMod2019.1': HisarMod2019
}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert DeepSig Dataset')
    parser.add_argument('--data_root', type=str,
                        help='data root to deepsig data', default='../../data/ModulationClassification')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_ratios = [0.5, 0.1, 0.4]
    for dataset in DatasetToClass:
        data_generator = DatasetToClass[dataset](args.data_root, dataset, data_ratios)
        data_generator.generate()


if __name__ == '__main__':
    main()
