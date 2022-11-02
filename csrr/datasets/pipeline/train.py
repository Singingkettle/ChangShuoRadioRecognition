import math

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class RebaseModLabelBySNR:

    def __init__(self, alpha=0.1, beta=10, class_num=11):
        self.alpha = alpha
        self.beta = beta
        self.class_num = class_num

    def __call__(self, results):
        snr = results['item_snr_value']
        gt = results['item_mod_label']
        target = (1 - 1.0 / self.class_num) * (
                1.0 / (1 + math.exp(-self.alpha * (snr + self.beta)))) + 1.0 / self.class_num
        label = np.zeros(self.class_num, dtype=np.float32)
        label[gt] = target
        results['mod_labels'] = label
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, beta={self.beta}, class_num={self.class_num})'


@PIPELINES.register_module()
class SigmoidLossWeight:
    def __init__(self, alpha=0.1, beta=10, class_num=11):
        self.alpha = alpha
        self.beta = beta
        self.class_num = class_num

    def __call__(self, results):
        snr = results['item_snr_value']
        gt = results['item_mod_label']
        weight = np.zeros(self.class_num, dtype=np.float32)
        weight[:] = 1.0 / (1 + math.exp(-self.alpha * (snr + self.beta)))
        weight[gt] = 1
        results['weight'] = weight
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, beta={self.beta}, class_num={self.class_num})'


@PIPELINES.register_module()
class ASSMaskWeight:
    def __init__(self, mod_cls_num, dev_cls_num, is_abs=True):
        self.mod_cls_num = mod_cls_num
        self.dev_cls_num = dev_cls_num
        self.is_abs = is_abs

    def __call__(self, results):
        gt_dev = results['item_dev_label']
        gt_mod = results['item_mod_label']
        if self.is_abs:
            mask_weight = np.zeros((self.dev_cls_num, 1), dtype=np.float32)
            mask_weight[gt_dev, 0] = 1
        else:
            mask_weight = np.zeros((self.mod_cls_num, 1), dtype=np.float32)
            mask_weight[gt_mod, 0] = 1
        results['mask_weight'] = mask_weight
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(mod_cls_num={self.mod_cls_num}, ' \
                                         f'dev_cls_num={self.dev_cls_num}, is_abs={self.is_abs})'
