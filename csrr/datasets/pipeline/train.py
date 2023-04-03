import math

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class RebaseModLabelBySNR:

    def __init__(self, alpha=0.25, beta=1, num_class=11):
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class

    def __call__(self, results):
        snr = results['snr']
        gt = results['modulation']
        target = (1 - 1.0 / self.num_class) / (1 + math.exp(-self.alpha * (snr + self.beta))) + 1.0 / self.num_class
        no_target = (1 - target) / (self.num_class - 1)
        label = np.ones(self.num_class, dtype=np.float32) * no_target
        label[gt] = 0
        label[gt] = 1 - np.sum(label)
        results['targets']['rl_modulations'] = label
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, beta={self.beta}, num_class={self.num_class})'


@PIPELINES.register_module()
class SigmoidLossWeight:
    def __init__(self, alpha=0.1, beta=10, num_class=11):
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class

    def __call__(self, results):
        snr = results['snr']
        gt = results['modulation']
        weight = np.zeros(self.num_class, dtype=np.float32)
        weight[:] = 1.0 / (1 + math.exp(-self.alpha * (snr + self.beta)))
        weight[gt] = 1
        results['weights']['modulation'] = weight
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, beta={self.beta}, num_class={self.num_class})'


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
