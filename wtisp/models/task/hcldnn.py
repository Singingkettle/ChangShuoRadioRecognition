import torch

from .amc import BaseAMC
from ..builder import TASKS, build_backbone, build_head
from ...common.utils import outs2result


@TASKS.register_module()
class HCLDNN(BaseAMC):

    def __init__(self, backbone, classifier_head, channel_mode=False, train_cfg=None, test_cfg=None):
        super(HCLDNN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.classifier_head = build_head(classifier_head)
        self.channel_mode= channel_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if 'vis_fea' in self.test_cfg:
            self.vis_fea = self.test_cfg['vis_fea']
        else:
            self.vis_fea = False

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(HCLDNN, self).init_weights(pre_trained)
        self.backbone.init_weights(pre_trained=pre_trained)

        self.classifier_head.init_weights()

    def extract_feat(self, x):
        """Directly extract features from the backbone."""
        x = self.backbone(x)
        return x

    def forward_train(self, iqs, aps, cos, mod_labels):
        if iqs is None:
            x = aps
        elif aps is None:
            x = iqs
        else:
            if self.channel_mode:
                x = torch.cat((iqs, aps), dim=1)
            else:
                x = torch.cat((iqs, aps), dim=2)
        x = self.extract_feat(x)
        losses = self.classifier_head.forward_train(x, mod_labels=mod_labels)

        return losses

    def simple_test(self, iqs, aps, cos):
        if iqs is None:
            x = aps
        elif aps is None:
            x = iqs
        else:
            if self.channel_mode:
                x = torch.cat((iqs, aps), dim=1)
            else:
                x = torch.cat((iqs, aps), dim=2)
        x = self.extract_feat(x)
        outs = self.classifier_head(x, vis_fea=self.vis_fea)

        if isinstance(outs, dict):
            results_list = []
            keys = list(outs.keys())
            batch_size = outs[keys[0]].shape[0]
            for idx in range(batch_size):
                item = dict()
                for key_str in keys:
                    item[key_str] = outs2result(outs[key_str][idx, :])
                results_list.append(item)
        else:
            results_list = []
            for idx in range(outs.shape[0]):
                result = outs2result(outs[idx, :])
                results_list.append(result)

        return results_list

    def forward_dummy(self, iqs, aps, cos):
        if iqs is None:
            x = aps
        elif aps is None:
            x = iqs
        else:
            if self.channel_mode:
                x = torch.cat((iqs, aps), dim=1)
            else:
                x = torch.cat((iqs, aps), dim=2)
        x = self.extract_feat(x)
        outs = self.classifier_head(x, vis_fea=self.vis_fea, mode='test')
        return outs
