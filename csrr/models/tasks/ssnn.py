import torch

from .base import BaseDNN
from ..builder import TASKS, build_task
from ...common.utils import outs2result


@TASKS.register_module()
class SSNNTwoStage(BaseDNN):

    def __init__(self, band_net, mod_net, num_anchor, num_band, num_mod, vis_fea=False, method_name='SSNN'):
        super(SSNNTwoStage, self).__init__()
        self.band_net = build_task(band_net)
        self.mod_net = build_task(mod_net)
        self.num_anchor = num_anchor
        self.num_band = num_band
        self.num_mod = num_mod
        self.vis_fea = vis_fea
        self.method_name = method_name

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in task.

        Args:
            pre_trained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SSNNTwoStage, self).init_weights(pre_trained)
        self.band_net.init_weights(pre_trained=pre_trained)
        self.mod_net.init_weights(pre_trained=pre_trained)


    def extract_feat(self, iqs, iqs_):
        """Directly extract features from the backbone."""
        return 0

    def forward_train(self, iqs, iqs_, band_labels, mod_labels):
        iqs_ = iqs_.view(-1, 2, 4096)
        mod_labels = mod_labels.view(-1)
        x = self.band_net.backbone(iqs)
        x_ = self.mod_net.backbone(iqs_)
        band_losses = self.band_net.classifier_head.forward_train(x, band_labels)
        mod_losses = self.mod_net.classifier_head.forward_train(x_, mod_labels)
        losses = {**band_losses, **mod_losses}
        return losses

    def simple_test(self, iqs, iqs_):
        x = self.band_net.backbone(iqs)
        band_outs = self.band_net.classifier_head(x)
        sort_index = torch.argsort(band_outs, dim=1, descending=True)

        steps = sort_index.new_tensor([[i] for i in range(self.num_band)])
        sort_index = sort_index[:, :self.num_anchor]
        sort_index = torch.mul(sort_index, steps)
        sort_index = sort_index.view(-1)
        iqs_ = iqs_.view(-1, 2, 4096)
        iqs_ = torch.index_select(iqs_, 0, sort_index)

        x_ = self.mod_net.backbone(iqs_)
        mod_outs = self.mod_net.classifier_head(x_)
        mod_outs = mod_outs.view(-1, self.num_anchor, self.num_mod)

        results = []
        for idx in range(band_outs.shape[0]):
            band_result = outs2result(band_outs[idx, :])
            mod_result = outs2result(mod_outs[idx, :, :])
            results.append(dict(Band=band_result, Mod=mod_result))

        return results

    def forward_dummy(self, iqs, iqs_):
        x = self.band_net.backbone(iqs)

        band_outs = self.band_net.classifier_head(x)
        sort_index = torch.argsort(band_outs, dim=1, descending=True)

        steps = sort_index.new_tensor([[i] for i in range(self.num_band)])
        sort_index = sort_index[:, :self.num_anchor]
        sort_index = torch.mul(sort_index, steps)
        sort_index = sort_index.view(-1)
        iqs_ = iqs_.view(-1, 2, 4096)
        iqs_ = torch.index_select(iqs_, 0, sort_index)

        x_ = self.mod_net.backbone(iqs_)
        mod_outs = self.mod_net.classifier_head(x_)
        return dict(Band=band_outs, Mod=mod_outs)
