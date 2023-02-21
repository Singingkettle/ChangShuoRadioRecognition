import torch

from .single_head_classifier import SingleHeadClassifier
from .base import BaseDNN
from ..builder import TASKS, build_method
from ...common.utils import outs2result


@TASKS.register_module()
class SSNNTwoStage(BaseDNN):

    def __init__(self, band_net, mod_net, num_band, num_mod, vis_fea=False, method_name='SSNN'):
        super(SSNNTwoStage, self).__init__()
        self.band_net = build_method(band_net)
        self.mod_net = build_method(mod_net)
        self.num_band = num_band
        self.num_mod = num_mod
        self.vis_fea = vis_fea
        self.method_name = method_name

        # init weights
        self.init_weights()

    def init_weights(self, pre_trained=None):
        """Initialize the weights in method.

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

    def forward_train(self, inputs, input_metas, targets, **kwargs):
        iqs = inputs['iqs']
        iqs_ = inputs['iqs_']
        mod_labels = targets['mod_labels']
        band_labels = targets['band_labels']

        iqs = iqs.view(-1, 1, 2, 1025)
        iqs_ = iqs_.view(-1, 1, 2, 1025)
        mod_labels = mod_labels.view(-1)
        x = self.band_net.backbone(iqs)
        x_ = self.mod_net.backbone(iqs_)
        band_losses = self.band_net.classifier_head.forward_train(x, band_labels)
        mod_losses = self.mod_net.classifier_head.forward_train(x_, mod_labels)
        losses = dict(loss_Mod=mod_losses['loss_Final'], loss_band=band_losses['loss_Final'])
        return losses

    def simple_test(self, inputs, input_metas, **kwargs):
        iqs = inputs['iqs']
        iqs_ = inputs['iqs_']
        iqs = iqs.view(-1, 1, 2, 1025)
        iqs_ = iqs_.view(-1, 1, 2, 1025)

        x = self.band_net.backbone(iqs)
        band_outs = self.band_net.classifier_head(x)

        x_ = self.mod_net.backbone(iqs_)
        mod_outs = self.mod_net.classifier_head(x_)
        mod_outs = mod_outs.view(-1, self.num_band, self.num_mod)

        results = []
        for idx in range(band_outs.shape[0]):
            band_result = outs2result(band_outs[idx, :])
            mod_result = outs2result(mod_outs[idx, :, :])
            results.append(dict(Band=band_result, Mod=mod_result))

        return results

    def forward_dummy(self, inputs, input_metas):
        iqs = inputs['iqs']
        iqs_ = inputs['iqs_']
        iqs = iqs.view(-1, 1, 2, 1025)
        iqs_ = iqs_.view(-1, 1, 2, 1025)

        x = self.band_net.backbone(iqs)
        band_outs = self.band_net.classifier_head(x)

        x_ = self.mod_net.backbone(iqs_)
        mod_outs = self.mod_net.classifier_head(x_)
        mod_outs = mod_outs.view(-1, self.num_band, self.num_mod)
        return dict(Band=band_outs, Mod=mod_outs)


@TASKS.register_module()
class SSNNSingleStage(SingleHeadClassifier):
    def __init__(self, backbone, classifier_head, num_mod, vis_fea=False, method_name=None):
        super(SSNNSingleStage, self).__init__(backbone, classifier_head, vis_fea, method_name)
        self.num_mod = num_mod

    def simple_test(self, inputs, input_metas, **kwargs):
        x = self.extract_feat(**inputs)
        outs = self.classifier_head(x, self.vis_fea, True)

        outs = outs.view(len(input_metas), -1, self.num_mod)
        results_list = []
        for idx in range(outs.shape[0]):
            result = outs2result(outs[idx, :, :])
            result = {'Final': result}
            results_list.append(result)
        return results_list
