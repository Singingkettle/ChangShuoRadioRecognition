import torch

from .base import BaseDNN
from ..builder import METHODS, build_backbone, build_head
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model.base_model import BaseDataPreprocessor

class DetDataPreprocessor(BaseDataPreprocessor):

    def __init__(self):
        super().__init__()

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs = torch.stack(inputs)
        samplelist_boxtype2tensor(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}


@METHODS.register_module()
class BaseDetector(BaseDNN):

    def __init__(self, backbone, detector_head, method_name=None, init_cfg=None, test_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.detector_head = build_head(detector_head)
        self.test_cfg = test_cfg
        self.data_preprocessor = DetDataPreprocessor()
        if method_name is None:
            raise ValueError('You should give a method name when using this method class!')
        else:
            self.method_name = method_name

    def extract_feat(self, iqs):
        """Directly extract features from the backbone."""
        x = self.backbone(iqs)
        return x

    def forward_train(self, inputs, targets, **kwargs):
        data = dict(inputs=inputs['iqs'], data_samples=targets['data_samples'])
        data = self.data_preprocessor(data)
        x = self.extract_feat(data['inputs'])
        losses = self.detector_head.forward_train(x, data['data_samples'], **kwargs)

        return losses

    def forward_test(self, inputs, input_metas=None, **kwargs):
        x = self.extract_feat(inputs)
        results = self.detector_head(x, True, input_metas=input_metas)
        return results

    def forward_dummy(self, inputs):
        x = self.extract_feat(inputs)
        outs = self.classifier_head(x)
        return outs
