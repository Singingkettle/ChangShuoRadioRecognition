from .base import BaseDNN
from ..builder import METHODS, build_backbone, build_head


@METHODS.register_module()
class BaseDetector(BaseDNN):

    def __init__(self, backbone, detector_head, method_name=None, init_cfg=None, test_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.detector_head = build_head(detector_head)
        self.test_cfg = test_cfg
        if method_name is None:
            raise ValueError('You should give a method name when using this method class!')
        else:
            self.method_name = method_name

    def extract_feat(self, inputs):
        """Directly extract features from the backbone."""
        x = self.backbone(**inputs)
        return x

    def forward_train(self, inputs, targets, **kwargs):
        x = self.extract_feat(inputs)
        losses = self.detector_head.forward_train(x, targets, **kwargs)

        return losses

    def forward_test(self, inputs, input_metas=None, **kwargs):
        x = self.extract_feat(inputs)
        results = self.detector_head(x, True)
        return results

    def forward_dummy(self, inputs):
        x = self.extract_feat(inputs)
        outs = self.classifier_head(x)
        return outs
