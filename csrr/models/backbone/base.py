from abc import ABCMeta, abstractmethod

from ...runner import BaseModule


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.
    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [
                dict(type='Xavier', layer=['Conv2d', 'Conv1d'], distribution='uniform'),
                dict(type='RNN', layer='LSTM', gain=4),
                dict(type='RNN', layer='GRU', gain=3),
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]
        super(BaseBackbone, self).__init__(init_cfg)

    @abstractmethod
    def forward(self, *args):
        """Forward computation.
        Args:
            args (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation.
        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)
