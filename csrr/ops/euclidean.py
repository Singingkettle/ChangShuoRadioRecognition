import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class Euclidean(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Euclidean, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        return -1 * torch.cdist(input, self.weight, p=2)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features,
        )


if __name__ == '__main__':
    a = Euclidean(2, 3)
    b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
    print(a(b))
