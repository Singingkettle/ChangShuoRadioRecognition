import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


def unit_vector(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)

    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_norm = unit_vector(a, eps)
    b_norm = unit_vector(b, eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class Cosine(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None, eps=1e-8) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Cosine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        cos = sim_matrix(input, self.weight)
        return cos

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features,
        )


if __name__ == '__main__':
    a = Cosine(2, 3)
    b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
    print(a(b))
