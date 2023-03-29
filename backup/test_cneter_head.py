import torch

from csrr.models import CenterHead

data = torch.rand(size=(8, 100))
y = torch.randint(0, 6, (8, ))
a = CenterHead(6, in_size=100)

m = a(data)
n = a.loss(m, y)