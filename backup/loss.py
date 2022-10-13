import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.LogSoftmax()
input = torch.randn(3, 2)
print(input)
a = [0, 0, 1]
input[torch.arange(0, 3), a]=0
print(input)