import torch
import torch.nn as nn
# Example of target with class indices
loss = nn.CrossEntropyLoss()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.zeros(3, 5)
output = loss(input, target)
output.backward()
print(input.grad)






