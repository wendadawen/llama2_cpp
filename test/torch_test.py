import torch
import torch.nn as nn

w = nn.Linear(4, 3)
print(w.weight.shape)  # torch.Size([3, 4])
print(w.weight)