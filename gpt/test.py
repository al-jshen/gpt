import torch
from gpt.model import PEG

x = torch.randn(4, 16**3, 256)
peg = PEG(256, 256)
y = peg(x, 16, 16, 16)
print(x.shape, y.shape)
