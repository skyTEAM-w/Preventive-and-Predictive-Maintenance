import torch

from src.CWRU.CWRUCNN import CWRUCNN

model = CWRUCNN(54, 54, 55, 27, 16, 16)

model.load_state_dict(torch.load('./202311272101_96.pth'))

print(model)

