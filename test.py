import torch
x = torch.zeros(2, 3, 9, 9)   
print(x.flatten(0).shape)
print(x.flatten(1).shape)
print(x.flatten(2).shape)