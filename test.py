from safetensors.torch import load_file 
import torch 
import os 

# inputs = torch.randn(4, 3, 4)
# temp = torch.randn(4, 4)
# loss_A = torch.randn(4, )
# loss_B = torch.randn(4, )

# a_mask = loss_A < loss_B
# b_mask = loss_B < loss_A
# a_id = torch.nonzero(a_mask).squeeze()
# b_id = torch.nonzero(b_mask).squeeze()
# print(a_id, b_id)
# a_outputs = temp[a_id]
# b_outputs = temp[b_id]
# results = torch.zeros(4, 4)
# ids = torch.cat([a_id, b_id], dim=0)
# outputs = torch.cat([a_outputs, b_outputs], dim=0)
# results = torch.scatter_add(results, 0, ids.unsqueeze(1).expand(-1, 4), outputs)
# print(temp)
# print(results)
# print("*" * 100)
# print(a_outputs)
# print("*" * 100)
# print(b_outputs)
x = torch.tensor([True, False, False])
print(torch.nonzero(x).squeeze(-1))