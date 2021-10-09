
import torch

def mse(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss

def hybrid_mse(output, target, coe1=1, coe2=0.2):
    comp1 = torch.mean((output - target) ** 2)
    comp2 = torch.mean((torch.sum(output, dim=-1) - torch.sum(target, dim=-1)) ** 2)
    loss = coe1 * comp1 + coe2 * comp2
    return loss
