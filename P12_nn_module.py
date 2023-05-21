"""
作者：ZWP
日期：2022.07.22
"""
from torch import nn
import torch
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self,input):
        output=input+1
        return output

network=Network()
x=torch.tensor(1.0)
print(network.forward(x))
