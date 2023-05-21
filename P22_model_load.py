"""
作者：ZWP
日期：2022.07.25
"""
import torch
from torch import nn

#method1
from P22_model_save import *   #必须要引入网络结构才能load
net=torch.load("save_method_1.pth")
print(net)

#method2
net=NetWork()
net.load_state_dict(torch.load("save_method_2.pth"))
print(net)
