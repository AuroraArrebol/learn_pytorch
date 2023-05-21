"""
作者：ZWP
日期：2022.07.26
"""
import torch
from torch import nn

class NetWork(torch.nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.model1=torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),  #nn.Flatten
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x=self.model1.forward(x)
        return x