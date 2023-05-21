"""
作者：ZWP
日期：2022.07.25
"""
import torch
import torchvision
from torch import nn


#vgg16_false=torchvision.models.vgg16()
#vgg16_true=torchvision.models.vgg16(pretrained=True)



class NetWork(torch.nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.model1=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x=self.model1.forward(x)
        return x

net=NetWork()

#method1，在load之前要定义class：NetWork
torch.save(net,"save_method_1.pth")

#method2
torch.save(net.state_dict(),"save_method_2.pth")