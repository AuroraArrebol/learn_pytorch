"""
作者：ZWP
日期：2022.07.23
"""
import torch
import torchvision
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

class NetWork(torch.nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.linear=torch.nn.Linear(196608,10)
    def forward(self,x):
        out=self.linear.forward(x)
        return out
net=NetWork()

for data in dataloader:
    imgs,label=data
    print(imgs.shape)
    #imgs=torch.reshape(imgs,(1,1,1,-1))
    imgs=torch.flatten(imgs)
    print(imgs.shape)
    output=net.forward(imgs)
    print(output.shape)
