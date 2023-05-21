"""
作者：ZWP
日期：2022.07.22
"""
import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset, batch_size=64,shuffle=True,drop_last=True)

class NetWork(torch.nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1.forward(x)
        return x

network=NetWork()
#print(network)

writer=SummaryWriter("nn_logs")
step=0
for data in dataloader:
    imgs,label=data
    output=network.forward(imgs)
    #print(imgs.shape)
    #print(output.shape)
    writer.add_images("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))#六通道变为三通道
    writer.add_images("output", output, step)
    step+=1

writer.close()