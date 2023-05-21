"""
作者：ZWP
日期：2022.07.24
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

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
        '''
        self.conv1=nn.Conv2d(3,32,5,padding=2)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,5,padding=2)
        self.pool2=nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(64*4*4,64)
        self.linear2=nn.Linear(64,10)
        '''

    def forward(self,x):
        x=self.model1.forward(x)
        '''
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.conv3.forward(x)
        x = self.pool3.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        '''
        return x

net=NetWork()
print(net)
input=torch.ones((64,3,32,32))
output=net.forward(input)
print(output.shape)

writer=SummaryWriter("nn_Sequential_logs")
writer.add_graph(net,input)
writer.close()