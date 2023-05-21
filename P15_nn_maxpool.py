"""
作者：ZWP
日期：2022.07.23
"""
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

dataset=datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))

class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1.forward(input)
        return output

net=NetWork()
output=net.forward(input)
print(output)

writer=SummaryWriter("nn_maxpool_logs")
step=0
for data in dataloader:
    imgs,label=data
    output=net.forward(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output, step)
    step+=1
writer.close()



