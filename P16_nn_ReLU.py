"""
作者：ZWP
日期：2022.07.23
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[-1,1],
                    [0.5,-2]])
input=torch.reshape(input,(-1,1,2,2))
print(input)

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

writer=SummaryWriter("nn_activation_logs")

class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.ReLU1=nn.ReLU(inplace=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        ans=self.sigmoid.forward(x)
        return ans

net=NetWork()

step=0
for data in dataloader:
    imgs=data[0]
    output=net.forward(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step+=1
writer.close()