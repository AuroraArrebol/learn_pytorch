"""
作者：ZWP
日期：2022.07.24
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from  torch import nn


x=torch.tensor([1,2,3],dtype=torch.float)
y=torch.tensor([1,2,5],dtype=torch.float)

#loss1
loss=torch.nn.L1Loss(reduction='sum')
ans=loss.forward(x,y)
#loss2
loss_mse=torch.nn.MSELoss()
ans_mse=loss_mse.forward(x,y)

print(ans)
print(ans_mse)

#loss3
x=torch.tensor([[0.1,0.2,0.3],
                [1,2,3],
                [0.5,0.6,0.1]])
y=torch.tensor([1,2,0])
x=torch.reshape(x,(3,3))
cross_loss=torch.nn.CrossEntropyLoss()
ans=cross_loss(x,y)
print(ans)

#在数据集中看损失函数
dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=1,drop_last=True)
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

loss4=nn.CrossEntropyLoss()
net=NetWork()
for data in dataloader:
    imgs,label=data
    output=net.forward(imgs)
    result_loss=loss4.forward(output,label)
    result_loss.backward()
    print(result_loss)
