"""
作者：ZWP
日期：2022.07.24
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=128,drop_last=True)
class NetWork(torch.nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        '''self.model1=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )'''

        self.conv1=nn.Conv2d(3,32,5,padding=2)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,5,padding=2)
        self.pool2=nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(64*4*4,64)
        self.linear2=nn.Linear(64,10)


    def forward(self,x):
        #x=self.model1.forward(x)

        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.conv3.forward(x)
        x = self.pool3.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)

        return x

loss=nn.CrossEntropyLoss()
loss=loss.cuda()

net=NetWork()
net=net.cuda()

#net.load_state_dict(torch.load("net_data.pth"))

optim=torch.optim.SGD(net.parameters(),lr=0.001)

for epoch in range(100):
    running_loss=0
    for data in dataloader:
        imgs,label=data

        imgs=imgs.cuda()
        label=label.cuda()

        output=net.forward(imgs)
        result_loss=loss.forward(output,label)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss+=result_loss
    print(epoch)
    print(running_loss)

#torch.save(net.state_dict(),"net_data.pth")
