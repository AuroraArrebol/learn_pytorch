"""
作者：ZWP
日期：2022.07.25
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
#from P20_nn_optimize import  NetWork
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

dataset=torchvision.datasets.CIFAR10("dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=1,drop_last=True)
print(dataset.classes)

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
net.load_state_dict(torch.load("net_data.pth"))
img=Image.open("OIP-C.jpg")
#img=Image.open("download.jpg")
img_raw=img
transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32,32))
])
img=transform(img)
img=torch.reshape(img,(1,3,32,32))

net.eval()                                  #####important
with torch.no_grad():                       #####important
    output=torch.argmax(net.forward(img))   #####important

print(dataset.classes[output])
img_raw.show()






'''
writer=SummaryWriter("text")
counter=0
for data in dataloader:
    if counter<100:
        img,label=data
        writer.add_images("imgs",img,counter)
        out=net.forward(img)
        out=torch.argmax(out)
        print(f"{counter}:  {label}...{out}")
        counter+=1

writer.close()
'''

