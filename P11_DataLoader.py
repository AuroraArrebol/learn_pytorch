"""
作者：ZWP
日期：2022.07.21
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#测试集
test_data=torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=81,shuffle=True,num_workers=0,drop_last=True)

#测试数据集中第一张图片与target
img,target=test_data[0]
print(img.shape)
print(target)

writer=SummaryWriter("dataLoader_logs")
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,target=data
        #print(imgs.shape)
        #print(target)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step=step+1
writer.close()
