"""
作者：ZWP
日期：2022.07.21
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])

train_set=torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",train=False,transform=dataset_transform,download=True)

'''
print(test_set[0])#得到一个元组，第一个元素是PIL图片，第二个元素是其label
print(test_set.classes)#10种label对应的类别
img,label=test_set[0]
print(img)
print(test_set.classes[label])
img.show()
'''
#print(train_set[0])
writer=SummaryWriter("P10_logs")
for i in range(10):
    img,label=test_set[i]
    writer.add_image("test_set",img,i)
writer.close()
