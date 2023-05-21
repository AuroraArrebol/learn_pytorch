"""
作者：ZWP
日期：2022.07.26
"""
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

#准备数据集
from torch.utils.tensorboard import SummaryWriter

train_data=torchvision.datasets.CIFAR10(root="dataset_CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
print(f"训练集大小:{len(train_data)}")
print(f"测试集大小:{len(test_data)}")
#加载数据
train_dataloader=DataLoader(train_data,batch_size=64,drop_last=True)
test_dataloader=DataLoader(test_data,batch_size=64,drop_last=True)
#引入网络
from CIFAR10_net_model import NetWork
net=NetWork()
net=net.cuda()
net.load_state_dict(torch.load("net_data.pth"))
#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
#优化器
learning_rate=0.001
optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate)

#设置网络的一些参数
total_train_step=0  #训练次数
total_test_step=0   #测试次数
epoch=10             #训练的轮数

#添加tensorboard
writer=SummaryWriter("CIFAR10_train_logs")
start_time=time.time()
for i in range(epoch):
    print(f"-----从第{i+1}轮训练开始-----")

    #训练
    net.train()
    for data in train_dataloader:
        input,label=data
        input = input.cuda()
        label = label.cuda()
        output=net.forward(input)
        loss=loss_fn(output,label)

        optimizer.zero_grad()     #梯度清零
        loss.backward()           #反向传播
        optimizer.step()          #优化参数

        total_train_step+=1
        if total_train_step%100==0:
            end_time = time.time()
            print(end_time-start_time)
            print(f"训练次数：{total_train_step},loss={loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试
    net.eval()
    total_test_loss=0
    total_acurrate_num=0
    with torch.no_grad():
        for data in test_dataloader:
            input, label = data
            input = input.cuda()
            label = label.cuda()
            output = net.forward(input)
            loss=loss_fn(output,label)
            total_test_loss+=loss
            acurrate_num=(output.argmax(1)==label).sum()
            total_acurrate_num+=acurrate_num
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的Acurracy: {total_acurrate_num/len(test_data)}")
    writer.add_scalar("test_loss", total_test_loss.item(), i)
    writer.add_scalar("test_acurracy", total_acurrate_num/len(test_data), i)
    #保存
    torch.save(net.state_dict(),"net_data.pth")
writer.close()

