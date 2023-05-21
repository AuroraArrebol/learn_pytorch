"""
作者：ZWP
日期：2022.07.25
"""
import torchvision

pretrained_net=torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)