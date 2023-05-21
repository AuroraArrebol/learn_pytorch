"""
作者：ZWP
日期：2022.07.14
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

#tensor数据类型
#通过transforms.ToTensor看两个问题：
#  1.transforms应该如何使用

# 绝对路径：D:\A My Programming Works\python\learn_pytorch\dataset\train\ants_image\0013035.jpg
# 相对路径：dataset\train\ants_image\0013035.jpg
img_path=r"dataset\train\ants_image\0013035.jpg"
tensor_trans=transforms.ToTensor()#可以把PIL、ndarray格式的图片转换成Tensor类型图片
writer=SummaryWriter("logs")
#用Image读图，读到的是PIL格式的图
img=Image.open(img_path)
print(img)
tensor_img=tensor_trans(img)
print(tensor_img)
writer.add_image("TensorImg",tensor_img)

#用opencv读图,读到的是numpy格式的图
cv_img=cv2.imread(img_path)
print(cv_img)
tensor_img=tensor_trans(cv_img)
print(tensor_img)

#  2.tensor的数据类型有什么特殊之处
#   特殊之处-》tensor数据类型便于深度学习的训练
writer.close()