"""
作者：ZWP
日期：2022.07.14u
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writter=SummaryWriter("logs")
#图片
image_path=r"dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
writter.add_image("test",img_array,2,dataformats='HWC')#图片格式应该是ndarray或Tensor

#y=x
for i in range(100):
    writter.add_scalar("y=2x",2*i,i)

writter.close()
#运行后在终端窗口中输入    tensorboard --logdir=logs
#点击链接即可