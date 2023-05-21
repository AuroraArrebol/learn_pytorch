"""
作者：ZWP
日期：2022.07.19
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
img=Image.open(r"dataset/train/bees_image/29494643_e3410f0d37.jpg")

#transforms.ToTensor转化为Tensor类型
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#transforms.Normalize归一化
trans_norm=transforms.Normalize([1,2,5],[5,2,1])
img_norm=trans_norm.forward(img_tensor)
print(img_tensor[0][0][0])
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#transforms.Resize尺寸裁剪(可以是PIL格式或者是Tensor格式)参数为元组或者int型
##参数为元组时，元组即为裁剪后尺寸；为int型时 把图片的短边裁剪为int值大小
print(img.size)
trans_resize=transforms.Resize(20)
img_resize=trans_resize.forward(img_tensor)
writer.add_image("Resize",img_resize)

#transforms.Compose(把多种transform操作放在一个类里面)
trans_compose=transforms.Compose([trans_totensor,trans_resize])
img_resize_2=trans_compose(img)
writer.add_image("Resize2",img_resize_2)

#transforms.RandomCrop(随机截取图片的一部分，大小为定值)，参数为元组或者int型
#参数为元组时，元组即为裁剪后尺寸；为int型时 则裁剪为长为int值的正方形
trans_random=transforms.RandomCrop((50,40))
trans_compose_2=transforms.Compose([trans_totensor,trans_random])
for i in range(1,10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop,i)



writer.close()

