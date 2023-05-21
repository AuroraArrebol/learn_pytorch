from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.path,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir.spilt('_')[0]
        return img,label

    def __len__(self):
        return len(self.img_path)

ants_dataset=MyData(r"D:\A My Programming Works\python\learn_pytorch\dataset\train","ants_image")
#bees_dataset=MyData(r"D:\A My Programming Works\python\learn_pytorch\dataset\train","bees_image")
#train_dataset=ants_dataset+ants_dataset