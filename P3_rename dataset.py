"""
作者：ZWP
日期：2022.07.14
"""
import os
roor_dir=r"dataset/train"
target_dir=r"bees_image"
img_path=os.listdir(os.path.join(roor_dir,target_dir))
label=target_dir.split('_')[0]
out_dir='bees_label'
for i in img_path:
    file_name=i.split('.jpg')[0]
    with open(os.path.join(roor_dir,out_dir,file_name+'.txt'),'w') as f:
        f.write(label)