import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import io

class Video(Dataset):
    def __init__(self, data_root="./data/CAIN_data/", mode='train'):
        # 根据模式选择数据
        if mode == "train":
            # 构建目录文件所在地址
            train_txt_path = os.path.join(data_root, "train_data.txt")
            with open(train_txt_path, 'r') as file:
                # 逐行读取文件内容并存储在一个列表中
                name_list = file.readlines()
            name_list = [line.strip() for line in name_list]

        elif mode == "test":
            test_txt_path = os.path.join(data_root, "test_data.txt")
            with open(test_txt_path, 'r') as file:
                name_list = file.readlines()
            name_list = [line.strip() for line in name_list]
        
        self.imglist = []
        for l in name_list:
            self.imglist.append(os.path.join(data_root, l))

        print('[%d] images ready to be loaded' % len(self.imglist))


    def __getitem__(self, index):
        imgpaths = self.imglist[index]

        # Load images and ToTensor
        T = transforms.ToTensor()
        imgs = []
        for i in range(3):
            slice_path = os.path.join(imgpaths, str(i) + ".TIF")
            img = T(io.imread(slice_path))
            imgs.append(img)
        
        meta = {'imgpath': imgpaths}
        return imgs, meta

    def __len__(self):
        return len(self.imglist)


def get_loader(mode, data_root, batch_size, shuffle=False, num_workers=0, n_frames=1):
    dataset = Video(data_root,mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
