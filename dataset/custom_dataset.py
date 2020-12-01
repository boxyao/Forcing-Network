#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 16:02
# @Author  : LUYAO
# @File    : custom_dataset.py

from PIL import Image
from torch.utils.data import Dataset
from utils.util import get_transform
import os

class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = get_transform(training)
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            # self.datas = f.readlines()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                # label = int(label)
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list) + 1

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

    
