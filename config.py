#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:32
# @Author  : LUYAO
# @File    : config.py.py

#########################################
#Training Config
#########################################
workers = 8             # number of Dataloader workers
epochs = 100            # nums of epochs
batch_size = 6       # batch size
learning_rate = 0.001   # initial learning rate
optimizer = 'sgd'       # 'sgd','adam'1
scheduler = 'muilt_step'#'muilt_step'# 'step', 'plateau', 'muilt_step', 'cosine'
use_gpu = True
multi_gpu = False
gpu_ids = '0'
weight_decay = 1e-4
momentum = 0.9


#########################################
#Model Config
#########################################
image_size = (600,600)      # resize image size
input_size = (448,448)      # random crop size
cam_num = 8                 # the number of cam map used
mask_train_th = (0.3,0.6)  # train th
mask_test_th = 0.3          # test th
drop_th = (0.75,0.9)
drop_out = True             # whether dropout?


#########################################
#Dataset/Saving path Config
#########################################
dataset_tag = 'car'        # 'bird',   'dog',  'car',  'aircraft' ,'wilddogs'
class_num = 196           # '200',    '120',  '196',     '100'
checkpoint_path = 'checkpoint/%s' % dataset_tag+'dense161'
model_load_path = 'checkpoint/%s%s/model_best.pth.tar' % (dataset_tag, gpu_ids)
model_name = 'resnet50'        #'resnet50' 'resnet101' 'resnest50''densenet161','densenet169''densenet121'
log_name = ''
resume = '' #''
print_freq = 100

def getDatasetConfig(dataset_name):
    assert dataset_name in ['bird', 'car',
                            'aircraft','dog','wilddogs'], 'No dataset named %s!' % dataset_name
    dataset_dict = {
        'bird': {'train_root': 'data/Bird/images',  # the root path of the train images stored
                 'val_root': 'data/Bird/images',  # the root path of the validate images stored
                                                    # training list file (aranged as filename lable)
                 'train': 'data/bird_train.txt',
                 'val': 'data/bird_test.txt'},  # validate list file
        'car': {'train_root': 'data/Car/cars_train',
                'val_root': 'data/Car/cars_test',
                'train': 'data/car_train.txt',
                'val': 'data/car_test.txt'},
        'aircraft': {'train_root': 'data/Aircraft/images',
                     'val_root': 'data/Aircraft/images',
                     'train': 'data/aircraft_train.txt',
                     'val': 'data/aircraft_test.txt'},
        'dog': {'train_root': 'data/Dog/Images',
                'val_root': 'data/Dog/Images',
                'train': 'data/dog_train.txt',
                'val': 'data/dog_test.txt'},
        'wilddogs': {'train_root': 'data/WildDog',
                'val_root': 'data/WildDog',
                'train': 'data/wild_dogs_train.txt',
                'val': 'data/wild_dogs_test.txt'},
    }
    return dataset_dict[dataset_name]


    




