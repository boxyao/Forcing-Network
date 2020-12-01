#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:52
# @Author  : LUYAO
# @File    : visual_test.py


import config
from utils.enigine import  Engine
from torch.utils.data import DataLoader
from dataset.custom_dataset import CustomDataset
from config import getDatasetConfig
import torch
from model.network import init_model
def test():
    engine = Engine()
    #define test dataset
    data_config = getDatasetConfig(config.dataset_tag)
    test_dataset = CustomDataset(data_config['val'], data_config['val_root'], False)
    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.workers,
                            pin_memory=True)
    # define model

    net = init_model(pretrained=True, model_name=config.model_name, class_num=config.class_num)

    # load checkpoint
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    ckpt = torch.load(config.model_load_path)
    net.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    prec1, prec5 = engine.test(test_loader, net, criterion)

if __name__ == '__main__':
    test()

    
