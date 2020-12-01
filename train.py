#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:52
# @Author  : LUYAO
# @File    : train.py
import config
from utils.enigine import  Engine
from utils.util import get_lr, save_checkpoint,set_seed,config_print
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.custom_dataset import CustomDataset
from  torchstat import stat

from config import getDatasetConfig
import torch
from model.network import init_model
import os
import random
import numpy as np

GLOBAL_SEED = np.random.randint(1,1000,size=1)[0]

GLOBAL_SEED = 555

def _init_fn(worker_id):
    set_seed(GLOBAL_SEED+worker_id)
def train():
    config_print()
    print("SEED : {}".format(GLOBAL_SEED))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    set_seed(GLOBAL_SEED)
    best_prec1 = 0.
    write_log = 'logs/%s' % config.dataset_tag+config.gpu_ids
    write_val_log = 'logs/val%s' % config.dataset_tag+config.gpu_ids
    write = SummaryWriter(log_dir=write_log)
    write_val = SummaryWriter(log_dir=write_val_log)
    data_config = getDatasetConfig(config.dataset_tag)

    #load dataset
    train_dataset = CustomDataset(data_config['train'],data_config['train_root'],True)#txt.file,train_root_dir,is_traning
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.workers,
                              pin_memory=True,
                              worker_init_fn=_init_fn)
    val_dataset = CustomDataset(data_config['val'],data_config['val_root'],False)
    val_loader = DataLoader(val_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.workers,
                              pin_memory=True)#,worker_init_fn=_init_fn)

    print('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(
        dataset_name=config.dataset_tag,
        train_num=len(train_dataset),
        val_num=len(val_dataset)))

    # define model

    net = init_model(pretrained=True, model_name=config.model_name, class_num=config.class_num)


    # gup config
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    # define potimizer
    assert config.optimizer in ['sgd', 'adam'], 'optim name not found!'
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # define learning scheduler
    assert config.scheduler in ['plateau',
                                'step',
                                'muilt_step',
                                'cosine'], 'scheduler not supported!!!'
    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
    elif config.scheduler == 'muilt_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 100], gamma=0.1)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()



    if use_gpu:
        criterion = criterion.cuda()
        # train val parameters dict
    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'config': config, 'optimizer': optimizer, 'write': write, 'write_val':write_val}
    # define resume
    start_epoch = 0
    if config.resume:
        ckpt = torch.load(config.resume)
        net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        best_prec1 = ckpt['best_prec1']
        optimizer.load_state_dict(ckpt['optimizer'])

        # train and val
    engine = Engine()
    for e in range(start_epoch, config.epochs+1):
        if config.scheduler in ['step', 'muilt_step']:
            scheduler.step()
        lr_train = get_lr(optimizer)
        print("Start epoch %d ==========,lr=%f" % (e, lr_train))
        train_prec, train_loss= engine.train(state, e)
        prec1, val_loss = engine.validate(state,e)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, config.checkpoint_path)
        write.add_scalars("Accurancy", {'train': train_prec, 'val': prec1}, e)
        write.add_scalars("Loss", {'train': train_loss, 'val': val_loss}, e)
        if config.scheduler == 'plateau':
            scheduler.step(val_loss)

if __name__ == '__main__':

    train()














    
