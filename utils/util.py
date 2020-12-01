#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:54
# @Author  : LUYAO
# @File    : util.py
import torch
import os
import shutil
import torchvision.transforms as transforms
from PIL import Image
import config
import random
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED']  = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def accuracy(output, target, topk=(1,)):
    """
    computes the precision@k

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def save_checkpoint(state, is_best, path='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s==" %
              os.path.join(path, 'model_best.pth.tar'))

def get_transform(traning = True):
    if traning:
        return  transforms.Compose([
        transforms.Resize(size=config.image_size, interpolation=Image.BILINEAR),
        transforms.RandomCrop(size=config.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    else:
        return transforms.Compose([
        transforms.Resize(size=config.image_size,interpolation=Image.BILINEAR),
        transforms.CenterCrop(size=config.input_size),
        transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def image_plot(img,masks):
    """
    :param img: [N，3，H，W]
    :param mask: [N, 3, h, w]
    :return: mask是crop img的部分 resize之后的图像
    """
    _, _, H, W = img.size()
    _, part, h, w = masks.size()

    var = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1).expand(3, H * W).view(3, H, W)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1).expand(3, H * W).view(3, H, W)

    out = []
    out2 = []

    for i in range(4):
        ori_img = img[i]
        ori_img = torch.clamp(ori_img*std+var,0,1)
        out.append(ori_img)

        crop_img = masks[i]
        ori_img = torch.clamp(crop_img*std+var,0,1)
        out2.append(ori_img)
    out = torch.stack(out,0)
    out2 = torch.stack(out2,0)
    return out, out2
def config_print():
    print("Batch_size : {}".format(config.batch_size))
    print("Epochs : {}".format(config.epochs))
    print("CAM_num : {}".format(config.cam_num))
    print("Model_name :{}".format(config.model_name))
    print("Gpu_is : {}".format(config.gpu_ids))
    print("Drop_th : {}".format(config.drop_th))
    print("Dataset_tag : {}".format(config.dataset_tag))
    print("Checkpoint_path : {}".format(config.checkpoint_path))
    print("model_load_path : {}".format(config.model_load_path))
    print("Optimizer : {}".format(config.optimizer))
    print("scheduler: {}".format(config.scheduler))



if __name__ == '__main__':
    img = torch.ones((12,3,448,448))
    masks = torch.ones((12, 1, 14, 14))
    config_print()



