#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/5 18:45
# @Author  : LUYAO
# @File    : new_attention.py
import torch
import random
import numpy as np

import torch.nn.functional as F
import config


def attention_drop_train(attention_maps, input_image, theta, top1_cam):
    attention_maps = attention_maps.cpu()
    top1_cam = top1_cam.cpu()
    B, N, H, W = input_image.shape
    input_tensor = input_image.cpu()
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    top1_cam = torch.nn.functional.interpolate(top1_cam.detach(), size=(W, H), mode='bilinear')
    ret_imgs = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        mask = attention_map[0]
        if isinstance(theta, tuple):
            threshold = random.uniform(*theta)
        else:
            threshold = theta

        #drop_operator
        top1_f = top1_cam[i]
        top1_f = top1_f[0]
        max_value = top1_f.max()
        top1_f = top1_f.unsqueeze(0)
        
        d = random.uniform(*(config.drop_th))

        th = (d * max_value).expand(1, H * W).view(1, H, W)
        drop = (top1_f < th).float()
      

        itemindex = np.where(mask >= mask.max() * threshold)
        padding_h = int(0.1 * H)
        padding_w = int(0.1 * W)
        height_min = itemindex[0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[1].max() + padding_w
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)

        #####
        # drop.repeat(3,0)
        drop = drop.repeat(3,1,1).unsqueeze(0)

        out_img = out_img*drop
        out_img = out_img.squeeze(0)
        # out_img = image_random_patch(img=out_img)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    return ret_imgs






def attention_crop_test(attention_maps, input_image, theta):
    attention_maps = attention_maps.cpu()
    B, N, W, H = input_image.shape
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    ret_imgs = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        mask = attention_map[0]
        if isinstance(theta, tuple):
            threshold = random.uniform(*theta)
        else:
            threshold = theta
        itemindex = np.where(mask >= mask.max() * threshold)
        padding_h = int(0.1 * H)
        padding_w = int(0.1 * W)
        height_min = itemindex[0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[1].max() + padding_w
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    return ret_imgs


def drop_descrim_area(features,drop_nums):
   
    batch, c, h, w = features.size()#b,1,14,14
    masks = torch.ones_like(features).cuda()
    features = features.view(batch,c,h*w)
    values_top, index = torch.topk(features,drop_nums)#value top 是第drop——num 的值，index是下标
    values = values_top[:,:,drop_nums-1:drop_nums].expand(batch,c,h*w)
    m = features>=values
    m = m.view(batch,c,h,w)
    masks[torch.where(m>0,torch.ones_like(m),torch.zeros_like(m))] = 0.5
    return masks
















