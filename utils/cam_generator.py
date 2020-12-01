#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:53
# @Author  : LUYAO
# @File    : cam_generator.py
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms


def image_with_mask(img,masks):
    """
    :param img: [N，3，H，W]
    :param mask: [N, 1, h, w]由0，1组成的mmask
    :return: 原图上面加mask
    """
    _, _, H, W = img.size()
    _, part, h, w = masks.size()

    var = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3,1).expand(3,H*W).view(3,H,W)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3,1).expand(3,H*W).view(3,H,W)

    out = []
    masks = torch.nn.functional.interpolate(masks,(H,W))
    for i in range(4):
        ori_img = img[i]
        ori_img = torch.clamp(ori_img * std + var, 0, 1)
        img_i = ori_img*masks[i]
        out.append(img_i)
    return torch.stack(out, 0 )

def image_with_cam(image, cams):
    """
    :param image: 用于训练的原图，是经过标准化之后的数值[N,3,H,W]
    :param masks: CAM 图 [N,1,h2,w2]
    :return: 原图上面加上cam图
    """
    batch, _, image_h, image_w = image.size()
    _, _, masks_h, masks_w = cams.size()
    cams = cams.detach().cpu().numpy()
    tf = transforms.Compose(
        [transforms.ToTensor(),  # turn a PIL.Image [0,255] shape(H,W,C) into [0,1.0] shape(C,H,W) torch.FloatTensor
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    var = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1).expand(3, image_h * image_w).view(3, image_h, image_w)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1).expand(3, image_h * image_w).view(3, image_h, image_w)
    out = []
    # mask还回标准化
    for i in range(batch):
        cam_i = cams[i]
        cam_i = cam_i.reshape(masks_h, masks_w)
        cam_i = cam_i - np.min(cam_i)
        cam_i = cam_i / np.max(cam_i)
        cam_i = np.uint8(255 * cam_i)
        heatmap = cv2.applyColorMap(cv2.resize(cam_i, (image_h, image_w)), 2)
        #这里从cv2 的格式是BGR 这里必须改变为RGB模式
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        ##显示图片
        # cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        # cv2.imshow("image",heatmap)
        # key = cv2.waitKey()
        # if key == ord('s'):
        #     cv2.imwrite("./img",heatmap)
        # else:
        #     cv2.destroyWindow("image")
        heatmap = tf(heatmap).cuda()
        img_and_cam = heatmap * 0.3 + image[i] * 0.5
        img_and_cam = torch.clamp(img_and_cam * std + var, 0, 1)
        out.append(img_and_cam)
    out = torch.stack(out, 0)
    return out[:1]

def image_plot(img):
    """
    :param img: [N，3，H，W]
    :param mask: [N, 3, h, w]
    :return: mask是crop img的部分 resize之后的图像
    """
    _, _, H, W = img.size()

    var = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1).expand(3, H * W).view(3, H, W)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1).expand(3, H * W).view(3, H, W)
    out = []
    for i in range(1):
        ori_img = img[i]
        ori_img = torch.clamp(ori_img*std+var,0,1)
        out.append(ori_img)
    out = torch.stack(out,0)
    return out



if __name__ == '__main__':
    image = torch.tensor([[1,1,1],[1,1,1],[1,1,1]])
    image = image.expand((3,3,3))
    image = image.view(1,3,3,3)
    masks = torch.tensor([[9,2,3],[4,5,6],[7,8,9]])
    masks = masks.view(1,1,3,3)
    image_with_cam(image, masks)




