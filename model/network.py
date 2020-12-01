#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:53
# @Author  : LUYAO
# @File    : network.py

import torch

import config
import torch.nn as nn
from model.new_attention import drop_descrim_area
from torchvision import models


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}

class Model(nn.Module):
    def __init__(self, mode_name='resnet50', class_num=200):
        super(Model, self).__init__()
        if mode_name == "resnet50":
            network = models.resnet50(pretrained=False)
            layers = list(network.children())[:-2]
            final_channel = 2048
        elif mode_name == "resnet101":
            network = models.resnet101(pretrained=False)
            layers = list(network.children())[:-2]
            final_channel = 2048
        elif mode_name == 'densenet121':
            network = models.densenet121(pretrained=False)
            layers = list(network.children())[:-1]
            final_channel = 1024
        elif mode_name == 'densenet161':
            network = models.densenet161(pretrained=False)
            layers = list(network.children())[:-1]
            final_channel = 2208
        elif mode_name == 'densenet169':
            network = models.densenet169(pretrained=False)
            layers = list(network.children())[:-1]
            final_channel = 1664
        else:
            pass
        #Feature Extracting  Module
        self.feature_extract_layer = nn.Sequential(*layers)

        # Original branch
        self.original_branch = nn.Conv2d(final_channel, class_num, 1, 1, 0)
        self.original_pool = nn.AdaptiveAvgPool2d(1)

        # Foring branch
        self.forcing_branch = nn.Conv2d(final_channel, class_num, 1, 1, 0)
        self.forcing_pool = nn.AdaptiveAvgPool2d(1)

        # final
        self.final_pool = nn.AdaptiveAvgPool2d(1)

        self.init_parameter()


    def init_parameter(self):
        # initial all parameters
        print("Initial  parameters....")
        nn.init.xavier_normal_(self.original_branch.weight.data)
        nn.init.xavier_normal_(self.forcing_branch.weight.data)
        if self.original_branch.bias is not None:
            nn.init.constant_(self.original_branch.bias.data, 0)
        if self.forcing_branch.bias is not None:
            nn.init.constant_(self.forcing_branch.bias.data, 0)

    def get_top_cam(self, cam, topn_pos):
        """
        按照position 选择每个batch的前k个cam
        """
        N,C,H,W = cam.size()
        out = []
        for i in range(N):
            map_i = torch.index_select(cam[i],0,topn_pos[i])
            out.append(map_i)
        return torch.stack(out,0)


    def forward(self, x):

        #feature extracting
        x = self.feature_extract_layer(x)

        #original branch
        M_1 = self.original_branch(x)

        #suppressive mask generate
        if self.training:
            with torch.no_grad():
                M_1_prob = self.original_pool(M_1)
                M_1_prob = M_1_prob.view(M_1_prob.size(0), M_1_prob.size(1))
                _, indicate1 = torch.topk(M_1_prob, 1)
                M_1_p = self.get_top_cam(M_1, indicate1)
                B = drop_descrim_area(M_1_p, 4)
            G = B * x
        else:
            G = x

        # forcing branch
        M_2 = self.forcing_branch(G)

        # finla_result
        M = M_1 + M_2
        prob = self.final_pool(M)
        prob = prob.view(prob.size(0), prob.size(1))


        # crop_cam_Generate
        with torch.no_grad():
            #top_8 cam
            _, indicate8 = torch.topk(prob, config.cam_num)
            top_8 = self.get_top_cam(M, indicate8)
            M_p = torch.sum(top_8, 1) / top_8.size(1)
            M_p = M_p.unsqueeze(1)

            #top_1 cam
            _, indicate1 = torch.topk(prob, 1)
            cam_top1 = self.get_top_cam(M, indicate1)


        return prob, cam_top1, M_p

def init_model(pretrained=False, model_name='resnet50', class_num=200):
    model = Model(model_name, class_num)
    # load pretrained model
    if pretrained:
        print(" Loading pretrained weight from %s" % model_urls[model_name])
        pretrained_dict = load_state_dict_from_url(model_urls[model_name])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and model_dict[k].size() == v.size()}
        # print(pretrained_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    net = init_model(pretrained=False, model_name="resnet50", class_num=200)
    net.eval()
    input = torch.ones(12, 3, 448, 448)
    out = net(input, parts=2)
    print(out.size())







