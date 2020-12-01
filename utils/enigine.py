#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 16:38
# @Author  : LUYAO
# @File    : enigine.py
from utils.util import AverageMeter, accuracy
from utils.cam_generator import image_with_cam
import torch
from tqdm import tqdm
import config
from model.new_attention import attention_crop_test,attention_drop_train
import torch.nn.functional as F


class Engine():###合并之后的
    def __init__(self):
        pass
    def train(self, state, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        print_freq = config.print_freq
        net = state['model']
        criterion = state['criterion']
        optimizer = state['optimizer']
        train_loader = state['train_loader']
        write = state['write']
        net.train()
        pbar = tqdm(total = len(train_loader), unit='batches')
        pbar.set_description('Epoch {}/{}'.format(epoch+1,config.epochs))
        for i, (img, label) in enumerate(train_loader):
            # if config.use_gpu:
            if img.size(0) == 1:
                img = img.repeat(2, 1, 1, 1)
                label = label.repeat(2)
            target = label.cuda()
            input = img.cuda()
            optimizer.zero_grad()

            #net forward
            prob1, cam_top1, M_p= net(input)
            crop_img = attention_drop_train(M_p, input, config.mask_train_th,cam_top1)
            crop_img = crop_img.cuda()
            prob2, cam_top1_2, _ = net(crop_img)

            loss1 = criterion(prob1, target)
            loss2 = criterion(prob2, target)
            loss = (loss1 + loss2)/2

            #net train  accuracy
            prec1, prec5 = accuracy(prob1, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            loss.backward()
            optimizer.step()
            if i % 300 == 0:
                #first image show
                first_pre = image_with_cam(input, cam_top1)

                #second image show
                second_pre = image_with_cam(crop_img, cam_top1_2)

                #可视化mask4
                write.add_images('first_pre', first_pre, 0, dataformats='NCHW')
                write.add_images('second_pre', second_pre, 0, dataformats='NCHW')

            pbar.update()
            # pbar.set_postfix_str(batch_info)
        pbar.close()
        return top1.avg, losses.avg

    def validate(self, state,epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        print_freq = config.print_freq
        net = state['model']
        val_loader = state['val_loader']
        criterion = state['criterion']
        write_val = state['write_val']
        # switch to evaluate mode
        net.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                if input.size(0) == 1:
                    input = input.repeat(2,1,1,1)
                    label = label.repeat(2)

                target = label.cuda()
                input = input.cuda()
                # forward
                prob1, cam_top1, M_p = net(input)
                crop_img = attention_crop_test(M_p, input, config.mask_test_th)
                crop_img = crop_img.cuda()
                prob2, cam_top1_2, _ = net(crop_img)


                loss1 = criterion(prob1, target)
                loss2 = criterion(prob2, target)
                loss = (loss1 + loss2)/2
                out = (F.softmax(prob1,dim=-1)+F.softmax(prob2,dim=-1))/2

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))


                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), loss=losses,
                        top1=top1, top5=top5))
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
        return top1.avg, losses.avg

    def test(self,val_loader, net, criterion):
        top1 = AverageMeter()
        top5 = AverageMeter()
        print_freq = 100
        # switch to evaluate mode
        net.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                target = label.cuda()
                input = input.cuda()
                # forwardclea
                prob1, cam_top1, M_p = net(input)
                crop_img = attention_crop_test(M_p, input, config.mask_test_th)
                crop_img = crop_img.cuda()
                prob2, cam_top1_2, _ = net(crop_img)


                # measure accuracy and record loss
                out = (F.softmax(prob1, dim=-1) + F.softmax(prob2, dim=-1)) / 2
                prec1, prec5 = accuracy(out, target, topk=(1, 5))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader),
                        top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
        return top1.avg, top5.avg



