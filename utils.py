#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
References:
    https://evalai-forum.cloudcv.org/t/fyi-on-semantic-segmentation/180
    https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py
    https://discuss.pytorch.org/t/using-cross-entropy-loss-with-semantic-segmentation-model/31988
    https://github.com/LIVIAETS/surface-loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import cv2
import os

from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.ndimage import distance_transform_edt as distance


class FocalLoss2d(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss((1 - nn.Softmax2d()(outputs)).pow(self.gamma) * torch.log(nn.Softmax2d()(outputs)), targets)


###https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py
# https://discuss.pytorch.org/t/using-cross-entropy-loss-with-semantic-segmentation-model/31988
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class SurfaceLoss(nn.Module):
    # 作者：Rakshit Kothari
    def __init__(self, epsilon=1e-5, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []

    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2) * distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2)  # 每个通道的像素的平均值
        score = torch.mean(score, dim=1)  # 通道之间的平均值
        return score


class GeneralizedDiceLoss(nn.Module):
    # 作者：Rakshit Kothari
    # 输入: (B, C, ...)
    # 目标: (B, C, ...)
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, ip, target):

        # 快速转换为 one-hot。对于将来的版本，使用 functional
        #        Label = (np.arange(4) == target.cpu().numpy()[..., None]).astype(np.uint8)
        Label = (np.arange(2) == target.cpu().numpy()[..., None]).astype(np.uint8)  # 将 4 改为 2
        target = torch.from_numpy(np.rollaxis(Label, 3, start=1)).cuda()

        assert ip.shape == target.shape
        ip = self.norm(ip)

        # 为多维数据展平
        ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(torch.float32)

        numerator = ip * target
        denominator = ip + target

        class_weights = 1. / (torch.sum(target, dim=2) ** 2).clamp(min=self.epsilon)

        A = class_weights * torch.sum(numerator, dim=2)
        B = class_weights * torch.sum(denominator, dim=2)

        dice_metric = 2. * torch.sum(A, dim=1) / torch.sum(B, dim=1)
        if self.reduction:
            return torch.mean(1. - dice_metric.clamp(min=self.epsilon))
        else:
            return 1. - dice_metric.clamp(min=self.epsilon)


# https://github.com/LIVIAETS/surface-loss
def one_hot2dist(posmask):
    # 输入: Mask。将被转换为布尔型。
    # 作者：Rakshit Kothari
    assert len(posmask.shape) == 2
    h, w = posmask.shape
    res = np.zeros_like(posmask)
    posmask = posmask.astype(np.bool)
    mxDist = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res / mxDist


def mIoU(predictions, targets, info=False):  ###每类平均准确率
    unique_labels = np.unique(targets)
    num_unique_labels = len(unique_labels)
    ious = []
    for index in range(num_unique_labels):
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection.numpy()) / np.sum(union.numpy())
        ious.append(iou_score)
    if info:
        print("每类 mIOU: ", ious)
    return np.mean(ious)


# https://evalai-forum.cloudcv.org/t/fyi-on-semantic-segmentation/180
# GA: 全局像素准确度
# CA: 不同类的平均类准确度
#
# Back: 背景（眼部周围区域的非眼部分）
# Sclera: 巩膜
# Iris: 虹膜
# Pupil: 瞳孔
# Precision: 使用 sklearn.metrics.precision_score(pred, gt, ‘weighted’) 计算
# Recall: 使用 sklearn.metrics.recall_score(pred, gt, ‘weighted’) 计算
# F1: 使用 sklearn.metrics.f1_score(pred, gt, ‘weighted’) 计算
# IoU: 使用下面的函数计算
def compute_mean_iou(flat_pred, flat_label, info=False):
    '''
    计算所有类别的平均交集联合(IOU)
    :param flat_pred: 扁平化的预测矩阵
    :param flat_label: 扁平化的标签矩阵
    :return: 平均 IOU
    '''
    unique_labels = np.unique(flat_label)
    num_unique_labels = len(unique_labels)

    Intersect = np.zeros(num_unique_labels)
    Union = np.zeros(num_unique_labels)
    precision = np.zeros(num_unique_labels)
    recall = np.zeros(num_unique_labels)
    f1 = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = flat_pred == val
        label_i = flat_label == val

        if info:
            precision[index] = precision_score(pred_i, label_i, 'weighted')
            recall[index] = recall_score(pred_i, label_i, 'weighted')
            f1[index] = f1_score(pred_i, label_i, 'weighted')

        Intersect[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        Union[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    if info:
        print("每类 mIOU: ", Intersect / Union)
        print("每类 precision: ", precision)
        print("每类 recall: ", recall)
        print("每类 f1: ", f1)
    mean_iou = np.mean(Intersect / Union)
    return mean_iou


def total_metric(nparams, miou):
    S = nparams * 4.0 / (1024 * 1024)
    total = min(1, 1.0 / S) + miou
    return total * 0.5


def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs, h, w)  # bs x h x w
    return indices


class Logger():
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

    def write_summary(self, msg):
        self.log_file.write(msg)
        self.log_file.write('\n')
        self.log_file.flush()
        print(msg)

