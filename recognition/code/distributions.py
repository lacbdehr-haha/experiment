# coding=utf-8
from __future__ import print_function
import torch
import torch.utils.data
from torch.autograd import Variable
import math
import numpy as np

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.-1e-5

PI = Variable(torch.FloatTensor([math.pi]))     # 转换成32位浮点tensor
PI.requires_grad = False
if torch.cuda.is_available():
    PI = PI.cuda()

# N(x | mu, var) = 1/sqrt{2pi var} exp[-1/(2 var) (x-mean)(x-mean)]
# log N(x| mu, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    """
    对数正态分布   VAE(计算熵)
    reciprocal():取导数
    """
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    # log_norm = -0.5 * (log_var + 1)
    # x1 = -0.5 * (log_var)
    # x2 = -0.5 * ((x - mean) * (x - mean) * log_var.exp().reciprocal())

    if reduce:
        if average:
            # 对输入的tensor数据的某一维度求均值
            return torch.mean(log_norm, dim)
        else:
            # 对输入的tensor数据的某一维度求和（指定维度求和最常见的应用是求均值）
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    """对数正态分布归一化"""
    log_norm = -(x - mean) * (x - mean)
    log_norm *= torch.reciprocal(2.* log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * torch.log(2. * PI)

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    """对数正态分布标准化"""
    log_norm = -0.5 * x * x - 0.5 * (np.log(np.pi * 2))

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_bernoulli(x, mean, average=False, reduce=True, dim=None):
    # 对数伯努利分布
    # 将 mean 限制到 min 和 max 的范围内
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if reduce:
        if average:
            return torch.mean(log_bern, dim)
        else:
            return torch.sum(log_bern, dim)
    else:
        return log_bern


