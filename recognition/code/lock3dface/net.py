# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import math
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(block, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channel_out, eps=2e-5, momentum=0.9)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class Led3Dnet_soft(nn.Module):
    def __init__(self):
        super(Led3Dnet_soft, self).__init__()

        self.conv1 = block(3, 32)
        self.conv2 = block(32, 64)
        self.conv3 = block(64, 128)
        self.conv4 = block(128, 256)

        self.conv5 = block(480, 960)
        self.SAV = nn.Conv2d(960, 960, kernel_size=8, stride=1, groups=960)
        self.SAV_var = nn.Sequential(nn.Conv2d(960, 960, kernel_size=8, stride=1, groups=960), nn.Softplus())
        self.dp = nn.Dropout(0.2)

    def sample(self, mean, var, std):
        eps = torch.randn_like(mean)
        if std:
            return mean + var.sqrt() * eps
        else:
            return mean

    def forward(self, x, std=False):
        x1 = self.conv1(x)
        pool1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)

        x2 = self.conv2(pool1)
        pool2 = F.max_pool2d(x2, kernel_size=3, stride=2, padding=1)

        x3 = self.conv3(pool2)
        pool3 = F.max_pool2d(x3, kernel_size=3, stride=2, padding=1)

        x4 = self.conv4(pool3)
        pool4 = F.max_pool2d(x4, kernel_size=3, stride=2, padding=1)

        global_pool1 = F.max_pool2d(x1, kernel_size=33, stride=16, padding=16)
        global_pool2 = F.max_pool2d(x2, kernel_size=17, stride=8, padding=8)
        global_pool3 = F.max_pool2d(x3, kernel_size=9, stride=4, padding=4)
        global_pool4 = pool4

        x = torch.cat((global_pool1, global_pool2, global_pool3, global_pool4), dim=1)
        x = self.conv5(x)

        mean = self.SAV(x)
        mean = self.dp(mean).view(mean.size(0), -1)

        if std:
            var = self.SAV_var(x).view(mean.size(0), -1) 
            sample = self.sample(mean, var, std).view(mean.size(0), -1)
            return x, sample, mean, var
        else:
            return x, mean.view(mean.shape[0], 960)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class con_layer(nn.Module):
    # 原先context_dim=64
    def __init__(self, context_dim=64):
        super(con_layer, self).__init__()
        self.conv = nn.Conv2d(960, context_dim, kernel_size=8, stride=1, groups=context_dim, bias=False)
        self.bn = nn.BatchNorm2d(context_dim)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FC(nn.Module):
    def __init__(self, num_class):
        super(FC, self).__init__()
        self.fc = nn.Linear(960, num_class, bias=False)
        # self.bn = nn.BatchNorm1d(960)

    def forward(self, input):
        # weight = F.normalize(self.fc.weight)
        # return F.linear(F.normalize(input), weight)#
        return  self.fc(input)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)