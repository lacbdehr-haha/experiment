# coding=utf-8
"""
original:extend_multi_dim
"""
import torch.utils.data as data
import cv2
import torch
import torchvision
import numpy as np
from glob import glob


def depth2normal(depth):
    # 输入单通道深度图像，输出三通道法向图像
    w, h = depth.shape

    dy, dx = np.gradient(depth)
    dz = np.ones((w, h))
    dl = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx = dx / dl * 0.5 + 0.5
    dy = dy / dl * 0.5 + 0.5
    dz = dz / dl * 0.5 + 0.5
    normal = np.concatenate([dy[np.newaxis, :, :], dx[np.newaxis, :, :], dz[np.newaxis, :, :]], axis=0)
    normal = np.array(normal * 255).astype(np.uint8)
    normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)
    return normal


class SCU_loader(data.Dataset):
    def __init__(self, root_dir, list_dir, transform=None):

        self.root_dir = root_dir
        self.list_file = list_dir
        with open(self.list_file, 'r') as fp:
            content = fp.readlines()
            self.str_list = [s.rstrip().split() for s in content]

        idlist = set()
        for i in self.str_list:
            idlist.add(i[1])
        self.classnum = len(idlist)
        self.transform = transform

    def __getitem__(self, item):
        path = self.root_dir + self.str_list[item][0]
        label = int(self.str_list[item][1])
        image = cv2.imread(path, 1)
        # image = depth2normal(image)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.str_list)

    def class_num(self):
        return int(self.classnum)


class SCU_testloader(data.Dataset):
    def __init__(self, root_dir, list_dir, transform=None):

        self.root_dir = root_dir
        self.list_file = list_dir
        with open(self.list_file, 'r') as fp:
            content = fp.readlines()
            self.str_list = [s.rstrip().split() for s in content]

        idlist = set()
        for i in self.str_list:
            idlist.add(i[1])
        self.classnum = len(idlist)
        self.transform = transform

    def __getitem__(self, item):
        path = self.root_dir + self.str_list[item][0]
        label = int(self.str_list[item][1])
        image = cv2.imread(path, 1)
        image = self.transform(image)

        return image, label, self.str_list[item][0]

    def __len__(self):
        return len(self.str_list)

    def class_num(self):
        return int(self.classnum)