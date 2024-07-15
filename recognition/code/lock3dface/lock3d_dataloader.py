# coding=utf-8
import os
import re
import cv2
import json
import torch.utils.data as data
import torchvision.transforms as transforms


class lock3d_trainloader(data.Dataset):
    def __init__(self, root_dir, list_dir, transform=None):
        self.root_dir = root_dir
        self.list_file = list_dir
        self.arr_id, self.arr_join = self.load(list_dir)
        self.transform = transform

    def __getitem__(self, item):
        label = self.arr_join[item]['e_id']
        high_path = os.path.join(self.root_dir, self.arr_join[item]['nu_path'])
        low_path = os.path.join(self.root_dir, self.arr_join[item]['low_path'])
        high_image = cv2.imread(high_path, 1)
        low_image = cv2.imread(low_path, 1)
        high_image = self.transform(high_image)
        low_image = self.transform(low_image)

        return label, high_image, low_image

    def __len__(self):
        return len(self.arr_join)

    def class_num(self):
        return max(self.arr_id) + 1

    def load(self, list_dir):
        arr_join = []
        arr_id = []
        c = []
        new_files = []
        j = 0
        text = os.listdir(list_dir)
        for t in text:
            t = re.sub("\D", "", t)
            c.append(t)
        c_1 = sorted(list(map(int, c)))
        for i in c_1:
            if len(str(i)) < 2:
                i = '00' + str(i)
            elif len(str(i)) < 3:
                i = '0' + str(i)
            else:
                i = str(i)
            new_files.append(str(i) + '.txt')

        for example in new_files:
            j += 1
            e_id = int(example[:3])
            arr_id.append(e_id)
            i_path = os.path.join(list_dir, example)
            with open(i_path, 'r') as file:
                str1 = file.read()
                data = json.loads(str1)
                high_path = data['nu_path']
                low_path_list = data['low_file_path_list']
                for low_path in low_path_list:
                    child_dict = {"e_id": e_id, "nu_path": high_path, "low_path": low_path}
                    arr_join.append(child_dict)
            if j > 339:
                break
        return arr_id, arr_join


class lock3d_testloader(data.Dataset):
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

        return image, label, self.str_list[item][0]

    def __len__(self):
        return len(self.str_list)

    def class_num(self):
        return int(self.classnum)
