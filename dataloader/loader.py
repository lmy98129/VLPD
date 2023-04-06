from __future__ import division
from distutils.command.config import config
import pickle
import sys
import os
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config import Config
from dataloader import data_augment
from lib.gen_pseudo_mask import ResNet50_CLIP
from .load_data import get_citypersons, get_caltech
# from memory_profiler import profile
import pdb
from torch.nn import functional as F

sys.path.append(os.path.join(sys.path[0], '..'))

def calc_gt_center(size_train, gts, igs, radius=2, stride=4):

    def gaussian(kernel):
        sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
        s = 2*(sigma**2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    scale_map = np.zeros((2, int(size_train[0] / stride), int(size_train[1] / stride)))
    offset_map = np.zeros((3, int(size_train[0] / stride), int(size_train[1] / stride)))
    pos_map = np.zeros((4, int(size_train[0] / stride), int(size_train[1] / stride)))
    pos_map[1, :, :, ] = 1  # channel 1: 1-value mask, ignore area will be set to 0

    if len(igs) > 0:
        igs = igs / stride
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            pos_map[1, y1:y2, x1:x2] = 0

    if len(gts) > 0:
        gts = gts / stride
        for ind in range(len(gts)):
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)

            dx = gaussian(x2-x1)
            dy = gaussian(y2-y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            pos_map[0, y1:y2, x1:x2] = np.maximum(pos_map[0, y1:y2, x1:x2], gau_map)  # gauss map
            pos_map[1, y1:y2, x1:x2] = 1  # 1-mask map (NOTE: pos-neg map)
            pos_map[2, c_y, c_x] = 1  # center map
            pos_map[3, y1:y2, x1:x2] = 1  # box map

            scale_map[0, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
            scale_map[1, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = 1  # 1-mask

            offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5  # height-Y offset
            offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5  # width-X offset
            offset_map[2, c_y, c_x] = 1  # 1-mask

    return pos_map, scale_map, offset_map

class CityPersons(Dataset):
    def __init__(self, path, type, config: Config, preloaded=False, transform=None):

        self.dataset = get_citypersons(root_dir=path, type=type)
        self.dataset_len = len(self.dataset)
        self.type = type

        self.config = config
        self.transform = transform

        self.preloaded = preloaded

        if self.preloaded:
            self.img_cache = []
            for i, data in enumerate(self.dataset):
                self.img_cache.append(cv2.imread(data['filepath']))
                print('%d/%d\r' % (i+1, self.dataset_len),end='')
                sys.stdout.flush()
            print('')
  
    def __getitem__(self, item):
        img_data = self.dataset[item]
        if self.preloaded:
            img = self.img_cache[item]
        else:
            img = cv2.imread(img_data['filepath'])
        if self.type == 'train':
            img_data, x_img = data_augment.augment(self.dataset[item], self.config, img)

            gts = img_data['bboxes'].copy()
            igs = img_data['ignoreareas'].copy()

            y_center, y_height, y_offset = calc_gt_center(self.config.size_train, gts, igs, radius=2, stride=self.config.down)

            x_img = x_img.astype(np.float32)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            x_img = (x_img - self.config.norm_mean) / self.config.norm_std
            x_img = torch.from_numpy(x_img.transpose(2, 0, 1))
            return x_img, [y_center, y_height, y_offset]

        else:
            (h, w) = self.config.size_test
            img = cv2.resize(img, (w, h))
            x_img = img.astype(np.float32)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            x_img = (x_img - self.config.norm_mean) / self.config.norm_std
            x_img = torch.from_numpy(x_img.transpose(2, 0, 1))
            return x_img

    def __len__(self):
        return self.dataset_len

class Caltech(Dataset):
    def __init__(self, path, type, config: Config):
        self.root_dir = path
        self.dataset = get_caltech(root_dir=self.root_dir, type=type)
        self.dataset_len = len(self.dataset)
        self.type = type

        self.config = config
  
    def __getitem__(self, item):
        img_data = self.dataset[item]
        file_path:str = img_data['filepath']
        file_path = os.path.join(self.root_dir, '/'.join(file_path.split('/')[-3:]))
        img = cv2.imread(file_path)

        if self.type in ['train_gt', 'train_nogt']:
            img_data, x_img = data_augment.augment(self.dataset[item], self.config, img)

            gts = img_data['bboxes'].copy()
            igs = img_data['ignoreareas'].copy()

            y_center, y_height, y_offset = calc_gt_center(self.config.size_train, gts, igs, radius=2, stride=self.config.down)

            x_img = x_img.astype(np.float32)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            x_img = (x_img - self.config.norm_mean) / self.config.norm_std
            x_img = torch.from_numpy(x_img.transpose(2, 0, 1))
            return x_img, [y_center, y_height, y_offset]

        else:
            (h, w) = self.config.size_test
            img = cv2.resize(img, (w, h))
            x_img = img.astype(np.float32)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            x_img = (x_img - self.config.norm_mean) / self.config.norm_std
            x_img = torch.from_numpy(x_img.transpose(2, 0, 1))

            return x_img, item

    def __len__(self):
        return self.dataset_len