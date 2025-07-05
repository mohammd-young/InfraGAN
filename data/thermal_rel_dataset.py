import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.image_folder import make_thermal_dataset
from PIL import Image


class ThermalRelDataset(BaseDataset):
    def initialize(self, opt):
        print('ThermalRelDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = make_thermal_dataset(self.dir_AB)
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
    # grab paths
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']
        ann_path = self.AB_paths[index]['annotation_file']
    
        # ─── Load I-channel (grayscale) ───
        A = Image.open(A_path).convert('L')            # ensure single-channel
        A = transforms.ToTensor()(A).float()           # [1, H, W]
    
        # ─── Load thermal image B as grayscale ───
        B = Image.open(B_path).convert('L')
        B = transforms.ToTensor()(B).float()           # [1, H, W]
    
        # ─── Center-crop to fineSize × fineSize ───
        h, w = A.size(1), A.size(2)
        h_offset = max(0, (h - self.opt.fineSize) // 2)
        w_offset = max(0, (w - self.opt.fineSize) // 2)
        A = A[:, h_offset : h_offset + self.opt.fineSize,
                w_offset : w_offset + self.opt.fineSize]
        B = B[:, h_offset : h_offset + self.opt.fineSize,
                w_offset : w_offset + self.opt.fineSize]
    
        # ─── Normalize both channels to mean=0.5, std=0.5 ───
        A = transforms.Normalize([0.5], [0.5])(A)
        B = transforms.Normalize([0.5], [0.5])(B)
    
        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
            'annotation_file': ann_path
        }


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ThermalDataset'
