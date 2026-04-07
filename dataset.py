import os
import random
import numpy as np
import torch
from skimage import io
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

from utils import normalization_utils as norm_utils

ST_NUM_CLASSES = 7
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
ST_COLORMAP = [
    [255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], 
    [0, 255, 0], [128, 0, 0], [255, 0, 0]
]

LANDSAT_NUM_CLASSES = 5
LANDSAT_CLASSES = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
LANDSAT_COLORMAP = [
    [255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]
]

def build_colormap_lookup(colormap):
    lookup = np.zeros(256 ** 3, dtype=np.uint8)
    for i, cm in enumerate(colormap):
        idx = (cm[0] * 256 + cm[1]) * 256 + cm[2]
        lookup[idx] = i
    return lookup

st_colormap2label = build_colormap_lookup(ST_COLORMAP)
landsat_colormap2label = build_colormap_lookup(LANDSAT_COLORMAP)

def Color2Index(ColorLabel, lookup_table, num_classes):
    if len(ColorLabel.shape) == 2: 
        return ColorLabel
        
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    
    IndexMap = lookup_table[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    
    return IndexMap.astype(np.int64)

class SCDDataset(data.Dataset):
    def __init__(self, root, mode, dataset_name='SECOND', patch_mode=True, random_flip=True, random_swap=True):
        self.root = root
        self.mode = mode
        self.dataset_name = dataset_name
        self.patch_mode = patch_mode
        self.random_flip = random_flip if mode == 'train' else False
        self.random_swap = random_swap if mode == 'train' else False
        
        if 'second' in dataset_name.lower():
            self.lookup_table = st_colormap2label
            self.num_classes = ST_NUM_CLASSES
            self.class_names = ST_CLASSES
        elif 'landsat' in dataset_name.lower():
            self.lookup_table = landsat_colormap2label
            self.num_classes = LANDSAT_NUM_CLASSES
            self.class_names = LANDSAT_CLASSES
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'SECOND' or 'LandsatSCD'.")

        base_path = os.path.join(root, mode)
        
        self.dir_A = os.path.join(base_path, 'A')
        self.dir_B = os.path.join(base_path, 'B')
        self.dir_sem1 = os.path.join(base_path, 'labelA_rgb')
        self.dir_sem2 = os.path.join(base_path, 'labelB_rgb')
        self.dir_bcd = os.path.join(base_path, 'label_bcd')

        self.file_list = self._get_valid_file_list()
        print(f"[{dataset_name}] {mode} set loaded. Found {len(self.file_list)} samples (Patch Mode: {patch_mode}).")

        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))

    def _get_valid_file_list(self):
        files = [f for f in os.listdir(self.dir_A) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
        valid_items = []
        for f in files:
            if self.patch_mode:
                for i in range(4): valid_items.append((f, i))
            else:
                valid_items.append((f, -1))
        return valid_items

    def _get_patch(self, img, patch_idx, crop_size=256):
        if patch_idx == -1: return img
        if patch_idx == 0: return TF.crop(img, 0, 0, crop_size, crop_size)
        elif patch_idx == 1: return TF.crop(img, 0, crop_size, crop_size, crop_size)
        elif patch_idx == 2: return TF.crop(img, crop_size, 0, crop_size, crop_size)
        elif patch_idx == 3: return TF.crop(img, crop_size, crop_size, crop_size, crop_size)
        else: return img

    def _sync_transform(self, img_A, img_B, sem1, sem2, bcd):
        # [SOTA FIX 3] Re-introduced Scale Invariance safely.
        # Instead of destructive RandomResizedCrop, we use Affine scaling.
        # This prevents jagged edges on the masks while still forcing the model 
        # to learn large and small features robustly.
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            # Affine requires center and angle, angle=0 to isolate scaling
            img_A = TF.affine(img_A, angle=0, translate=(0,0), scale=scale_factor, shear=0, interpolation=transforms.InterpolationMode.BILINEAR)
            img_B = TF.affine(img_B, angle=0, translate=(0,0), scale=scale_factor, shear=0, interpolation=transforms.InterpolationMode.BILINEAR)
            sem1 = TF.affine(sem1, angle=0, translate=(0,0), scale=scale_factor, shear=0, interpolation=transforms.InterpolationMode.NEAREST)
            sem2 = TF.affine(sem2, angle=0, translate=(0,0), scale=scale_factor, shear=0, interpolation=transforms.InterpolationMode.NEAREST)
            bcd = TF.affine(bcd, angle=0, translate=(0,0), scale=scale_factor, shear=0, interpolation=transforms.InterpolationMode.NEAREST)

        # 1. Flips
        if self.random_flip:
            if random.random() > 0.5:
                img_A = TF.hflip(img_A); img_B = TF.hflip(img_B)
                sem1 = TF.hflip(sem1); sem2 = TF.hflip(sem2); bcd = TF.hflip(bcd)
            if random.random() > 0.5:
                img_A = TF.vflip(img_A); img_B = TF.vflip(img_B)
                sem1 = TF.vflip(sem1); sem2 = TF.vflip(sem2); bcd = TF.vflip(bcd)
            
            # Rotations
            if random.random() > 0.5:
                angle = random.choice([0, 90, 180, 270])
                if angle > 0:
                    img_A = TF.rotate(img_A, angle); img_B = TF.rotate(img_B, angle)
                    sem1 = TF.rotate(sem1, angle); sem2 = TF.rotate(sem2, angle); bcd = TF.rotate(bcd, angle)

        # 2. Color & Blur
        # if random.random() < 0.8: img_A = self.color_jitter(img_A)
        # if random.random() < 0.8: img_B = self.color_jitter(img_B)
        # if random.random() < 0.5: img_A = self.gaussian_blur(img_A)
        # if random.random() < 0.5: img_B = self.gaussian_blur(img_B)
            
        return img_A, img_B, sem1, sem2, bcd

    def __getitem__(self, idx):
        filename, patch_idx = self.file_list[idx]
        
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)
        img_A_np = io.imread(path_A)
        img_B_np = io.imread(path_B)

        path_sem1 = os.path.join(self.dir_sem1, filename)
        path_sem2 = os.path.join(self.dir_sem2, filename)
        path_bcd = os.path.join(self.dir_bcd, filename)
        
        label_sem1 = io.imread(path_sem1)
        label_sem2 = io.imread(path_sem2)
        label_bcd = io.imread(path_bcd)

        label_sem1 = Color2Index(label_sem1, self.lookup_table, self.num_classes)
        label_sem2 = Color2Index(label_sem2, self.lookup_table, self.num_classes)
        
        if len(label_bcd.shape) == 3: label_bcd = label_bcd[:, :, 0]
        label_bcd = (label_bcd > 0).astype(np.uint8)

        p_img_A = Image.fromarray(img_A_np)
        p_img_B = Image.fromarray(img_B_np)
        p_sem1 = Image.fromarray(label_sem1.astype(np.uint8)) 
        p_sem2 = Image.fromarray(label_sem2.astype(np.uint8))
        p_bcd = Image.fromarray(label_bcd)

        if self.patch_mode:
            p_img_A = self._get_patch(p_img_A, patch_idx)
            p_img_B = self._get_patch(p_img_B, patch_idx)
            p_sem1 = self._get_patch(p_sem1, patch_idx)
            p_sem2 = self._get_patch(p_sem2, patch_idx)
            p_bcd = self._get_patch(p_bcd, patch_idx)

        if self.mode == 'train':
            p_img_A, p_img_B, p_sem1, p_sem2, p_bcd = self._sync_transform(p_img_A, p_img_B, p_sem1, p_sem2, p_bcd)
            if self.random_swap and random.random() > 0.5:
                p_img_A, p_img_B = p_img_B, p_img_A
                p_sem1, p_sem2 = p_sem2, p_sem1

        img_A_np = np.array(p_img_A)
        img_B_np = np.array(p_img_B)
        label_sem1 = np.array(p_sem1).astype(np.int64)
        label_sem2 = np.array(p_sem2).astype(np.int64)
        label_bcd = np.array(p_bcd).astype(np.int64)

        img_A = norm_utils.normalize_image(img_A_np, 'A', self.dataset_name)
        img_B = norm_utils.normalize_image(img_B_np, 'B', self.dataset_name)

        t_img_A = TF.to_tensor(img_A).float()
        t_img_B = TF.to_tensor(img_B).float()
        t_sem1 = torch.from_numpy(label_sem1).long()
        t_sem2 = torch.from_numpy(label_sem2).long()
        t_bcd = torch.from_numpy(label_bcd).float()

        return {
            'img_A': t_img_A,
            'img_B': t_img_B,
            'sem1': t_sem1,
            'sem2': t_sem2,
            'bcd': t_bcd,
            'filename': f"{filename}_patch{patch_idx}"
        }

    def __len__(self):
        return len(self.file_list)
