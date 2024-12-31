import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List

class LaneDataset(Dataset):
    """TuSimple Lane Detection Dataset"""
    
    def __init__(self, 
                 dataset_path: str,
                 train: bool = True,
                 size: Tuple[int, int] = (800, 360)):
        """
        Initialize the LaneDataset.
        
        Args:
            dataset_path (str): Path to the dataset
            train (bool): Whether to use train or test set
            size (tuple): Target size for images (width, height)
        """
        self._dataset_path = dataset_path
        self._mode = "train" if train else "test"
        self._image_size = size
        self._data = []
        
        # Initialize dataset
        self._init_dataset()
    
    def _init_dataset(self):
        """Initialize dataset by reading file lists"""
        file_path = "train_val_gt.txt" if self._mode == "train" else "test_gt.txt"
        list_path = os.path.join(self._dataset_path, "train_set/seg_label/list", file_path)
        self._process_list(list_path)
    
    def _process_list(self, file_path: str):
        """Process the dataset list file"""
        with open(file_path) as f:
            for line in f:
                words = line.split()
                image = words[0]
                segmentation = words[1]
                exists = words[2:]
                self._data.append((image, segmentation, exists))
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset"""        
        # img_path = os.path.join(self._dataset_path,
        #                       "train_set" if self._mode == "train" else "test_set",
        #                       self._data[idx][0])
        img_path = '/home/rebbapragada.s/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple'
        # if + "/train_set" if self._mode == "train" else "/test_set" 
        if self._mode == "train":
            img_path = img_path + "/train_set"
        else:
            img_path = img_path + "/test_set"
        img_path = img_path + self._data[idx][0]        
        # print(img_path)
        # Load and process image
        image = cv2.imread(img_path)        
        raw_image = image.copy()
        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load and process segmentation
        seg_path = '/home/rebbapragada.s/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple'
        if self._mode == "train":
            seg_path = seg_path + "/train_set"
        else:
            seg_path = seg_path + "/test_set"        
        seg_path = seg_path + self._data[idx][0]
        # print(seg_path)
        seg_image = cv2.imread(seg_path)
        seg_image = seg_image[:, :, 0]  # Take first channel
        seg_image = cv2.resize(seg_image, self._image_size, interpolation=cv2.INTER_LINEAR)
        
        # Create binary segmentation
        binary_seg = seg_image.copy()
        binary_seg[binary_seg > 0] = 1
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().permute((2, 0, 1))
        seg_tensor = torch.from_numpy(binary_seg).to(torch.int64)
        exists_tensor = torch.as_tensor([int(i) for i in self._data[idx][2]])
        
        return {
            'img_path': img_path,
            'img': image_tensor,
            'meta': {
                'full_img_path': img_path,
                'img_name': self._data[idx][0]
            },
            'segLabel': seg_tensor,
            'exist': exists_tensor,
            'original_image': raw_image,
            'label': seg_tensor
        }
    
    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self._data)
