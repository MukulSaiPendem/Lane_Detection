#!/usr/bin/env python
# coding: utf-8

# ### Lane Detection Training on Multi-CPU Setup using PyTorch DDP

# In[1]:


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.optim import Adam
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, performance_report
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.nn as nn

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available Device :", device )


# ### Dataset Class
# 
# This cell defines the `LaneDataset` class to handle the TUSimple lane detection dataset:
# 
# 1. **Initialization (`__init__`)**:
#    - Loads the dataset based on mode (`train` or `test`).
#    - Sets image size and paths for preprocessing.
# 
# 2. **Data Loading (`__getitem__`)**:
#    - Reads and preprocesses images and segmentation labels.
#    - Converts data to tensors and prepares output as a dictionary.
# 
# 3. **Visualization (`_show_sample_dataset`)**:
#    - Displays sample images and segmentation labels for verification.
# 
# 4. **Length (`__len__`)**:
#    - Returns the number of samples in the dataset.
# 
# This class prepares the dataset for training and testing. Test it by creating an instance and visualizing samples.

# In[4]:


PATH = './dataset/TUSimple'

class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path= PATH, train=True, size=(800, 360)):
        self._dataset_path = dataset_path
        self._mode = "train" if train else "test"
        self._image_size = size # w, h
        self._data = []

        if self._mode == "train":
            file_path = "train_val_gt.txt"
        elif self._mode == "test":
            file_path = "test_gt.txt"
        self._process_list(os.path.join(self._dataset_path, "train_set/seg_label/list", file_path))
            
    def __getitem__(self, idx):
        img_path = self._dataset_path + ("/train_set" if self._mode == "train" else "/test_set") + self._data[idx][0]
        image = cv2.imread(img_path)
        h, w, c = image.shape
        raw_image = image
        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ins_segmentation_path = self._dataset_path + "/train_set" + self._data[idx][1]
        ins_segmentation_image = cv2.imread(ins_segmentation_path)
        ins_segmentation_image = ins_segmentation_image[:, :, 0]
        ins_segmentation_image = cv2.resize(ins_segmentation_image, self._image_size, interpolation=cv2.INTER_LINEAR)
        
        segmentation_image = ins_segmentation_image.copy()
        segmentation_image[segmentation_image > 0] = 1
        
        image = torch.from_numpy(image).float().permute((2, 0, 1))
        segmentation_image = torch.from_numpy(segmentation_image.copy()).to(torch.int64)
        ins_segmentation_image =  torch.from_numpy(ins_segmentation_image.copy())
        
        exists = [int(i) for i in self._data[idx][2]]
        exists = torch.as_tensor(exists)
        
        output = {
            'img_path' : img_path,
            'img' : image,
            "meta" : { "full_img_path" : img_path ,
                     "img_name" :  self._data[ idx ][ 0 ]},
            'segLabel' : segmentation_image,
            'IsegLabel' : ins_segmentation_image,
            'exist' : exists,
            "original_image" : raw_image,
            "label" : segmentation_image
        }
        
        return output
    
    def probmap2lane(self, seg_pred, exist, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold
    
        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []
    
        for i in range(self.cfg.num_classes - 1):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
    
    
        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        #print(coordinates)
    
        return coordinates
    
    def _process_list(self, file_path):
        with open(file_path) as f:
            for line in f:
                words = line.split()
                image = words[0]
                segmentation = words[1]
                exists = words[2:]
                self._data.append((image, segmentation, exists)) 
                
    def _show_sample_dataset( self, number_samples = 10 ):
        
        # Visualizing some Lane Detection dataset
    
        sns.set_theme()

        f, axarr = plt.subplots( number_samples   ,3 , figsize = ( 20 , 30 ))
        
        plt.axis('off')

        for i in range( number_samples ):

            axarr[ i , 0].imshow(  self.__getitem__( idx = i )[ "original_image" ]  )
            axarr[ i , 0 ].set_title( "Lane Image Data No " + str( i + 1) )
            axarr[ i , 0 ].set_axis_off()
            
            axarr[ i , 1 ].imshow(  self.__getitem__( idx = i )[ "segLabel" ] )
            axarr[ i , 1 ].set_title( "Lane Image Segmentation Data No " + str( i + 1) )
            axarr[ i , 1 ].set_axis_off()
            
            axarr[ i , 2 ].imshow(  self.__getitem__( idx = i )[ "IsegLabel" ] )
            axarr[ i , 2 ].set_title( "Lane Image Segmentation Data No " + str( i + 1) )
            axarr[ i , 2 ].set_axis_off()

        f.tight_layout()
        plt.show()
                
    def __len__(self):
        return len(self._data)  


# ### Image Loading Test
# 
# 
# 1. **Input**:
#    - The `image_path` specifies the path to the image file.
# 
# 2. **Loading**:
#    - Uses OpenCV (`cv2.imread`) to load the image.
# 
# 3. **Validation**:
#    - Checks if the image is loaded properly.
#    - Prints a success message if the image is loaded.
#    - Displays an error message if the image cannot be loaded (e.g., file corrupted or unsupported format).
# 
# quick check to ensure the dataset is accessible and the file paths are correctly set.

# In[5]:


import cv2

image_path = './dataset/TUSimple/train_set/clips/0313-1/6040/20.jpg'
image = cv2.imread(image_path)
if image is None:
    print("ERROR: Unable to load image. File may be corrupted or unsupported format.")
else:
    print("Image loaded successfully.")


# ### Dataset Creation and Visualization
# 
# 1. **Dataset Creation**:
#    - The `LaneDataset` object is instantiated with a specified image size of `(880, 368)`.
#    - This initializes the dataset for training or testing purposes, depending on the `train` parameter in the class.
# 
# 2. **Visualization**:
#    - The `_show_sample_dataset()` method is called with `number_samples=10`.
#    - Displays 10 random samples from the dataset.
#      - First column: Original lane images.
#      - Second column: Segmentation labels.
#      - Third column: Instance segmentation labels.
# 
# 3. **Purpose**:
#    - Confirms that the dataset has been properly loaded and processed.
#    - Allows for visual inspection of the dataset to ensure it aligns with the task objectives.
# 
# crucial for verifying data integrity and understanding the dataset's structure before training a model.

# ### Model Definition and Loss Functions
# 
# 1. **Coordinate Attention Mechanism**:
#    - A `CoordAttention` class enhances the spatial and channel attention of the model.
#    - Features:
#      - Uses horizontal and vertical pooling to capture cross-dimensional relationships.
#      - Reduces dimensionality using a convolutional layer, processes separately, and then combines them.
#      - Helps the model focus on relevant regions in the image while preserving input features.
# 
# 2. **Lane Detection Model**:
#    - Built using ResNet-50 as a backbone.
#    - Encoder:
#      - Uses the pre-trained ResNet-50 (excluding fully connected layers).
#    - Coordinate Attention:
#      - Integrated after encoding for better focus on spatial details.
#    - Decoder:
#      - U-Net-like structure with transposed convolutions for upsampling.
#      - Outputs a segmentation mask with the specified number of classes.
# 
# 3. **Dice Loss**:
#    - Measures overlap between predicted and ground truth masks.
#    - Returns a value between 0 and 1, with 0 indicating perfect overlap.
#    - Suitable for binary segmentation tasks.
# 
# 4. **IoU Loss**:
#    - Computes the Intersection over Union (IoU) between predicted and ground truth masks.
#    - Penalizes incorrect segmentation more effectively in multi-class or overlapping regions.
# 
# 5. **Purpose**:
#    - The attention mechanism and decoder structure make the model robust for lane segmentation.
#    - Loss functions (DiceLoss and IoULoss) are optimized for pixel-level segmentation tasks, ensuring accurate results.

# In[7]:


import torch
import torch.nn as nn
import torchvision.models as models

class CoordAttention(nn.Module):
    """
    Coordinate Attention Mechanism to enhance spatial and channel focus.
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Horizontal pooling (height remains, width = 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Vertical pooling (width remains, height = 1)

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x  # Save the input for later addition
        n, c, h, w = x.size()

        # Perform horizontal and vertical pooling
        x_h = self.pool_h(x)  # Shape: (n, c, h, 1)
        x_w = self.pool_w(x)  # Shape: (n, c, 1, w)

        # Pass pooled features through conv1
        x_h = self.conv1(x_h)  # Shape: (n, mid_channels, h, 1)
        x_w = self.conv1(x_w)  # Shape: (n, mid_channels, 1, w)

        # Broadcast to match spatial dimensions
        x_h = x_h.expand(-1, -1, h, w)  # Expand width to match input (n, mid_channels, h, w)
        x_w = x_w.expand(-1, -1, h, w)  # Expand height to match input (n, mid_channels, h, w)

        # Combine horizontally and vertically processed features
        combined = x_h + x_w  # Shape: (n, mid_channels, h, w)

        # Pass through relu and final conv layers
        combined = self.relu(combined)
        x_h = self.conv_h(combined)  # Shape: (n, out_channels, h, w)
        x_w = self.conv_w(combined)  # Shape: (n, out_channels, h, w)

        return torch.sigmoid(x_h + x_w) * identity  # Combine with input features


class LaneDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(LaneDetectionModel, self).__init__()
        # ResNet-50 backbone
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers

        # Coordinate Attention
        self.coord_att = CoordAttention(2048, 2048)

        # Upsampling head (U-Net-like)
        self.up1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)  # Output segmentation mask

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Coordinate Attention
        features = self.coord_att(features)

        # Decoder
        x = self.up1(features)
        x = self.up2(x)
        x = self.up3(x)

        # Final segmentation output
        x = self.final_conv(x)
        return x


# In[8]:


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid for binary masks
        intersection = (pred * target).sum(dim=(2, 3))
        dice_score = 2.0 * intersection / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)))
        return 1 - dice_score.mean()


# In[9]:


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target - pred * target).sum(dim=(2, 3))
        iou = intersection / union
        return 1 - iou.mean()




# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# In[12]:


import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# DDP Setup Functions
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()


def log_metrics(metrics, file_path):
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

# DDP Training Function
import os
import json
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime


def train_ddp(rank, world_size, train_dataset, batch_size, lr, output_file, log_interval=120):
    setup_ddp(rank, world_size)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size // world_size, sampler=sampler)

    # Initialize model, optimizer, and loss
    device = torch.device(f"cpu")
    model = LaneDetectionModel(num_classes=2).to(device)
    model = DDP(model)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    metrics = []
    start_time = time.time()
    last_log_time = start_time

    progress_bar = tqdm(train_loader, desc=f"Rank {rank} Training", position=rank, leave=True)
    for batch_idx, batch in enumerate(progress_bar):
        current_time = time.time()

        # Forward pass
        images = batch['img'].to(device)
        targets = batch['segLabel'].to(device)
        outputs = model(images)
        outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record metrics every log_interval seconds
        if current_time - last_log_time >= log_interval:
            metrics.append({
                "timestamp": datetime.now().isoformat(),
                "rank": rank,
                "step": batch_idx,
                "loss": loss.item(),
                "batch_size": batch_size,
                "learning_rate": lr,
                "elapsed_time": current_time - start_time
            })
            last_log_time = current_time

    # Save metrics at the end
    if rank == 0:  # Only the master process writes the file
        log_metrics(metrics, output_file)

    cleanup_ddp()


if __name__ == "__main__":
    world_size = 1  # Single process
    train_dataset = LaneDataset(dataset_path="./dataset/TUSimple", train=True, size=(800, 360))
    batch_size = 32
    lr = 0.001
    output_file = "./metrics.json"

    os.environ['OMP_NUM_THREADS'] = str(world_size)
    torch.set_num_threads(world_size)

    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size, train_dataset, batch_size, lr, output_file),
        nprocs=world_size,
        join=True
    )
