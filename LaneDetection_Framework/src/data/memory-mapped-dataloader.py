import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from pathlib import Path
import psutil
import sys
sys.path.append('..')
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

class ConservativeMemoryMappedDataset(Dataset):
    """TuSimple Lane Detection Dataset with Conservative Memory Mapping"""
    
    def __init__(self, 
                 dataset_path: str,
                 train: bool = True,
                 size: Tuple[int, int] = (800, 360),
                 max_cache_items: int = 50,  # Reduced cache size
                 dataset_memory_limit_gb: float = 8.0):  # Very conservative memory limit for dataset
        """
        Initialize Conservative Memory-Mapped Dataset
        
        Args:
            dataset_path: Path to dataset
            train: Whether to use train or test set
            size: Target image size (width, height)
            max_cache_items: Maximum number of items to keep in memory
            dataset_memory_limit_gb: Maximum memory allocation for dataset
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._dataset_path = dataset_path
        self._mode = "train" if train else "test"
        self._image_size = size
        self._max_cache_items = max_cache_items
        self._memory_limit = dataset_memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Initialize data list
        self._data = []
        
        # Setup cache with lock for thread safety
        self._cache_lock = threading.Lock()
        self._memmap_cache = {}  # Path -> memmap array
        self._cache_timestamps = {}  # Path -> last access time
        
        # Create process-specific cache directory
        rank = getattr(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, 'rank', 0)
        self._cache_dir = Path(tempfile.gettempdir()) / f"lane_dataset_cache_rank_{rank}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset
        self._init_dataset()
        
        # Initial memory check
        self._check_system_resources()
        
    def _check_system_resources(self):
        """Check system resources and log warnings if necessary"""
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            self.logger.warning(f"High system memory usage: {memory.percent}%")
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"Current GPU memory usage: {gpu_memory_used:.2f} GB")
    
    def _init_dataset(self):
        """Initialize dataset by reading file lists"""
        try:
            file_path = "train_val_gt.txt" if self._mode == "train" else "test_gt.txt"
            list_path = os.path.join(self._dataset_path, "train_set/seg_label/list", file_path)
            self._process_list(list_path)
        except Exception as e:
            self.logger.error(f"Error initializing dataset: {str(e)}")
            raise
    
    def _process_list(self, file_path: str):
        """Process the dataset list file"""
        with open(file_path) as f:
            for line in f:
                words = line.split()
                if len(words) >= 3:  # Basic validation
                    image = words[0]
                    segmentation = words[1]
                    exists = words[2:]
                    self._data.append((image, segmentation, exists))
    
    def _manage_cache(self, new_path: str):
        """Manage cache size and memory usage"""
        with self._cache_lock:
            # Check if we need to remove items
            while (len(self._memmap_cache) >= self._max_cache_items or 
                   psutil.Process().memory_info().rss > self._memory_limit):
                if not self._memmap_cache:
                    break
                    
                # Remove oldest item
                oldest_path = min(self._cache_timestamps, key=self._cache_timestamps.get)
                self._remove_from_cache(oldest_path)
    
    def _remove_from_cache(self, path: str):
        """Remove an item from cache and clean up resources"""
        if path in self._memmap_cache:
            try:
                memmap_array = self._memmap_cache[path]
                del self._memmap_cache[path]
                del self._cache_timestamps[path]
                
                # Close memmap if it exists
                if hasattr(memmap_array, '_mmap') and memmap_array._mmap is not None:
                    memmap_array._mmap.close()
                
                # Remove cache file
                cache_file = self._cache_dir / f"{hash(path)}.npy"
                if cache_file.exists():
                    cache_file.unlink()
                    
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error removing cache item {path}: {str(e)}")
    
    def _load_and_process_image(self, img_path: str) -> np.ndarray:
        """Load and process a single image"""
        try:
            # Check if in cache first
            with self._cache_lock:
                if img_path in self._memmap_cache:
                    self._cache_timestamps[img_path] = time.time()
                    return np.array(self._memmap_cache[img_path])
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            image = cv2.resize(image, self._image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Manage cache before adding new item
            self._manage_cache(img_path)
            
            # Create cache file and memmap
            cache_file = self._cache_dir / f"{hash(img_path)}.npy"
            np.save(cache_file, image)
            memmap = np.load(cache_file, mmap_mode='r')
            
            # Update cache
            with self._cache_lock:
                self._memmap_cache[img_path] = memmap
                self._cache_timestamps[img_path] = time.time()
            
            return np.array(memmap)
            
        except Exception as e:
            self.logger.error(f"Error processing image {img_path}: {str(e)}")
            raise
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset"""
        try:
            # Construct image path
            img_path = os.path.join(self._dataset_path, 
                                  "train_set" if self._mode == "train" else "test_set",
                                  self._data[idx][0].lstrip('/'))
            
            # Load and process image
            image = self._load_and_process_image(img_path)
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
            
            # Load segmentation
            seg_path = os.path.join(self._dataset_path,
                                  "train_set" if self._mode == "train" else "test_set",
                                  self._data[idx][1].lstrip('/'))
            
            # Process segmentation (without caching)
            seg_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            seg_image = cv2.resize(seg_image, self._image_size)
            binary_seg = (seg_image > 0).astype(np.uint8)
            
            # Create return dictionary
            return {
                'img_path': img_path,
                'img': image_tensor,
                'meta': {
                    'full_img_path': img_path,
                    'img_name': self._data[idx][0]
                },
                'segLabel': torch.from_numpy(binary_seg).long(),
                'exist': torch.tensor([int(x) for x in self._data[idx][2]]),
                'label': torch.from_numpy(binary_seg).long()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting item {idx}: {str(e)}")
            raise
    
    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self._data)
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # Clean up cache
            with self._cache_lock:
                for path in list(self._memmap_cache.keys()):
                    self._remove_from_cache(path)
            
            # Remove cache directory
            if self._cache_dir.exists():
                for file in self._cache_dir.glob("*.npy"):
                    file.unlink(missing_ok=True)
                self._cache_dir.rmdir()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# # Create dataset
# dataset = ConservativeMemoryMappedDataset(
#     dataset_path='/path/to/dataset',
#     train=True,
#     size=(800, 360),
#     max_cache_items=50,  # Very conservative cache size
#     dataset_memory_limit_gb=8.0  # Very conservative memory limit
# )

# # Create sampler for distributed training
# sampler = torch.utils.data.distributed.DistributedSampler(
#     dataset,
#     num_replicas=world_size,
#     rank=rank
# )

# # Create dataloader
# loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=batch_size // world_size,
#     sampler=sampler,
#     num_workers=4,  # Reduced number of workers
#     pin_memory=True
# )