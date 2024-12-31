import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging
from pathlib import Path

class DaskProcessor:
    """Dask-based data processing and analysis utilities"""
    
    def __init__(self, 
                 n_workers: int = 4,
                 memory_limit: str = '4GB'):
        """
        Initialize Dask processor
        
        Args:
            n_workers: Number of Dask workers
            memory_limit: Memory limit per worker
        """
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=2,
            memory_limit=memory_limit
        )
        self.client = Client(self.cluster)
        logging.info(f"Initialized Dask cluster with {n_workers} workers")

    def process_image_batch(self, 
                          image_paths: List[str],
                          target_size: Tuple[int, int]) -> da.Array:
        """
        Process a batch of images using Dask
        
        Args:
            image_paths: List of image file paths
            target_size: Target size for resized images
            
        Returns:
            Dask array containing processed images
        """
        def load_and_process(path: str) -> np.ndarray:
            img = cv2.imread(path)
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        # Create Dask array from lazy computations
        lazy_arrays = [
            dask.delayed(load_and_process)(path)
            for path in image_paths
        ]
        
        # Stack into a single Dask array
        sample = load_and_process(image_paths[0])
        arrays = [
            da.from_delayed(
                lazy_array,
                shape=sample.shape,
                dtype=sample.dtype
            )
            for lazy_array in lazy_arrays
        ]
        
        return da.stack(arrays)

    def parallel_data_preparation(self,
                                data_dir: str,
                                annotation_file: str) -> dd.DataFrame:
        """
        Parallel data preparation using Dask
        
        Args:
            data_dir: Directory containing data
            annotation_file: Path to annotation file
            
        Returns:
            Dask DataFrame with processed annotations
        """
        # Read annotations with Dask
        df = dd.read_csv(annotation_file)
        
        # Add path columns
        def add_paths(row):
            return {
                'image_path': str(Path(data_dir) / row['image_name']),
                'label_path': str(Path(data_dir) / row['label_name'])
            }
        
        df = df.assign(**df.map_partitions(lambda x: x.apply(add_paths, axis=1)))
        return df

    def analyze_dataset_statistics(self, 
                                 image_paths: List[str]) -> dict:
        """
        Analyze dataset statistics using Dask
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing dataset statistics
        """
        def compute_image_stats(path: str) -> dict:
            img = cv2.imread(path)
            return {
                'mean': np.mean(img, axis=(0, 1)),
                'std': np.std(img, axis=(0, 1)),
                'size': img.shape
            }

        # Create delayed computations
        stats = [
            dask.delayed(compute_image_stats)(path)
            for path in image_paths
        ]
        
        # Compute all statistics
        results = dask.compute(*stats)
        
        # Aggregate statistics
        means = np.array([r['mean'] for r in results])
        stds = np.array([r['std'] for r in results])
        
        return {
            'mean': np.mean(means, axis=0),
            'std': np.mean(stds, axis=0),
            'shapes': [r['size'] for r in results]
        }

    def parallel_performance_analysis(self,
                                    metrics_data: List[dict]) -> dd.DataFrame:
        """
        Analyze performance metrics using Dask
        
        Args:
            metrics_data: List of performance metric dictionaries
            
        Returns:
            Dask DataFrame with analyzed metrics
        """
        # Convert to Dask DataFrame
        df = dd.from_pandas(pd.DataFrame(metrics_data), npartitions=self.cluster.n_workers)
        
        # Compute various statistics
        analysis = {
            'mean_cpu_util': df['cpu_utilization'].mean(),
            'mean_memory_util': df['memory_utilization'].mean(),
            'training_time_per_epoch': df.groupby('epoch')['time_taken'].mean(),
            'gpu_util_stats': df['gpu_utilization'].describe() if 'gpu_utilization' in df.columns else None
        }
        
        return dask.compute(analysis)[0]

    def cleanup(self):
        """Cleanup Dask cluster and client"""
        self.client.close()
        self.cluster.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()