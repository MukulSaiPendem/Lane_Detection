import torch
import numpy as np
from typing import Dict, Any
import time
import psutil
import GPUtil
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingMetrics:
    """Class for tracking training metrics"""
    epoch: int
    batch_loss: float
    learning_rate: float
    time_taken: float
    memory_used: float
    cpu_utilization: float
    gpu_utilization: float = None
    gpu_memory_used: float = None

class MetricsTracker:
    """
    Tracks and logs various training metrics
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.current_time = time.time()
        self.metrics_history = []

    def start_epoch(self):
        """Start timing an epoch"""
        self.current_time = time.time()

    def end_epoch(self, metrics: TrainingMetrics):
        """
        Log metrics for an epoch
        
        Args:
            metrics (TrainingMetrics): Metrics for the epoch
        """
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {
            'cpu_utilization': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().percent,
        }

        # Get GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_utilization'] = gpus[0].load * 100
                metrics['gpu_memory_used'] = gpus[0].memoryUtil * 100
        except:
            pass

        return metrics

    def calculate_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate segmentation accuracy
        
        Args:
            pred (torch.Tensor): Predicted segmentation
            target (torch.Tensor): Ground truth segmentation
            
        Returns:
            float: Accuracy value
        """
        pred_mask = torch.argmax(pred, dim=1)
        correct = (pred_mask == target).float().mean()
        return correct.item()

    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate IoU (Intersection over Union)
        
        Args:
            pred (torch.Tensor): Predicted segmentation
            target (torch.Tensor): Ground truth segmentation
            
        Returns:
            float: IoU value
        """
        pred_mask = (torch.argmax(pred, dim=1) > 0).float()
        target_mask = (target > 0).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        
        return (intersection / (union + 1e-7)).item()

    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = (f"[{timestamp}] Epoch {metrics.epoch}: "
                  f"Loss={metrics.batch_loss:.4f}, "
                  f"Time={metrics.time_taken:.2f}s, "
                  f"CPU={metrics.cpu_utilization:.1f}%, "
                  f"Memory={metrics.memory_used:.1f}%")
        
        if metrics.gpu_utilization is not None:
            log_str += (f", GPU={metrics.gpu_utilization:.1f}%, "
                       f"GPU_Memory={metrics.gpu_memory_used:.1f}%")
        
        with open(f"{self.log_dir}/training_log.txt", "a") as f:
            f.write(log_str + "\n")