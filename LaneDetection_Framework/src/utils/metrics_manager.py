# src/utils/metrics_manager.py

import os
import json
import time 
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class MetricsManager:
    """Enhanced metrics manager with DDP support"""
    
    def __init__(self, experiment_name: str, metrics_dir: Path, rank: Optional[int] = None):
        """Initialize metrics manager
        
        Args:
            experiment_name (str): Name of the experiment
            metrics_dir (Path): Directory to save metrics
            rank (Optional[int]): Process rank for DDP training
        """
        self.experiment_name = experiment_name
        self.rank = rank
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics structure with proper types
        self.current_metrics = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'rank': rank,
            'epochs': [],
            'train_metrics': [],
            'val_metrics': [],
            'system_metrics': [],
            'ddp_metrics': []
        }
        
        self.last_save_time = time.time()
        self.save_interval = 30  # Save every 30 seconds
        
        # Create rank-specific filename for DDP training
        self.metrics_file = self.metrics_dir / f"metrics_rank{rank if rank is not None else 'single'}.json"

    def update_metrics(self, metrics: Dict[str, Any], metric_type: str = 'train'):
        """Update metrics and save periodically
        
        Args:
            metrics (Dict[str, Any]): Metrics to update
            metric_type (str): Type of metrics ('train', 'val', 'system', or 'ddp')
        """
        if not isinstance(metrics, dict):
            return
            
        # Add timestamp and rank
        metrics['timestamp'] = datetime.now().isoformat()
        if self.rank is not None:
            metrics['rank'] = self.rank
            
        # Convert any numeric values to float for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
                
        # Update appropriate metric type
        if metric_type == 'train':
            self.current_metrics['train_metrics'].append(metrics)
        elif metric_type == 'val':
            self.current_metrics['val_metrics'].append(metrics)
        elif metric_type == 'system':
            self.current_metrics['system_metrics'].append(metrics)
        elif metric_type == 'ddp':
            self.current_metrics['ddp_metrics'].append(metrics)
        
        # Save periodically
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save_metrics()
            self.last_save_time = current_time

    def add_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Add metrics for an entire epoch
        
        Args:
            epoch (int): Current epoch number
            metrics (Dict[str, Any]): Metrics for the epoch
        """
        if not isinstance(metrics, dict):
            return
        
        # Convert numeric values to float
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = value
                
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            **processed_metrics
        }
        
        self.current_metrics['epochs'].append(epoch_data)
        self.save_metrics()

    def update_ddp_metrics(self, ddp_metrics: Dict[str, Any]):
        """Add DDP-specific metrics
        
        Args:
            ddp_metrics (Dict[str, Any]): DDP-specific metrics to add
        """
        if not isinstance(ddp_metrics, dict):
            return
            
        # Add timestamp and rank
        ddp_metrics['timestamp'] = datetime.now().isoformat()
        ddp_metrics['rank'] = self.rank
        
        # Convert numeric values
        processed_metrics = {}
        for key, value in ddp_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = value
                
        self.current_metrics['ddp_metrics'].append(processed_metrics)
        self.save_metrics()

    def finalize(self):
        """Record end time and save final metrics"""
        try:
            self.current_metrics['end_time'] = datetime.now().isoformat()
            
            # Calculate training duration
            start_time = datetime.fromisoformat(self.current_metrics['start_time'])
            end_time = datetime.fromisoformat(self.current_metrics['end_time'])
            duration = end_time - start_time
            
            # Initialize summary if needed
            if 'summary' not in self.current_metrics:
                self.current_metrics['summary'] = {}
            
            # Add duration to summary
            self.current_metrics['summary'].update({
                'training_duration_seconds': float(duration.total_seconds()),
                'training_duration_formatted': str(duration)
            })
            
            # Process epoch metrics if available
            if self.current_metrics['epochs']:
                # Safely extract train losses
                train_losses = []
                for m in self.current_metrics['epochs']:
                    train_loss = m.get('train_loss')
                    if train_loss is not None:
                        try:
                            train_losses.append(float(train_loss))
                        except (ValueError, TypeError):
                            continue
                
                # Safely extract validation losses
                val_losses = []
                for m in self.current_metrics['epochs']:
                    val_loss = m.get('val_loss')
                    if val_loss is not None:
                        try:
                            val_losses.append(float(val_loss))
                        except (ValueError, TypeError):
                            continue
                
                # Calculate summary metrics
                summary_metrics = {
                    'total_epochs': len(self.current_metrics['epochs'])
                }
                
                if train_losses:
                    summary_metrics.update({
                        'average_train_loss': sum(train_losses) / len(train_losses),
                        'best_train_loss': min(train_losses)
                    })
                    
                if val_losses:
                    summary_metrics.update({
                        'average_val_loss': sum(val_losses) / len(val_losses),
                        'best_val_loss': min(val_losses)
                    })
                    
                self.current_metrics['summary'].update(summary_metrics)
            
            self.save_metrics()
            
        except Exception as e:
            print(f"Error in finalize: {str(e)}")
            # Ensure we at least try to save what we have
            self.save_metrics()

    def save_metrics(self):
        """Save current metrics to file"""
        print("-------------------saving metrics--------------")
        print(self.metrics_file)
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to {self.metrics_file}: {str(e)}")