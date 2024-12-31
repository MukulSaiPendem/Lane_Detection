import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import logging
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.data.dataset import LaneDataset
from src.models.lane_detection import LaneDetectionModel
from src.training.losses import CombinedLoss
from src.training.metrics import MetricsTracker, TrainingMetrics
from src.utils.metrics_manager import MetricsManager

class Trainer:
    """
    Trainer class for Lane Detection model
    """
    def __init__(self, config_path_or_dict: Union[str, Dict[str, Any]]):
        """
        Initialize trainer with configuration
        
        Args:
            config_path_or_dict: Path to config file or config dictionary
        """
        # Load configuration
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict
        
        self.device = self._setup_device()
        self.metrics = MetricsTracker(self.config['logging']['log_dir'])
        
        # Create directories
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)

        self.current_history = {}
        self.metrics_manager = None  # Will be set by DDPTrainer
        self.scaler = None  # Will be set by DDPTrainer if using AMP
        self.start_time = datetime.now()

        # Initialize logger
        logging.basicConfig(
            filename=os.path.join(self.config['logging']['log_dir'], 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _setup_device(self) -> torch.device:
        """Setup compute device based on configuration"""
        if self.config['system']['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            
            cuda_arch = torch.cuda.get_device_capability(0)
            is_old_gpu = cuda_arch[0] < 5
            
            if is_old_gpu:
                torch.backends.cudnn.enabled = False
                logging.info("Disabled cuDNN for older GPU architecture")
            else:
                torch.backends.cudnn.benchmark = self.config['optimization']['cudnn_benchmark']
                if self.config['optimization'].get('mixed_precision', False):
                    torch.backends.cudnn.enabled = True
                    
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(self.config['system'].get('num_threads', 4))
            logging.info(f"Using CPU with {self.config['system'].get('num_threads', 4)} threads")
        
        return device

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        if self.config['training']['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        return optimizer

    def train(self, 
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> Tuple[Dict[str, Any], str]:
        """Train the model"""
        model = model.to(self.device)
        criterion = CombinedLoss().to(self.device)
        optimizer = self._setup_optimizer(model)
        
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': []
        }
        self.current_history = training_history
        
        # Record training start
        if self.metrics_manager:
            self.metrics_manager.update_metrics({
                'event': 'training_start',
                'model_config': str(model),
                'optimizer_config': str(optimizer),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }, metric_type='system')
        
        logging.info("Starting training...")
        global_step = 0
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step
            )
            global_step += len(train_loader)
            
            # Update history
            training_history['train_metrics'].append(train_metrics)
            training_history['train_losses'].append(train_metrics.batch_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            training_history['epochs'].append(epoch + 1)
            
            # Validate if loader provided
            if val_loader is not None:
                val_metrics = self._validate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    epoch=epoch
                )
                training_history['val_metrics'].append(val_metrics)
                training_history['val_losses'].append(val_metrics.batch_loss)
                
                # Save best model
                if val_metrics.batch_loss < best_val_loss:
                    best_val_loss = val_metrics.batch_loss
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )
            
            # Record epoch metrics
            if self.metrics_manager:
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics.batch_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'time_taken': train_metrics.time_taken,
                    'gpu_memory_used': train_metrics.gpu_memory_used if train_metrics.gpu_memory_used is not None else 0,
                    'gpu_utilization': train_metrics.gpu_utilization if train_metrics.gpu_utilization is not None else 0,
                    'val_loss': val_metrics.batch_loss if val_loader is not None else None,
                }
                self.metrics_manager.add_epoch_metrics(epoch, epoch_metrics)
            
            # Print epoch summary
            logging.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics.batch_loss:.4f} - "
                f"Time: {train_metrics.time_taken:.2f}s"
            )
            if val_loader is not None:
                logging.info(f"Val Loss: {val_metrics.batch_loss:.4f}")
        
        self.current_history = training_history
        
        final_model_name = self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=self.config['training']['epochs']-1,
            metrics=train_metrics,
            is_best=False
        )
        
        # Record training end
        if self.metrics_manager:
            self.metrics_manager.finalize()

        return training_history, final_model_name

    def _train_epoch(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    global_step: int) -> TrainingMetrics:
        """Train for one epoch"""
        if hasattr(model, 'module'):  # This checks if model is wrapped in DDP
            logging.info(f"Model is using DDP on rank {torch.distributed.get_rank()}")
            batch_size_per_gpu = next(iter(train_loader))['img'].shape[0]
            logging.info(f"Batch size per GPU: {batch_size_per_gpu}")
        model.train()
        total_loss = 0
        batch_sizes = []
        batch_times = []
        batch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        system_metrics = self.metrics.get_system_metrics()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Get batch data
            images = batch['img'].to(self.device)
            targets = batch['segLabel'].to(self.device)
            
            # Forward pass with mixed precision where available
            if self.scaler is not None:
                with autocast():
                    outputs = model(images)
                    outputs = F.interpolate(outputs, 
                                        size=targets.shape[1:],
                                        mode='bilinear',
                                        align_corners=False)
                    loss = criterion(outputs, targets)
                
                # Backward pass with scaling
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                outputs = F.interpolate(outputs, 
                                    size=targets.shape[1:],
                                    mode='bilinear',
                                    align_corners=False)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            batch_size = images.size(0)
            batch_loss = loss.item()
            
            total_loss += batch_loss
            batch_sizes.append(batch_size)
            batch_times.append(batch_time)
            batch_losses.append(batch_loss)
            
            # Update metrics manager
            if self.metrics_manager:
                batch_metrics = {
                    'step': global_step + batch_idx,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'batch_loss': batch_loss,
                    'batch_size': batch_size,
                    'batch_time': batch_time,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                if self.device.type == 'cuda':
                    batch_metrics.update({
                        'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                        'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9
                    })
                
                self.metrics_manager.update_metrics(batch_metrics, 'train')
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Collect system metrics
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            try:
                gpu_utilization = self.metrics.get_system_metrics().get('gpu_utilization', 0)
            except:
                gpu_utilization = 0
        else:
            gpu_memory = None
            gpu_utilization = None
        
        metrics = TrainingMetrics(
            epoch=epoch,
            batch_loss=avg_loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            time_taken=epoch_time,
            memory_used=system_metrics['memory_used'],
            cpu_utilization=system_metrics['cpu_utilization'],
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory
        )
        
        return metrics

    def _validate(self,
                model: nn.Module,
                val_loader: DataLoader,
                criterion: nn.Module,
                epoch: int) -> TrainingMetrics:
        """Validate the model"""
        model.eval()
        total_loss = 0
        batch_sizes = []
        batch_times = []
        batch_losses = []
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        system_metrics = self.metrics.get_system_metrics()
        validation_start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                images = batch['img'].to(self.device)
                targets = batch['segLabel'].to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = model(images)
                        outputs = F.interpolate(outputs, 
                                            size=targets.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    outputs = F.interpolate(outputs, 
                                        size=targets.shape[1:],
                                        mode='bilinear',
                                        align_corners=False)
                    loss = criterion(outputs, targets)
                
                # Update metrics
                batch_time = time.time() - batch_start_time
                batch_size = images.size(0)
                batch_loss = loss.item()
                
                total_loss += batch_loss
                batch_sizes.append(batch_size)
                batch_times.append(batch_time)
                batch_losses.append(batch_loss)
                
                # Update metrics manager
                if self.metrics_manager:
                    val_metrics = {
                        'batch_idx': batch_idx,
                        'batch_loss': batch_loss,
                        'batch_size': batch_size,
                        'batch_time': batch_time
                    }
                    
                    if self.device.type == 'cuda':
                        val_metrics.update({
                            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                            'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9
                        })
                    
                    self.metrics_manager.update_metrics(val_metrics, 'val')
                
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })

        avg_loss = total_loss / len(val_loader)
        validation_time = time.time() - validation_start_time
        
        metrics = TrainingMetrics(
            epoch=epoch,
            batch_loss=avg_loss,
            learning_rate=0.3,  # Not relevant for validation
            time_taken=validation_time,
            memory_used=system_metrics['memory_used'],
            cpu_utilization=system_metrics['cpu_utilization'],
            gpu_utilization=system_metrics.get('gpu_utilization'),
            gpu_memory_used=torch.cuda.memory_allocated() / 1e9 if self.device.type == 'cuda' else None
        )
        
        return metrics

    def _save_checkpoint(self,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    metrics: TrainingMetrics,
                    is_best: bool = False):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_{timestamp}"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics.__dict__,
            'config': self.config,
            'training_duration': str(datetime.now() - self.start_time)
        }
        
        if is_best:
            path = os.path.join(self.config['logging']['checkpoint_dir'], f'best_{model_name}.pth')
        else:
            path = os.path.join(
                self.config['logging']['checkpoint_dir'],
                f"{model_name}_epoch_{epoch+1}.pth"
            )
        
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved: {path}")
        
        # Save training history
        self._save_history(self.current_history, model_name)
        
        return model_name

    def _save_history(self, history: Dict[str, Any], model_name: str):
        """Save training history"""
        history_file = os.path.join(
            self.config['logging']['checkpoint_dir'], 
            f'history_{model_name}.pth'
        )
        torch.save(history, history_file)
        logging.info(f"History saved: {history_file}")

    def load_history(self, model_name: str) -> Dict[str, Any]:
        """Load training history for a specific model"""
        history_file = os.path.join(
            self.config['logging']['checkpoint_dir'], 
            f'history_{model_name}.pth'
        )
        if os.path.exists(history_file):
            return torch.load(history_file)
        else:
            raise FileNotFoundError(f"No history file found for model: {model_name}")

    def load_model(self, model_name: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model and its history"""
        checkpoint_path = os.path.join(
            self.config['logging']['checkpoint_dir'],
            f'{model_name}.pth'
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found for model: {model_name}")
        
        checkpoint = torch.load(checkpoint_path)
        model = LaneDetectionModel(
            num_classes=self.config['model']['num_classes'],
            backbone=self.config['model']['backbone'],
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history
        history = self.load_history(model_name)
        
        return model, history

    def get_formatted_history(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Format history for visualization"""
        formatted = {}
        
        # Extract metrics
        formatted['train_losses'] = [m.batch_loss for m in history['train_metrics']]
        formatted['val_losses'] = [m.batch_loss for m in history['val_metrics']] if history['val_metrics'] else []
        formatted['learning_rates'] = [m.learning_rate for m in history['train_metrics']]
        formatted['epochs'] = list(range(1, len(history['train_metrics']) + 1))
        
        # System metrics
        formatted['cpu_utilization'] = [m.cpu_utilization for m in history['train_metrics']]
        formatted['memory_used'] = [m.memory_used for m in history['train_metrics']]
        if history['train_metrics'][0].gpu_utilization is not None:
            formatted['gpu_utilization'] = [m.gpu_utilization for m in history['train_metrics']]
        
        # Training time
        formatted['time_per_epoch'] = [m.time_taken for m in history['train_metrics']]
        
        return formatted