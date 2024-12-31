import os
import argparse
import torch
import logging
import platform
import psutil
import sys
from datetime import datetime
import json
import yaml
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from src.data.dataset import LaneDataset
from src.models.lane_detection import LaneDetectionModel
from src.training.trainer import Trainer
from src.training.distributed import setup_ddp, cleanup_ddp
from src.utils.profiling import SystemProfiler
from src.utils.visualization import Visualizer
from src.utils.metrics_manager import MetricsManager

def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system information"""
    system_info = {
        "System": {
            "OS": f"{platform.system()} {platform.version()}",
            "Python Version": sys.version.split()[0],
            "CPU Count": psutil.cpu_count(),
            "Physical CPU Count": psutil.cpu_count(logical=False),
            "Available Memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Memory Usage": f"{psutil.virtual_memory().percent}%"
        },
        "PyTorch": {
            "Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available()
        }
    }
    
    if torch.cuda.is_available():
        system_info["CUDA"] = {
            "CUDA Version": torch.version.cuda,
            "GPU Count": torch.cuda.device_count(),
            "Current Device": torch.cuda.current_device(),
            "Devices": {}
        }
        
        # Get information for each GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            system_info["CUDA"]["Devices"][f"GPU {i}"] = {
                "Name": props.name,
                "Total Memory": f"{props.total_memory / (1024**3):.2f} GB",
                "Compute Capability": f"{props.major}.{props.minor}",
                "Current Memory Usage": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                "Memory Cached": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB"
            }
    
    return system_info

class DDPTrainer:
    def __init__(self, config_path: str):
        """Initialize DDP trainer with enhanced logging"""
        self._setup_base_logging()
        logging.info("Initializing DDPTrainer")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        
        # Setup version compatibility
        self._setup_version_compatibility()
        
        # Create directories
        self._setup_directories()
        
        # Store start time and experiment name
        self.training_start_time = datetime.now()
        self.experiment_name = f"ddp_training_{self.training_start_time.strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Initialized experiment: {self.experiment_name}")
        
        # Log initial system information
        self._log_initial_system_info()

    def _setup_base_logging(self):
        """Setup base logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - [%(processName)s-%(process)d] - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'ddp_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )

    def _setup_version_compatibility(self):
        """Setup version-specific configurations"""
        self.torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        logging.info(f"PyTorch version: {self.torch_version}")
        
        self.use_amp = self.torch_version >= (1, 6)
        self.use_ddp_scaler = self.torch_version >= (1, 7)
        self.use_persistent_workers = self.torch_version >= (2, 0)
        
        # Log feature availability
        logging.info("Feature availability:")
        logging.info(f"- Automatic Mixed Precision (AMP): {self.use_amp}")
        logging.info(f"- DDP Gradient Scaler: {self.use_ddp_scaler}")
        logging.info(f"- Persistent Workers: {self.use_persistent_workers}")
        
        if torch.cuda.is_available():
            self.cuda_arch = torch.cuda.get_device_capability(0)
            self.is_old_gpu = self.cuda_arch[0] < 5
            logging.info(f"CUDA Architecture: {self.cuda_arch}")
            
            if self.is_old_gpu:
                logging.warning(f"Older GPU architecture detected: {self.cuda_arch}")
                self._adjust_config_for_old_gpu()

    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['logging']['log_dir'],
            self.config['logging']['checkpoint_dir'],
            os.path.join(self.config['logging']['log_dir'], 'metrics')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

    def _log_initial_system_info(self):
        """Log initial system information"""
        system_info = get_system_info()
        logging.info("System Configuration:")
        logging.info(json.dumps(system_info, indent=2))

    def _adjust_config_for_old_gpu(self):
        """Adjust configuration for older GPUs"""
        original_batch_size = self.config['dataset']['batch_size']
        self.config['optimization']['mixed_precision'] = False
        self.config['optimization']['cudnn_benchmark'] = False
        self.config['dataset']['batch_size'] = min(original_batch_size, 16)
        
        logging.warning("Adjusted configuration for older GPU:")
        logging.warning(f"- Disabled mixed precision")
        logging.warning(f"- Disabled cuDNN benchmark")
        logging.warning(f"- Reduced batch size from {original_batch_size} to {self.config['dataset']['batch_size']}")

    def train_ddp(self, rank: int, world_size: int):
        """Training process for each GPU"""
        # Setup process-specific logging
        self._setup_process_logging(rank)
        
        try:
            logging.info(f"Starting DDP process on rank {rank} of {world_size}")
            self._log_gpu_info(rank)
            
            # Initialize process group
            backend = 'nccl' if (torch.cuda.is_available() and not self.is_old_gpu) else 'gloo'
            logging.info(f"Using {backend} backend for distributed training")
            setup_ddp(rank, world_size)
            
            # Setup monitoring
            metrics_manager, profiler = self._setup_monitoring(rank)
            
            # Initialize model and data
            model = self._setup_model(rank)
            train_loader = self._setup_dataloader(rank, world_size)
            
            # Initialize trainer
            trainer = self._setup_trainer(rank, metrics_manager)
            
            # Train model
            logging.info(f"Starting training on rank {rank}")
            training_history, model_name = trainer.train(model, train_loader)
            
            # Record end time and calculate duration
            training_end_time = datetime.now()
            training_duration = training_end_time - self.training_start_time
            
            # Log timing information
            logging.info(f"Training completed at: {training_end_time}")
            logging.info(f"Total training duration: {training_duration}")
            
            # Finalize training
            self._finalize_training(rank, metrics_manager, training_history)
            
        except Exception as e:
            logging.error(f"Error in rank {rank}: {str(e)}", exc_info=True)
            raise
        finally:
            logging.info(f"Cleaning up DDP process on rank {rank}")
            cleanup_ddp()

    def _setup_process_logging(self, rank: int):
        """Setup logging for individual processes"""
        log_file = os.path.join(
            self.config['logging']['log_dir'],
            f'training_rank{rank}_{self.experiment_name}.log'
        )
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.DEBUG,
            format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _log_gpu_info(self, rank: int):
        """Log GPU-specific information"""
        if torch.cuda.is_available():
            gpu_info = {
                "Device Name": torch.cuda.get_device_name(rank),
                "Memory Allocated": f"{torch.cuda.memory_allocated(rank)/1e9:.2f} GB",
                "Memory Cached": f"{torch.cuda.memory_reserved(rank)/1e9:.2f} GB",
                "Current Device": torch.cuda.current_device()
            }
            logging.info(f"GPU Information for rank {rank}:")
            logging.info(json.dumps(gpu_info, indent=2))

    def _setup_monitoring(self, rank: int):
        """Initialize monitoring tools"""
        metrics_manager = MetricsManager(
            experiment_name=self.experiment_name,
            metrics_dir=Path(self.config['logging']['log_dir']) / 'metrics',
            rank=rank
        )
        
        profiler = SystemProfiler(
            os.path.join(self.config['logging']['log_dir'], f'rank_{rank}')
        )
        profiler.start_profiling()
        
        logging.info(f"Initialized monitoring for rank {rank}")
        return metrics_manager, profiler

    def _setup_model(self, rank: int):
        """Setup model with DDP wrapper"""
        logging.info(f"Setting up model on rank {rank}")
        model = LaneDetectionModel(
            num_classes=self.config['model']['num_classes'],
            backbone=self.config['model']['backbone'],
            pretrained=self.config['model']['pretrained']
        )
        
        if torch.cuda.is_available():
            try:
                model = model.cuda(rank)
                logging.info(f"Model moved to GPU {rank}")
            except RuntimeError as e:
                logging.error(f"CUDA error while moving model to GPU: {e}")
                raise
        
        ddp_kwargs = {
            "device_ids": [rank],
            "find_unused_parameters": True,
            "gradient_as_bucket_view": self.use_persistent_workers
        }
        
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        logging.info(f"DDP model setup complete on rank {rank}")
        return model

    def _setup_dataloader(self, rank: int, world_size: int):
        """Setup data loader with distributed sampler"""
        logging.info(f"Setting up data loader for rank {rank}")
        
        train_dataset = LaneDataset(
            dataset_path=self.config['dataset']['path'],
            train=True,
            size=tuple(self.config['dataset']['image_size'])
        )
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Create loader kwargs dictionary
        loader_kwargs = {
            "batch_size": self.config['dataset']['batch_size'] // world_size,
            "num_workers": self.config['dataset']['num_workers'],
            "pin_memory": not self.is_old_gpu
        }
        
        if self.use_persistent_workers:
            loader_kwargs["persistent_workers"] = True
        
        # Log the serializable configuration parameters
        logging_info = {
            "batch_size": loader_kwargs["batch_size"],
            "num_workers": loader_kwargs["num_workers"],
            "pin_memory": loader_kwargs["pin_memory"],
            "persistent_workers": loader_kwargs.get("persistent_workers", False),
            "dataset_size": len(train_dataset),
            "world_size": world_size,
            "rank": rank
        }
        
        logging.debug(f"DataLoader configuration for rank {rank}:")
        logging.debug(json.dumps(logging_info, indent=2))
        
        # Add sampler to kwargs and create dataloader
        loader_kwargs["sampler"] = sampler
        return torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

    def _setup_trainer(self, rank: int, metrics_manager):
        """Setup trainer instance"""
        logging.info(f"Initializing trainer for rank {rank}")
        
        trainer_config = self.config.copy()
        if self.is_old_gpu:
            trainer_config["optimization"]["mixed_precision"] = False
        
        trainer = Trainer(trainer_config)
        trainer.metrics_manager = metrics_manager
        
        if self.use_amp and not self.is_old_gpu:
            trainer.scaler = torch.cuda.amp.GradScaler()
            logging.info(f"Initialized AMP scaler for rank {rank}")
        
        return trainer

    def _finalize_training(self, rank: int, metrics_manager, training_history):
        """Finalize training process"""
        metrics_manager.finalize()
        logging.info(f"Finalized metrics for rank {rank}")
        
        if rank == 0:
            visualizer = Visualizer(self.config['logging']['log_dir'])
            # Create an instance of Trainer to use its method
            trainer = Trainer(self.config)
            formatted_history = trainer.get_formatted_history(training_history)
            visualizer.plot_training_history(formatted_history)
            logging.info("Generated training visualizations")

def main(config_path: str, num_gpus: int):
    """Main function to start distributed training"""
    start_time = datetime.now()
    logging.info(f"Script started at: {start_time}")
    
    try:
        # Log initial system information
        system_info = get_system_info()
        logging.info("Initial System Configuration:")
        logging.info(json.dumps(system_info, indent=2))
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. DDP requires GPU support.")
        
        if num_gpus > torch.cuda.device_count():
            raise ValueError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
        
        logging.info(f"Initializing DDP training with {num_gpus} GPUs")
        torch.cuda.empty_cache()
        
        ddp_trainer = DDPTrainer(config_path)
        
        logging.info(f"Spawning {num_gpus} processes for DDP training")
        torch.multiprocessing.spawn(
            ddp_trainer.train_ddp,
            args=(num_gpus,),
            nprocs=num_gpus,
            join=True
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Training completed successfully for gpu count:" + str(num_gpus))
        logging.info("=" * 50)
        logging.info(f"Training timeline:")
        logging.info(f"Start time     : {start_time}")
        logging.info(f"End time       : {end_time}")
        logging.info(f"Total duration : {duration}")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error("Training failed:", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        # Log final system status
        final_system_info = get_system_info()
        logging.info("Final System Configuration:")
        logging.info(json.dumps(final_system_info, indent=2))
        
        logging.info("Training script completed")


if __name__ == "__main__":
    # Configure logging for the main process
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [MainProcess] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ddp_main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    parser = argparse.ArgumentParser(description="Distributed Lane Detection Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddp_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.config, args.num_gpus)
    except Exception as e:
        logging.error("Training script failed:", exc_info=True)
        sys.exit(1)