import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Callable, Dict, Any
import logging

def setup_ddp(rank: int, world_size: int):
    """
    Setup for distributed data parallel training
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    
    # Check GPU architecture
    cuda_arch = torch.cuda.get_device_capability(rank)
    is_old_gpu = cuda_arch[0] < 5  # Pre-Maxwell architecture
    
    # Initialize process group with appropriate backend
    backend = 'gloo'  # Default to gloo for better compatibility
    if torch.cuda.is_available() and not is_old_gpu:
        backend = 'nccl'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Disable cuDNN for older GPUs
    if is_old_gpu:
        torch.backends.cudnn.enabled = False
        logging.info(f"Rank {rank}: Disabled cuDNN for older GPU architecture")


def cleanup_ddp():
    """Cleanup distributed processing"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logging.warning(f"Error during process group cleanup: {e}")

class DDPWrapper:
    """Wrapper for distributed data parallel training"""
    
    def __init__(self, 
                 model_fn: Callable,
                 world_size: int,
                 config: Dict[str, Any]):
        """
        Initialize DDP wrapper
        
        Args:
            model_fn (Callable): Function that creates the model
            world_size (int): Number of processes for distributed training
            config (dict): Configuration dictionary
        """
        self.model_fn = model_fn
        self.world_size = world_size
        self.config = config

    def prepare_dataloader(self, dataset, batch_size: int, rank: int) -> torch.utils.data.DataLoader:
        """
        Prepare distributed dataloader
        
        Args:
            dataset: Dataset instance
            batch_size (int): Batch size per process
            rank (int): Process rank
            
        Returns:
            DataLoader: Distributed data loader
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=True
        )

    def setup_model(self, rank: int) -> DDP:
        """
        Setup distributed model
        
        Args:
            rank (int): Process rank
            
        Returns:
            DDP: Distributed model
        """
        model = self.model_fn().to(rank)
        return DDP(model, device_ids=[rank])

    def train(self, rank: int, train_fn: Callable):
        """
        Train the model in a distributed setting
        
        Args:
            rank (int): Process rank
            train_fn (Callable): Training function
        """
        try:
            # Setup process group
            setup_ddp(rank, self.world_size)
            
            # Create distributed model
            model = self.setup_model(rank)
            
            # Run training function
            train_fn(model, rank)
            
        except Exception as e:
            logging.error(f"Error in process {rank}: {str(e)}")
            raise
        
        finally:
            cleanup_ddp()

    @staticmethod
    def run_distributed(world_size: int, train_fn: Callable):
        """
        Run distributed training across multiple processes
        
        Args:
            world_size (int): Number of processes
            train_fn (Callable): Training function
        """
        mp.spawn(
            train_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )

class ModelParallelWrapper:
    """Wrapper for model parallel training"""
    
    def __init__(self, num_gpus: int):
        """
        Initialize model parallel wrapper
        
        Args:
            num_gpus (int): Number of GPUs to use for model parallelism
        """
        self.num_gpus = num_gpus
        assert torch.cuda.device_count() >= num_gpus, f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available"

    def partition_model(self, model: torch.nn.Module) -> torch.nn.ModuleList:
        """
        Partition model across multiple GPUs
        
        Args:
            model (nn.Module): Model to partition
            
        Returns:
            ModuleList: List of model partitions
        """
        # Example partitioning strategy - can be customized based on model architecture
        layers = []
        if hasattr(model, 'encoder'):
            layers.append(model.encoder)
        if hasattr(model, 'coord_att'):
            layers.append(model.coord_att)
        if hasattr(model, 'up1'):
            layers.extend([model.up1, model.up2, model.up3])
        if hasattr(model, 'final_conv'):
            layers.append(model.final_conv)
        
        # Distribute layers across GPUs
        partitions = torch.nn.ModuleList()
        layers_per_partition = max(1, len(layers) // self.num_gpus)
        
        for i in range(0, len(layers), layers_per_partition):
            partition = torch.nn.Sequential(*layers[i:i + layers_per_partition])
            partition.to(f'cuda:{i // layers_per_partition}')
            partitions.append(partition)
        
        return partitions

    def forward_backward(self, partitions: torch.nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """
        Forward and backward pass through partitioned model
        
        Args:
            partitions (ModuleList): List of model partitions
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        # Forward pass
        current_input = x
        intermediate_outputs = []
        
        for i, partition in enumerate(partitions):
            device = next(partition.parameters()).device
            current_input = current_input.to(device)
            current_output = partition(current_input)
            intermediate_outputs.append(current_output)
            current_input = current_output
        
        # Backward pass can be implemented here if needed
        
        return current_output