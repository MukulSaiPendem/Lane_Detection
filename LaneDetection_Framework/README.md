# HPC TuSimple Lane Detection

This repository contains a high-performance computing (HPC) implementation of lane detection using the TuSimple dataset. The implementation focuses on performance optimization through various parallelization techniques and provides comprehensive benchmarking tools.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Code Organization](#code-organization)
6. [Parallelization Techniques](#parallelization-techniques)
7. [Profiling and Benchmarking](#profiling-and-benchmarking)
8. [Visualization Tools](#visualization-tools)
9. [Training Pipeline](#training-pipeline)
10. [Performance Optimization](#performance-optimization)

## Features

- Modular implementation of TuSimple lane detection
- Multiple parallelization strategies:
  - Data Parallelism (DistributedDataParallel)
  - Model Parallelism
  - Hybrid Parallelism capabilities
- Comprehensive system profiling
- Performance benchmarking
- Resource utilization tracking
- Visualization tools
- Configurable training pipeline

## Project Structure

```
hpc-tusimple/
├── configs/                 # Configuration files
│   ├── base_config.yaml    # Base configuration template
│   ├── cpu_config.yaml     # CPU-specific settings
│   └── gpu_config.yaml     # GPU-specific settings
├── src/
│   ├── data/               # Data handling
│   │   ├── dataset.py      # TuSimple dataset implementation
│   │   └── transforms.py   # Data transformations
│   ├── models/             # Model architectures
│   │   ├── attention.py    # Coordinate Attention mechanism
│   │   └── lane_detection.py # Main model implementation
│   ├── training/           # Training components
│   │   ├── trainer.py      # Training loop implementation
│   │   ├── losses.py       # Loss functions
│   │   ├── metrics.py      # Performance metrics
│   │   └── distributed.py  # Distributed training utilities
│   └── utils/              # Utility functions
│       ├── profiling.py    # System profiling tools
│       └── visualization.py # Visualization utilities
├── benchmark.py            # Benchmarking script
├── main.py                # Main training script
├── setup.py               # Package setup
└── README.md              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hpc-tusimple.git
cd hpc-tusimple
```

2. Install dependencies:
```bash
pip install -e .
```

For development installation:
```bash
pip install -e ".[dev]"
```

## Configuration

### Base Configuration (base_config.yaml)
```yaml
dataset:
  path: './dataset/TUSimple'
  image_size: [800, 360]
  batch_size: 8
  num_workers: 4

model:
  name: 'LaneDetectionModel'
  num_classes: 2
  backbone: 'resnet50'
  pretrained: true

training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: 'adamw'

system:
  device: 'auto'  # 'auto', 'cpu', or 'cuda'
  precision: 'float32'
```

### CPU Configuration (cpu_config.yaml)
```yaml
system:
  device: 'cpu'
  num_threads: 4
  pin_memory: false
  
training:
  batch_size: 4  # Reduced for CPU
```

### GPU Configuration (gpu_config.yaml)
```yaml
system:
  device: 'cuda'
  cuda_devices: [0]
  pin_memory: true
  
training:
  batch_size: 16  # Increased for GPU
  
optimization:
  cudnn_benchmark: true
  mixed_precision: true
```

## Code Organization

### 1. Data Module (src/data/)
- `dataset.py`: TuSimple dataset implementation
  - Custom dataset class
  - Data loading and preprocessing
  - Augmentation pipeline
  
### 2. Models (src/models/)
- `attention.py`: Coordinate Attention implementation
  - Spatial and channel attention mechanism
  - Adaptive pooling and feature refinement
  
- `lane_detection.py`: Main model architecture
  - ResNet50 backbone
  - Coordinate Attention integration
  - Decoder with upsampling blocks
  
### 3. Training (src/training/)
- `trainer.py`: Training loop implementation
  - Epoch management
  - Loss computation
  - Optimization steps
  - Checkpoint handling
  
- `losses.py`: Loss functions
  - Dice Loss
  - IoU Loss
  - Combined Loss
  
- `metrics.py`: Performance metrics
  - Accuracy calculation
  - IoU computation
  - System metrics tracking
  
- `distributed.py`: Distributed training utilities
  - DDP wrapper
  - Model parallel wrapper
  - Process group management

### 4. Utils (src/utils/)
- `profiling.py`: System profiling
  - CPU/GPU utilization tracking
  - Memory usage monitoring
  - Training time profiling
  
- `visualization.py`: Visualization tools
  - Training metrics plots
  - System utilization graphs
  - Model predictions visualization

## Parallelization Techniques

### 1. Data Parallelism (DDP)
- Implementation in `distributed.py`
- Features:
  - Process group initialization
  - Gradient synchronization
  - Batch size scaling
  - Multi-GPU data distribution

### 2. Model Parallelism
- Implementation in `distributed.py`
- Features:
  - Model partitioning
  - Pipeline parallelism
  - Memory optimization
  - Cross-GPU communication

### 3. Hybrid Parallelism
- Combination of data and model parallelism
- Dynamic switching based on:
  - Model size
  - Batch size
  - Available resources

## Profiling and Benchmarking

### System Profiling (`profiling.py`)
- Metrics tracked:
  - CPU utilization
  - Memory usage
  - GPU utilization
  - Training time
  - I/O operations

### Benchmark Runner (`benchmark.py`)
- Benchmarking features:
  - Single GPU training
  - Multi-GPU DDP training
  - Model parallel training
  - CPU vs GPU comparison
  - Resource utilization analysis

## Visualization Tools

### Training Metrics (`visualization.py`)
- Plots available:
  - Loss curves
  - Accuracy metrics
  - Learning rate schedules
  - IoU progression

### System Metrics
- Visualizations:
  - Resource utilization over time
  - Training speed comparison
  - Memory usage patterns
  - GPU utilization graphs

### Model Predictions
- Visualization types:
  - Original images
  - Predicted lane markings
  - Ground truth comparison
  - Error analysis

## Training Pipeline

1. Data Loading:
   ```python
   dataset = LaneDataset(config['dataset']['path'])
   dataloader = DataLoader(dataset, batch_size=config['batch_size'])
   ```

2. Model Initialization:
   ```python
   model = LaneDetectionModel(config['model'])
   model = model.to(device)
   ```

3. Training Loop:
   ```python
   trainer = Trainer(config)
   trainer.train(model, train_loader, val_loader)
   ```

## Performance Optimization

### Resource Management
1. Memory Optimization:
   - Gradient accumulation
   - Mixed precision training
   - Memory-efficient backprop

2. CPU Optimization:
   - Thread management
   - Pinned memory
   - Efficient data loading

3. GPU Optimization:
   - CUDA graphs
   - Async data transfer
   - Kernel optimization

### Usage Examples

1. Single GPU Training:
```bash
python main.py --config configs/gpu_config.yaml
```

2. Multi-GPU Training:
```bash
python main.py --config configs/gpu_config.yaml --distributed
```

3. Run Benchmarks:
```bash
python benchmark.py --config configs/base_config.yaml --gpu-configs 1 2 4
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
