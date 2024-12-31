import torch
import platform
import os
import psutil
import sys

def check_system_config():
    # System Information
    print("## System Information")
    print(f"OS: {platform.system()} {platform.version()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # CPU Information
    print("\n## CPU Information")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"CPU Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Available Memory: {round(psutil.virtual_memory().total / (1024.0 ** 3), 2)} GB")
    
    # CUDA Information
    print("\n## CUDA Information")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # GPU Information
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Information:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Total: {round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)} GB")
            
    # Environment Information
    print("\n## Environment Information")
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    print(f"Conda Environment: {conda_env if conda_env else 'Not in a conda environment'}")
    
    # Default PyTorch Device
    print("\n## PyTorch Configuration")
    print(f"Default Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"Current GPU Device: cuda:{torch.cuda.current_device()}")
        print(f"TF32 Allowed: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

if __name__ == "__main__":
    check_system_config()
