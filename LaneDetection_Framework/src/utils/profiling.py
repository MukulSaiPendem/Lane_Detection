import time
import psutil
import GPUtil
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import logging

@dataclass
class SystemProfile:
    """Class for system profiling data"""
    timestamp: str
    cpu_count: int
    cpu_freq: float
    cpu_percent: float
    memory_total: float
    memory_used: float
    memory_percent: float
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None

class SystemProfiler:
    """System profiling utility for tracking hardware usage"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.profiles: List[Dict] = []
        self.is_profiling = False
        self.profile_thread = None
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.profile_file = os.path.join(log_dir, 'system_profile.json')
        self.operation_file = os.path.join(log_dir, 'operation_profiles.json')
        
        logging.basicConfig(
            filename=os.path.join(log_dir, 'profiler.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_system_profile(self) -> SystemProfile:
        """Get current system profile"""
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            profile = SystemProfile(
                timestamp=datetime.now().isoformat(),
                cpu_count=psutil.cpu_count(),
                cpu_freq=cpu_freq.current if cpu_freq else 0,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_total=memory.total / (1024 ** 3),  # GB
                memory_used=memory.used / (1024 ** 3),    # GB
                memory_percent=memory.percent
            )

            # GPU metrics if available
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                profile.gpu_name = gpu.name
                profile.gpu_memory_total = gpu.memoryTotal
                profile.gpu_memory_used = gpu.memoryUsed
                profile.gpu_utilization = gpu.load * 100
            
            return profile
            
        except Exception as e:
            logging.error(f"Error getting system profile: {str(e)}")
            return SystemProfile(
                timestamp=datetime.now().isoformat(),
                cpu_count=0, cpu_freq=0, cpu_percent=0,
                memory_total=0, memory_used=0, memory_percent=0
            )

    def start_profiling(self, interval: float = 1.0):
        """Start continuous system profiling"""
        def profile_loop():
            while self.is_profiling:
                try:
                    profile = self.get_system_profile()
                    self.profiles.append(asdict(profile))
                    self._save_profiles()
                    time.sleep(interval)
                except Exception as e:
                    logging.error(f"Error in profile loop: {str(e)}")
                    time.sleep(interval)

        self.is_profiling = True
        self.profile_thread = threading.Thread(target=profile_loop, daemon=True)
        self.profile_thread.start()
        logging.info("Started system profiling")

    def stop_profiling(self):
        """Stop continuous system profiling"""
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join()
        self._save_profiles()
        logging.info("Stopped system profiling")

    def profile_operation(self, operation_name: str):
        """Decorator for profiling specific operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_profile = self.get_system_profile()
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in operation {operation_name}: {str(e)}")
                    raise
                finally:
                    end_time = time.time()
                    end_profile = self.get_system_profile()
                    
                    profile_data = {
                        'operation': operation_name,
                        'duration': end_time - start_time,
                        'timestamp': datetime.now().isoformat(),
                        'cpu_util_change': end_profile.cpu_percent - start_profile.cpu_percent,
                        'memory_used_change': end_profile.memory_used - start_profile.memory_used
                    }
                    
                    if end_profile.gpu_utilization is not None:
                        profile_data['gpu_util_change'] = (
                            end_profile.gpu_utilization - start_profile.gpu_utilization
                        )
                    
                    self._save_operation_profile(profile_data)
                
                return result
            return wrapper
        return decorator

    def _save_profiles(self):
        """Save system profiles to file"""
        try:
            with open(self.profile_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving profiles: {str(e)}")

    def _save_operation_profile(self, profile_data: Dict[str, Any]):
        """Save operation profile"""
        try:
            # Load existing profiles
            try:
                with open(self.operation_file, 'r') as f:
                    profiles = json.load(f)
            except FileNotFoundError:
                profiles = []
            
            # Add new profile
            profiles.append(profile_data)
            
            # Save updated profiles
            with open(self.operation_file, 'w') as f:
                json.dump(profiles, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving operation profile: {str(e)}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report"""
        if not self.profiles:
            return {}
        
        try:
            # Time statistics
            start_time = datetime.fromisoformat(self.profiles[0]['timestamp'])
            end_time = datetime.fromisoformat(self.profiles[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
            
            # Resource statistics
            cpu_percentages = [p['cpu_percent'] for p in self.profiles]
            memory_percentages = [p['memory_percent'] for p in self.profiles]
            
            report = {
                'duration': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'seconds': duration
                },
                'cpu': {
                    'average_utilization': sum(cpu_percentages) / len(cpu_percentages),
                    'max_utilization': max(cpu_percentages),
                    'min_utilization': min(cpu_percentages)
                },
                'memory': {
                    'average_percent': sum(memory_percentages) / len(memory_percentages),
                    'max_percent': max(memory_percentages),
                    'min_percent': min(memory_percentages)
                }
            }
            
            # Add GPU statistics if available
            if self.profiles[0].get('gpu_utilization') is not None:
                gpu_percentages = [p['gpu_utilization'] for p in self.profiles]
                report['gpu'] = {
                    'name': self.profiles[0]['gpu_name'],
                    'average_utilization': sum(gpu_percentages) / len(gpu_percentages),
                    'max_utilization': max(gpu_percentages),
                    'min_utilization': min(gpu_percentages)
                }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return {}