from src.utils.dask_utils import DaskProcessor
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class DaskBenchmarkAnalyzer:
    """Benchmark analysis using Dask"""
    
    def __init__(self, n_workers: int = 4):
        self.dask_processor = DaskProcessor(n_workers=n_workers)
        
    def analyze_training_performance(self, 
                                   benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze training performance metrics using Dask
        
        Args:
            benchmark_results: List of benchmark result dictionaries
            
        Returns:
            Dictionary containing analyzed metrics
        """
        # Convert results to format suitable for Dask
        metrics_data = []
        for result in benchmark_results:
            metrics = {
                'config': result['name'],
                'total_time': result['time_taken'],
                'memory_used': result['memory_used'],
                'cpu_utilization': result['cpu_utilization']
            }
            if 'gpu_utilization' in result:
                metrics['gpu_utilization'] = result['gpu_utilization']
            metrics_data.append(metrics)
        
        # Perform analysis using Dask
        analysis = self.dask_processor.parallel_performance_analysis(metrics_data)
        
        return {
            'performance_summary': analysis,
            'config_comparison': self._compare_configurations(metrics_data),
            'resource_efficiency': self._analyze_resource_efficiency(metrics_data)
        }
    
    def _compare_configurations(self, metrics_data: List[dict]) -> Dict[str, Any]:
        """Compare different training configurations"""
        df = pd.DataFrame(metrics_data)
        
        # Group by configuration
        grouped = df.groupby('config').agg({
            'total_time': ['mean', 'std'],
            'memory_used': ['mean', 'max'],
            'cpu_utilization': ['mean', 'max']
        })
        
        # Calculate speedup relative to CPU-only
        if 'cpu' in df['config'].values:
            cpu_time = df[df['config'] == 'cpu']['total_time'].iloc[0]
            speedups = {
                config: cpu_time / time 
                for config, time in df.groupby('config')['total_time'].mean().items()
            }
        else:
            speedups = {}
        
        return {
            'metrics_by_config': grouped.to_dict(),
            'speedups': speedups
        }
    
    def _analyze_resource_efficiency(self, metrics_data: List[dict]) -> Dict[str, Any]:
        """Analyze resource efficiency of different configurations"""
        df = pd.DataFrame(metrics_data)
        
        efficiency_metrics = {}
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            # Calculate efficiency metrics
            time_per_unit = config_data['total_time'] / config_data['memory_used']
            cpu_efficiency = config_data['total_time'] / config_data['cpu_utilization']
            
            efficiency_metrics[config] = {
                'time_per_memory_unit': float(time_per_unit.mean()),
                'cpu_efficiency': float(cpu_efficiency.mean())
            }
            
            # Add GPU efficiency if available
            if 'gpu_utilization' in config_data.columns:
                gpu_efficiency = config_data['total_time'] / config_data['gpu_utilization']
                efficiency_metrics[config]['gpu_efficiency'] = float(gpu_efficiency.mean())
        
        return efficiency_metrics
    
    def generate_report(self, 
                       benchmark_results: List[Dict[str, Any]],
                       output_file: str):
        """
        Generate comprehensive benchmark report
        
        Args:
            benchmark_results: List of benchmark results
            output_file: Path to save the report
        """
        analysis = self.analyze_training_performance(benchmark_results)
        
        # Create report content
        report = [
            "# Benchmark Analysis Report\n",
            "\n## Performance Summary\n",
            f"- Mean CPU Utilization: {analysis['performance_summary']['mean_cpu_util']:.2f}%\n",
            f"- Mean Memory Utilization: {analysis['performance_summary']['mean_memory_util']:.2f}%\n",
            
            "\n## Configuration Comparison\n",
            "| Configuration | Mean Time (s) | Speedup | Peak Memory (GB) |\n",
            "|--------------|---------------|---------|------------------|\n"
        ]
        
        # Add configuration comparisons
        for config, metrics in analysis['config_comparison']['metrics_by_config'].items():
            speedup = analysis['config_comparison']['speedups'].get(config, 1.0)
            report.append(
                f"| {config} | {metrics['total_time']['mean']:.2f} | {speedup:.2f}x | "
                f"{metrics['memory_used']['max']:.2f} |\n"
            )
        
        # Add resource efficiency analysis
        report.extend([
            "\n## Resource Efficiency\n",
            "| Configuration | Time/Memory (s/GB) | CPU Efficiency | GPU Efficiency |\n",
            "|--------------|-------------------|----------------|----------------|\n"
        ])
        
        for config, metrics in analysis['resource_efficiency'].items():
            gpu_eff = f"{metrics.get('gpu_efficiency', '-'):.2f}" if 'gpu_efficiency' in metrics else '-'
            report.append(
                f"| {config} | {metrics['time_per_memory_unit']:.2f} | "
                f"{metrics['cpu_efficiency']:.2f} | {gpu_eff} |\n"
            )
        
        # Save report
        with open(output_file, 'w') as f:
            f.writelines(report)
        
        logging.info(f"Benchmark report saved to {output_file}")
    
    def cleanup(self):
        """Cleanup Dask resources"""
        self.dask_processor.cleanup()