### 🚦 Lane Detection Training Framework 🚦

Welcome to the **Lane Detection Training Framework**! This repository offers an optimized approach to training lane detection models with support for **multi-CPU** and **multi-GPU** setups. Explore data loader optimizations, mixed precision training, and performance analysis to supercharge your model training. 💻⚡

---

### 📂 **Directory Structure**

#### **📁 LaneDetection_Framework**
- Core lane detection model implementation and training scripts.

#### **📁 MultiCPU_Metric_JSONs**
- Performance metrics for multi-CPU experiments in JSON format.

#### **📁 dashboard_results_dask**
- Results and analysis for Dask-based data processing.

---

### 📄 **Files Overview**

#### **📝 DDP_CPU_baseline.py**
- Baseline script for distributed data parallelism (DDP) using CPU.

#### **📝 DDP_MultiGPU.py**
- Multi-GPU training script with DDP.

#### **📝 DDP_MultiGPU_and_mixed_precision.py**
- Multi-GPU training extended with mixed precision for faster performance.

#### **📝 DDP_multicpu_analysis_metrics.py**
- Script to collect and analyze metrics for multi-CPU training.

#### **📊 Dask_config_analysis.ipynb**
- Notebook to analyze data processing using Dask.

#### **📚 Explaining_Framework_Structure**
- Documentation on framework design and functionality.

#### **📑 Final_Report_Team19.docx**
- Final report summarizing project objectives, implementation, and results.

#### **📽️ HPC Presentation.pptx**
- Presentation showcasing high-performance computing for lane detection.

#### **🛠️ LaneDetect-Model_running_and_training_script**
- Main script for running and training the lane detection model.

#### **📈 MultiCPU_Metrics_Plots.ipynb**
- Notebook for visualizing multi-CPU performance metrics.

#### **📊 dask_vs_data_loader_vs_baseline**
- Comparative analysis of Dask, DataLoader, and baseline performance.

#### **📊 dataloader_optimization_analysis.ipynb**
- Insights into optimizing DataLoader for efficient training.

---

### 🚀 **Usage Instructions**

1. **📥 Clone the Repository**
   ```bash
   git clone https://github.com/<your-repo>.git
   ```

2. **⚙️ Setup Environment**
   Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **🏃 Run Training**
   - **CPU Baseline**:
     ```bash
     python DDP_CPU_baseline.py
     ```
   - **Multi-GPU Training**:
     ```bash
     python DDP_MultiGPU.py --gpus 4
     ```

4. **📊 Analyze Metrics**
   Open notebooks to visualize performance:
   ```bash
   jupyter notebook MultiCPU_Metrics_Plots.ipynb
   ```

---

### 🌟 **Key Features**
- 🚀 **Scalable Training:** Supports multi-CPU and multi-GPU configurations for efficient processing.
- ⚡ **Mixed Precision Training:** Reduces training time while maintaining model accuracy.
- 📦 **Dask Integration:** Parallel data loading for large datasets.
- 📈 **Comprehensive Analysis:** Visualize and analyze bottlenecks to optimize training performance.

---

### 🙌 **Acknowledgments**
This project leverages the power of **PyTorch**, **CUDA**, and **Dask** to build a high-performance lane detection framework. Special thanks to all contributors for their dedication to this project! 🎉

💡 Have questions or ideas? Feel free to reach out or raise an issue! 🚀
