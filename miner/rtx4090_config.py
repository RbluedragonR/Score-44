"""
RTX 4090 Optimized Configuration for Score Vision Mining
Optimized for 24GB VRAM and maximum performance
"""

import os
from typing import Dict, Any

# RTX 4090 Hardware Configuration
RTX4090_CONFIG = {
    # Device settings
    "device": "cuda",
    "gpu_memory_fraction": 0.95,  # Use 95% of 24GB VRAM
    
    # Batch processing optimization
    "batch_size": 10,  # Optimal batch size for RTX 4090
    "image_size": 1920,  # Larger image size for better accuracy
    
    # Model optimization
    "enable_tensorrt": True,
    "mixed_precision": True,
    "model_fusion": True,
    
    # Processing parameters
    "target_fps": 15,  # Target processing FPS
    "max_frames": 500,  # Maximum frames to process
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    
    # Memory management
    "enable_memory_cleanup": True,
    "cleanup_interval": 100,  # Cleanup every 100 frames
    "preallocate_memory": True,
    
    # Timeout settings (RTX 4090 can handle longer videos)
    "cuda_timeout": 10800.0,  # 3 hours
    "mps_timeout": 10800.0,
    "cpu_timeout": 10800.0,
    
    # Response optimization
    "enable_compression": True,
    "compression_threshold": 1024,  # 1KB
    "coordinate_precision": 2,  # Decimal places for coordinates
}

def get_rtx4090_config() -> Dict[str, Any]:
    """Get RTX 4090 optimized configuration."""
    return RTX4090_CONFIG.copy()

def get_environment_config() -> Dict[str, Any]:
    """Get configuration from environment variables with RTX 4090 defaults."""
    config = RTX4090_CONFIG.copy()
    
    # Override with environment variables if set
    config["device"] = os.getenv("DEVICE", config["device"])
    config["batch_size"] = int(os.getenv("BATCH_SIZE", config["batch_size"]))
    config["image_size"] = int(os.getenv("IMAGE_SIZE", config["image_size"]))
    config["target_fps"] = int(os.getenv("TARGET_FPS", config["target_fps"]))
    config["max_frames"] = int(os.getenv("MAX_FRAMES", config["max_frames"]))
    
    return config

# CUDA optimization functions
def optimize_cuda_settings():
    """Apply CUDA optimizations for RTX 4090."""
    import torch
    
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(RTX4090_CONFIG["gpu_memory_fraction"])
        
        # Enable memory pooling
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set optimal CUDA device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        print(f"RTX 4090 Configuration Applied:")
        print(f"  Device: {props.name}")
        print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Memory Fraction: {RTX4090_CONFIG['gpu_memory_fraction']}")
        print(f"  Batch Size: {RTX4090_CONFIG['batch_size']}")
        print(f"  Image Size: {RTX4090_CONFIG['image_size']}")
        
        return True
    return False

# Performance monitoring
class RTX4090PerformanceMonitor:
    """Monitor RTX 4090 performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        import time
        import psutil
        import GPUtil
        
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        
        if torch.cuda.is_available():
            self.start_gpu_memory = GPUtil.getGPUs()[0].memoryUsed
    
    def end_monitoring(self, frames_processed: int):
        """End monitoring and return metrics."""
        import time
        import psutil
        import GPUtil
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        processing_time = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        
        metrics = {
            "processing_time": processing_time,
            "frames_processed": frames_processed,
            "fps": frames_processed / processing_time if processing_time > 0 else 0,
            "memory_used_mb": memory_used / 1024 / 1024,
        }
        
        if torch.cuda.is_available():
            end_gpu_memory = GPUtil.getGPUs()[0].memoryUsed
            metrics["gpu_memory_used_mb"] = end_gpu_memory - self.start_gpu_memory
        
        self.metrics = metrics
        return metrics
    
    def log_metrics(self):
        """Log performance metrics."""
        print(f"RTX 4090 Performance Metrics:")
        print(f"  Processing Time: {self.metrics.get('processing_time', 0):.2f}s")
        print(f"  Frames Processed: {self.metrics.get('frames_processed', 0)}")
        print(f"  FPS: {self.metrics.get('fps', 0):.2f}")
        print(f"  Memory Used: {self.metrics.get('memory_used_mb', 0):.1f} MB")
        if "gpu_memory_used_mb" in self.metrics:
            print(f"  GPU Memory Used: {self.metrics['gpu_memory_used_mb']:.1f} MB") 