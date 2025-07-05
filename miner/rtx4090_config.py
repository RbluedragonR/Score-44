"""
RTX 4090 Optimized Configuration for Score Vision Mining
Optimized for 24GB VRAM and maximum performance
"""

import os
from typing import Dict, Any

# RTX 4090 Hardware Configuration - SPEED OPTIMIZED
RTX4090_CONFIG = {
    # Device settings
    "device": "cuda",
    "gpu_memory_fraction": 0.98,  # Use 98% of 24GB VRAM for maximum throughput
    
    # Batch processing optimization - INCREASED FOR SPEED
    "batch_size": 16,  # Increased from 10 to 16 for better GPU utilization
    "image_size": 1280,  # Reduced from 1920 to 1280 for speed (still high quality)
    
    # Model optimization
    "enable_tensorrt": True,
    "mixed_precision": True,
    "model_fusion": True,
    
    # Processing parameters - OPTIMIZED FOR SPEED
    "target_fps": 20,  # Increased target FPS
    "max_frames": 500,  # Maximum frames to process
    "confidence_threshold": 0.2,  # Slightly lower for faster processing
    "iou_threshold": 0.4,  # Slightly lower for faster NMS
    
    # Memory management
    "enable_memory_cleanup": True,
    "cleanup_interval": 50,  # More frequent cleanup for better memory management
    "preallocate_memory": True,
    
    # Timeout settings (RTX 4090 can handle longer videos)
    "cuda_timeout": 10800.0,  # 3 hours
    "mps_timeout": 10800.0,
    "cpu_timeout": 10800.0,
    
    # Response optimization
    "enable_compression": True,
    "compression_threshold": 512,  # Reduced compression threshold for faster response
    "coordinate_precision": 1,  # Reduced precision for smaller response size
    
    # Speed optimizations
    "enable_parallel_processing": True,
    "num_workers": 4,  # Parallel workers for data loading
    "pin_memory": True,  # Pin memory for faster GPU transfer
    "prefetch_factor": 2,  # Prefetch factor for data loading
}

# SPEED MODE CONFIGURATION
SPEED_MODE_CONFIG = {
    # Ultra-fast mode for maximum speed
    "batch_size": 20,  # Maximum batch size for RTX 4090
    "image_size": 1024,  # Smaller image size for speed
    "confidence_threshold": 0.15,  # Lower threshold for faster detection
    "iou_threshold": 0.35,  # Lower NMS threshold
    "target_fps": 25,  # Target 25 FPS
    "enable_quality_tradeoff": True,  # Trade some quality for speed
}

# QUALITY MODE CONFIGURATION (for when speed allows)
QUALITY_MODE_CONFIG = {
    # High quality mode when speed is sufficient
    "batch_size": 12,  # Balanced batch size
    "image_size": 1920,  # Full resolution for maximum quality
    "confidence_threshold": 0.25,  # Higher threshold for quality
    "iou_threshold": 0.45,  # Higher NMS threshold
    "target_fps": 15,  # Target 15 FPS
    "enable_quality_tradeoff": False,  # Prioritize quality
}

def get_rtx4090_config() -> Dict[str, Any]:
    """Get RTX 4090 optimized configuration."""
    return RTX4090_CONFIG.copy()

def get_speed_mode_config() -> Dict[str, Any]:
    """Get speed-optimized configuration for maximum throughput."""
    config = RTX4090_CONFIG.copy()
    config.update(SPEED_MODE_CONFIG)
    return config

def get_quality_mode_config() -> Dict[str, Any]:
    """Get quality-optimized configuration for maximum accuracy."""
    config = RTX4090_CONFIG.copy()
    config.update(QUALITY_MODE_CONFIG)
    return config

def get_adaptive_config(target_time: float = 12.0) -> Dict[str, Any]:
    """
    Get adaptive configuration based on target processing time.
    
    Args:
        target_time: Target processing time in seconds (default: 12s for safety margin)
    
    Returns:
        Optimized configuration
    """
    if target_time <= 10.0:
        # Ultra-fast mode
        return get_speed_mode_config()
    elif target_time <= 13.0:
        # Balanced mode
        return get_rtx4090_config()
    else:
        # Quality mode
        return get_quality_mode_config()

def get_environment_config() -> Dict[str, Any]:
    """Get configuration from environment variables with RTX 4090 defaults."""
    config = RTX4090_CONFIG.copy()
    
    # Override with environment variables if set
    config["device"] = os.getenv("DEVICE", config["device"])
    config["batch_size"] = int(os.getenv("BATCH_SIZE", config["batch_size"]))
    config["image_size"] = int(os.getenv("IMAGE_SIZE", config["image_size"]))
    config["target_fps"] = int(os.getenv("TARGET_FPS", config["target_fps"]))
    config["max_frames"] = int(os.getenv("MAX_FRAMES", config["max_frames"]))
    
    # Speed mode override
    if os.getenv("SPEED_MODE", "false").lower() == "true":
        config.update(SPEED_MODE_CONFIG)
    
    return config

# CUDA optimization functions
def optimize_cuda_settings():
    """Apply CUDA optimizations for RTX 4090."""
    import torch
    
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(RTX4090_CONFIG["gpu_memory_fraction"])
        
        # Enable memory pooling and optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set optimal CUDA device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        print(f"RTX 4090 Configuration Applied:")
        print(f"  Device: {props.name}")
        print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Memory Fraction: {RTX4090_CONFIG['gpu_memory_fraction']}")
        print(f"  Batch Size: {RTX4090_CONFIG['batch_size']}")
        print(f"  Image Size: {RTX4090_CONFIG['image_size']}")
        print(f"  Target FPS: {RTX4090_CONFIG['target_fps']}")
        
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
            "time_margin": 15.0 - processing_time,  # Margin from 15s limit
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
        print(f"  Time Margin: {self.metrics.get('time_margin', 0):.2f}s")
        print(f"  Frames Processed: {self.metrics.get('frames_processed', 0)}")
        print(f"  FPS: {self.metrics.get('fps', 0):.2f}")
        print(f"  Memory Used: {self.metrics.get('memory_used_mb', 0):.1f} MB")
        if "gpu_memory_used_mb" in self.metrics:
            print(f"  GPU Memory Used: {self.metrics['gpu_memory_used_mb']:.1f} MB")
        
        # Performance assessment
        time_margin = self.metrics.get('time_margin', 0)
        if time_margin < 1.0:
            print(f"  ⚠️  WARNING: Low time margin ({time_margin:.2f}s)")
        elif time_margin < 3.0:
            print(f"  ⚡ GOOD: Adequate time margin ({time_margin:.2f}s)")
        else:
            print(f"  🚀 EXCELLENT: High time margin ({time_margin:.2f}s)") 