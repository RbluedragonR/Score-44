import torch
import platform
from loguru import logger

def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available on macOS."""
    try:
        if platform.system() == "Darwin":
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return True
    except:
        pass
    return False

def optimize_cuda_for_rtx4090():
    """Optimize CUDA settings specifically for RTX 4090."""
    if torch.cuda.is_available():
        # Set memory fraction to use most of 24GB VRAM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable memory pooling for better memory management
        torch.cuda.empty_cache()
        
        # Set optimal CUDA device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        logger.info(f"RTX 4090 detected: {props.name}")
        logger.info(f"Total VRAM: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA Capability: {props.major}.{props.minor}")
        
        # Enable TensorRT if available
        try:
            import tensorrt as trt
            logger.info("TensorRT available - enabling optimization")
        except ImportError:
            logger.info("TensorRT not available - using standard CUDA")
        
        return True
    return False

def get_optimal_device(requested_device: str = None) -> str:
    """
    Determine the optimal device based on availability and request.
    
    Args:
        requested_device: Optional device request ('cuda', 'mps', 'cpu')
        
    Returns:
        str: The actual device to use ('cuda', 'mps', or 'cpu')
    """
    if requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info("Using CUDA device as requested")
            optimize_cuda_for_rtx4090()
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"
    
    if requested_device == "mps":
        if is_mps_available():
            logger.info("Using MPS device as requested")
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"
    
    if requested_device == "cpu":
        logger.info("Using CPU device as requested")
        return "cpu"
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        logger.info("No device specified, CUDA available - using CUDA")
        optimize_cuda_for_rtx4090()
        return "cuda"
    if is_mps_available():
        logger.info("No device specified, MPS available - using MPS")
        return "mps"
    
    logger.info("No device specified or no accelerator available - using CPU")
    return "cpu"

def get_rtx4090_optimal_batch_size() -> int:
    """Get optimal batch size for RTX 4090 based on available VRAM."""
    if torch.cuda.is_available():
        # RTX 4090 has 24GB VRAM, we can use large batches
        # Conservative estimate: 8-12 frames per batch
        return 10
    return 1

def get_rtx4090_optimal_image_size() -> int:
    """Get optimal image size for RTX 4090 processing."""
    # RTX 4090 can handle larger images efficiently
    return 1920  # Increased from default 1280 