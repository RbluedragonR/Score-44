#!/usr/bin/env python3
"""
Pre-build TensorRT engines for all models and input sizes.
This script should be run once to create all engine files, avoiding runtime export delays.
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger
from ultralytics import YOLO

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager

# Input sizes to build engines for
INPUT_SIZES = [
    (1280, 720),   # Standard 16:9
    (1280, 1280),  # Square
    (1920, 1080),  # Full HD
    (1920, 1920),  # Large square
]

# Model names
MODELS = ["player", "pitch"]

def build_engines_for_size(input_size: tuple, device: str = "cuda"):
    """Build engines for all models at a specific input size."""
    width, height = input_size
    logger.info(f"Building engines for input size {width}x{height}")
    
    model_manager = ModelManager(device=device, input_size=input_size)
    
    for model_name in MODELS:
        try:
            logger.info(f"Building engine for {model_name} model...")
            start_time = time.time()
            
            # This will trigger engine export if it doesn't exist
            model = model_manager.load_model(model_name)
            
            build_time = time.time() - start_time
            logger.info(f"✅ {model_name} engine built in {build_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to build {model_name} engine: {e}")
    
    # Clear cache after building
    model_manager.clear_cache()

def main():
    """Main function to build all engines."""
    logger.info("🚀 Starting TensorRT engine pre-build process")
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot build TensorRT engines.")
        return
    
    total_start = time.time()
    
    for input_size in INPUT_SIZES:
        logger.info(f"\n--- Building engines for {input_size[0]}x{input_size[1]} ---")
        build_engines_for_size(input_size)
    
    total_time = time.time() - total_start
    logger.info(f"\n🎉 All engines built successfully in {total_time:.2f}s")
    logger.info("Engine files are now ready for fast loading!")

if __name__ == "__main__":
    main() 